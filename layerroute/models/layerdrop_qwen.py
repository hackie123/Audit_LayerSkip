"""
models/layerdrop_qwen.py
=========================
LayerDrop (Fan et al., 2020) baseline, matched to GatedQwenLoRA's training
budget (same LoRA adapters, same optimizer/steps) so the comparison isolates
ONE variable: a LEARNED, INPUT-ADAPTIVE gate (LayerRoute) vs. a FIXED,
INPUT-INDEPENDENT structural pruning pattern discovered via stochastic
dropout training (LayerDrop).

Training: each layer is stochastically dropped with probability
`layerdrop_rate` per forward pass (Bernoulli, same rate for every layer,
matching the original paper's uniform-rate variant), training the backbone
+ LoRA to be robust to ANY layer being removed. No router, no learned
per-input decision -- this is the defining architectural difference from
LayerRoute.

Inference: a FIXED set of layers is pruned (every Nth layer, chosen to hit
a target skip ratio comparable to LayerRoute's observed skip rates), applied
IDENTICALLY to every input. skip_pct is therefore constant across inputs,
unlike LayerRoute's per-input-varying skip_pct.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from typing import Optional, Tuple

from models.lora import apply_lora_to_model
from utils.config import SystemConfig, QWEN_SPEC


class LayerDropQwenLoRA(nn.Module):

    def __init__(self, hf_model, cfg: SystemConfig):
        super().__init__()
        self.cfg      = cfg
        self.hf_model = hf_model
        self.n_layers = QWEN_SPEC["n_layers"]
        self.hidden   = QWEN_SPEC["hidden"]

        for p in self.hf_model.parameters():
            p.requires_grad = False
        n_frozen = sum(p.numel() for p in self.hf_model.parameters())
        print(f"  \u2713 Frozen {n_frozen:,} backbone parameters")

        n_replaced = apply_lora_to_model(
            self.hf_model, cfg, target_modules=cfg.lora.target_modules
        )
        n_lora = sum(p.numel() for n, p in self.hf_model.named_parameters() if p.requires_grad)
        print(f"  \u2713 LoRA applied to {n_replaced} projections \u2014 {n_lora:,} trainable params")

        # --- LayerDrop-specific config (read from cfg.layerdrop, added below) ---
        self.layerdrop_rate = getattr(cfg, "layerdrop", None) and cfg.layerdrop.rate or 0.2
        target_skip = getattr(cfg, "layerdrop", None) and cfg.layerdrop.inference_skip_ratio or 0.25
        # Fixed inference-time pruning pattern: drop every Nth layer (N chosen
        # from target_skip), excluding the first and last layer (never pruned,
        # matching LayerRoute's convention of always keeping layers 0 and L-1
        # for a fair comparison of the MIDDLE-layer redundancy hypothesis).
        prunable = list(range(1, self.n_layers - 1))
        n_to_drop = max(1, round(len(prunable) * target_skip))
        step = max(1, len(prunable) // n_to_drop)
        self._inference_drop_set = set(prunable[::step][:n_to_drop])
        print(f"  \u2713 LayerDrop: train-time rate={self.layerdrop_rate}, "
             f"inference-time fixed drop set ({len(self._inference_drop_set)}/{self.n_layers} layers): "
             f"{sorted(self._inference_drop_set)}")

        device = next(self.hf_model.parameters()).device
        self._device = device

    @classmethod
    def from_pretrained(cls, cfg: SystemConfig) -> "LayerDropQwenLoRA":
        hf_name   = QWEN_SPEC["hf_name"]
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dtype = dtype_map[cfg.training.dtype]
        print(f"  Loading '{hf_name}'...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            hf_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        )
        print(f"  \u2713 Pretrained weights loaded")
        return cls(hf_model, cfg)

    def _forward_layers(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        transformer = self.hf_model.model
        device      = input_ids.device
        T           = input_ids.shape[1]

        hidden              = transformer.embed_tokens(input_ids)
        model_dtype         = hidden.dtype
        position_ids        = torch.arange(T, device=device).unsqueeze(0)
        cos, sin            = transformer.rotary_emb(hidden, position_ids)
        position_embeddings = (cos, sin)

        layer_kwargs = dict(
            attention_mask=None, position_ids=position_ids,
            past_key_values=None, use_cache=False,
            position_embeddings=position_embeddings,
        )

        gate_values = []
        layers_run  = 0

        for i, layer in enumerate(transformer.layers):
            if self.training:
                # Stochastic drop: Bernoulli(1 - layerdrop_rate) keep-probability,
                # first/last layer never dropped (matches LayerRoute's convention).
                if 0 < i < self.n_layers - 1 and torch.rand(1).item() < self.layerdrop_rate:
                    gate_values.append(0.0)
                    continue
                gate_values.append(1.0)
                layers_run += 1
                out = layer(hidden, **layer_kwargs)
                hidden = (out[0] if isinstance(out, tuple) else out).to(model_dtype)
            else:
                # Fixed, input-independent pruning pattern (no adaptivity).
                if i in self._inference_drop_set:
                    gate_values.append(0.0)
                    continue
                gate_values.append(1.0)
                layers_run += 1
                out = layer(hidden, **layer_kwargs)
                hidden = (out[0] if isinstance(out, tuple) else out).to(model_dtype)

        hidden = transformer.norm(hidden)
        logits = self.hf_model.lm_head(hidden)

        return logits, {
            "gate_values": gate_values,
            "layers_run": layers_run,
            "skip_pct": round(100 * (self.n_layers - layers_run) / self.n_layers, 1),
        }

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        logits, gate_stats = self._forward_layers(input_ids)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, QWEN_SPEC["vocab"]),
                shift_labels.view(-1), ignore_index=-100,
            )
        return logits, loss, gate_stats

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save_adapters(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {k: v for k, v in self.state_dict().items() if "lora_A" in k or "lora_B" in k}
        torch.save(state, path)
        print(f"  \u2713 Adapters saved \u2192 {path}  ({len(state)} tensors)")

    def load_adapters(self, path: str):
        state = torch.load(path, map_location="cpu", weights_only=True)
        missing, unexpected = self.load_state_dict(state, strict=False)
        print(f"  \u2713 Adapters loaded \u2190 {path}")
        if unexpected:
            print(f"  \u26a0 Unexpected keys: {unexpected[:3]}")

    def get_skip_layers(self):
        """Matches ConfLayers/SWIFT's naming convention for audit-harness compatibility."""
        return sorted(self._inference_drop_set)

    def set_skip_layers(self, *args, **kwargs):
        pass  # LayerDrop's inference pattern is fixed at construction time, not settable.
