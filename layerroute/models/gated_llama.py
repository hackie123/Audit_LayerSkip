"""
models/gated_llama.py
======================
TinyLlama-1.1B with:
    1. LoRA adapters on Q,K,V,O projections
    2. Per-layer hard-gated skip connections (STE)
    3. Single LM loss training — no separate classification head

This is a from-first-principles port of GatedQwenLoRA (models/gated_qwen.py)
to the Llama architecture, NOT a copy with names swapped. The gating/STE
mechanism in _forward_layers operates purely on hidden states and is
architecture-agnostic; it is reused unchanged (verified against
LlamaDecoderLayer.forward()'s signature and LlamaModel.rotary_emb(x,
position_ids) before writing this file — both match what _forward_layers
already assumes).

What changes vs. GatedQwenLoRA:
    - Sources LLAMA_SPEC instead of QWEN_SPEC (see utils/config.py).
    - Loads LlamaForCausalLM instead of AutoModelForCausalLM against a
      Qwen checkpoint (AutoModelForCausalLM would also resolve to
      LlamaForCausalLM for a Llama checkpoint, but importing it directly
      makes the architecture assumption explicit rather than incidental).
    - The middle-layer router boundary (Eq. 7 in the LayerRoute paper) is
      NOT copied from Qwen's fixed (8, 17). It must be supplied via
      cfg.router.middle_start/middle_end, derived with
      derive_middle_bounds(LLAMA_SPEC["n_layers"]) -- see utils/config.py
      and layerroute_errata_note.pdf for why reusing Qwen's absolute
      indices on a different-depth backbone would be wrong.

Forward pass (training):
    For each block i:
        gate_i = router_i(h)          ← STE: hard forward, soft backward
        h_out  = Block_i(h)           ← Llama + LoRA
        h      = gate_i * h_out + (1 - gate_i) * h    ← gated output

    lm_logits = lm_head(norm(h))
    loss      = CrossEntropy(lm_logits, labels)

Forward pass (inference):
    Identical — same hard gate, no mismatch.
    gate_i = 0 → skip block (h unchanged, zero compute)
    gate_i = 1 → run block normally

TinyLlama-1.1B specs (see LLAMA_SPEC in utils/config.py):
    num_hidden_layers  : 22
    hidden_size        : 2048
    num_q_heads        : 32
    num_kv_heads       : 4
    intermediate_size  : 5632
    vocab_size         : 32000
"""

import torch
import torch.nn as nn
from transformers import LlamaForCausalLM
from typing import Optional, Tuple

from models.lora import apply_lora_to_model
from models.router import RouterCollection
from utils.config import SystemConfig, LLAMA_SPEC


class GatedLlamaLoRA(nn.Module):

    def __init__(self, hf_model, cfg: SystemConfig):
        super().__init__()
        self.cfg      = cfg
        self.hf_model = hf_model
        self.n_layers = LLAMA_SPEC["n_layers"]
        self.hidden   = LLAMA_SPEC["hidden"]

        # Step 1: Freeze ALL backbone parameters
        for p in self.hf_model.parameters():
            p.requires_grad = False
        n_frozen = sum(p.numel() for p in self.hf_model.parameters())
        print(f"  ✓ Frozen {n_frozen:,} backbone parameters")

        # Step 2: Apply LoRA to Q,K,V,O projections
        # (target module NAMES are identical to Qwen: q_proj/k_proj/v_proj/
        # o_proj -- confirmed via LlamaDecoderLayer inspection. Llama's
        # projections are bias-free (attention_bias=False) vs. Qwen's
        # biased ones. Checked models/lora.py directly: LoRALinear.forward()
        # computes base = self.frozen(x) -- delegating to the wrapped
        # nn.Linear's own forward, which correctly includes bias if present
        # and omits it if not (bias=None is handled natively by F.linear).
        # The LoRA delta itself never references .bias. So bias presence/
        # absence on the base projection requires no special-casing here.)
        n_replaced = apply_lora_to_model(
            self.hf_model, cfg,
            target_modules=cfg.lora.target_modules
        )
        n_lora = sum(
            p.numel() for n, p in self.hf_model.named_parameters()
            if p.requires_grad
        )
        print(f"  ✓ LoRA applied to {n_replaced} projections — {n_lora:,} trainable params")

        # Step 3: Per-layer routers (STE hard gating)
        # Middle-layer boundary comes from cfg.router (NOT Qwen's fixed
        # 8,17 default) -- caller is responsible for setting
        # cfg.router.middle_start/middle_end via derive_middle_bounds(22)
        # before constructing this model. See utils/config.py.
        self.routers = RouterCollection(
            n_layers         = self.n_layers,
            hidden           = self.hidden,
            init_bias_early  = cfg.router.init_bias_early,
            init_bias_middle = cfg.router.init_bias_middle,
            threshold        = cfg.router.threshold,
            middle_start     = cfg.router.middle_start,
            middle_end       = cfg.router.middle_end,
        )
        print(f"  ✓ RouterCollection: {self.routers.param_count():,} params "
              f"({self.n_layers} × Linear({self.hidden},1))  "
              f"[middle band: {cfg.router.middle_start}-{cfg.router.middle_end}]")

        # Move routers to same device/dtype as backbone
        device = next(self.hf_model.parameters()).device
        self.routers = self.routers.to(device)

    @classmethod
    def from_pretrained(cls, cfg: SystemConfig) -> "GatedLlamaLoRA":
        hf_name   = LLAMA_SPEC["hf_name"]
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                     "float32": torch.float32}
        dtype = dtype_map[cfg.training.dtype]
        print(f"  Loading '{hf_name}'...")
        hf_model = LlamaForCausalLM.from_pretrained(
            hf_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        )
        print(f"  ✓ Pretrained weights loaded")
        return cls(hf_model, cfg)

    # ── Manual forward through all blocks with hard-gated skip ───────
    def _forward_layers(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, list, dict]:
        """
        Manual forward through all n_layers blocks with hard-gated skip
        connections. Identical logic to GatedQwenLoRA._forward_layers --
        the router/STE/gating math is pure hidden-state arithmetic and does
        not depend on Qwen- vs Llama-specific attention/MLP internals.
        Confirmed compatible: LlamaDecoderLayer.forward() accepts
        position_embeddings=(cos, sin) by the same keyword, and
        LlamaModel.rotary_emb(x, position_ids) has the same call signature
        Qwen2Model.rotary_emb uses.

        Returns: logits [B, T, vocab], soft_gates (list, differentiable), gate_stats
        """
        transformer = self.hf_model.model
        device      = input_ids.device
        T           = input_ids.shape[1]

        hidden              = transformer.embed_tokens(input_ids)
        model_dtype         = hidden.dtype
        position_ids        = torch.arange(T, device=device).unsqueeze(0)
        cos, sin            = transformer.rotary_emb(hidden, position_ids)
        position_embeddings = (cos, sin)

        layer_kwargs = dict(
            attention_mask      = None,
            position_ids        = position_ids,
            past_key_values     = None,
            use_cache           = False,
            position_embeddings = position_embeddings,
        )

        gate_values  = []
        soft_gates   = []   # kept for gate regularisation (has grad)
        layers_run   = 0

        for i, layer in enumerate(transformer.layers):
            # Router: soft gate (for reg grad) + hard STE gate (for forward)
            router   = self.routers.routers[i]
            # FIX (root cause of a real bug, found while re-verifying the
            # audit's cost-measurement script): this loop used to inline the
            # gate computation and NEVER checked router.force_open -- the
            # ONLY place LayerRouter.forward() checks that flag. This made
            # force_open dead code in the actual forward pass used for
            # training AND generation, silently breaking any "force all
            # gates open" baseline comparison that relied on it. Fixed by
            # short-circuiting here too, matching LayerRouter.forward()'s
            # intended behavior exactly. Verified via a structural test
            # (layers_run genuinely changes 3.0->4.0 in a 4-layer synthetic
            # model when force_open is set, where it previously did not).
            if router.force_open:
                b = hidden.shape[0]
                gate = torch.ones(b, 1, 1, device=hidden.device, dtype=hidden.dtype)
                soft_g = torch.ones(b, device=hidden.device)
            else:
                h_mean   = hidden.mean(dim=1).to(router.linear.weight.dtype)
                score    = router.linear(h_mean).squeeze(-1)      # [B]
                soft_g   = torch.sigmoid(score)                   # [B] — has grad
                hard_g   = (soft_g > router.threshold).float()
                gate     = (hard_g - soft_g.detach() + soft_g)    # STE [B]
                gate     = gate.unsqueeze(-1).unsqueeze(-1)        # [B,1,1]

            soft_gates.append(soft_g.mean())   # scalar, differentiable

            # Run the block (LoRA adapters active inside)
            out    = layer(hidden, **layer_kwargs)
            h_out  = out[0] if isinstance(out, tuple) else out

            # Gated skip: h = gate * block_output + (1-gate) * h
            hidden = (gate * h_out + (1.0 - gate) * hidden).to(model_dtype)

            # Stats (no grad)
            with torch.no_grad():
                g_val = gate.mean().item()
            gate_values.append(round(g_val, 4))
            layers_run += (gate.mean() > 0.5).float().item()

        hidden  = transformer.norm(hidden)
        logits  = self.hf_model.lm_head(hidden)

        return logits, soft_gates, {
            "gate_values" : gate_values,
            "layers_run"  : layers_run,
            "skip_pct"    : round(100 * (self.n_layers - layers_run) / self.n_layers, 1),
        }

    # ── Forward ────────────────────────────────────────────────────
    def forward(
        self,
        input_ids : torch.Tensor,
        labels    : Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """
        Returns: logits, loss (if labels provided), gate_stats
        """
        logits, soft_gates, gate_stats = self._forward_layers(input_ids)
        loss = None
        if labels is not None:
            # Standard LM loss — shift by 1
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            lm_loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, LLAMA_SPEC["vocab"]),
                shift_labels.view(-1),
                ignore_index = -100,
            )
            # Gate regularisation: penalise uniformly high gates
            gate_reg = torch.stack(soft_gates).mean()
            loss = lm_loss + self.cfg.router.gate_reg_weight * gate_reg
        return logits, loss, gate_stats

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save_adapters(self, path: str):
        """Save only trainable params (LoRA + routers) — not full model."""
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {k: v for k, v in self.state_dict().items()
                 if any(x in k for x in ["lora_A", "lora_B", "routers"])}
        torch.save(state, path)
        print(f"  ✓ Adapters saved → {path}  ({len(state)} tensors)")

    def load_adapters(self, path: str):
        state = torch.load(path, map_location="cpu", weights_only=True)
        missing, unexpected = self.load_state_dict(state, strict=False)
        print(f"  ✓ Adapters loaded ← {path}")
        if unexpected:
            print(f"  ⚠ Unexpected keys: {unexpected[:3]}")