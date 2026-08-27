"""
main.py — ConfGate v6
======================
LoRA fine-tuning + per-layer hard-gated skip connections.
Supports two backbones via --model_family:
    qwen  -> Qwen2.5-0.5B-Instruct  (24 layers, hidden 896)
    llama -> TinyLlama-1.1B-Chat-v1.0 (22 layers, hidden 2048)

Architecture:
    For each transformer block:
        router_i(h) → STE gate gᵢ ∈ {0,1}
        h = gᵢ * Block_i(h) + (1-gᵢ) * h

    Block_i contains backbone weights + LoRA adapters.
    Single LM loss — no separate classification head.

Workflow:
    python main.py --mode train --model_family qwen
    python main.py --mode train --model_family llama
    python main.py --mode demo  --model_family llama --ckpt checkpoints/best_adapters.pt

CSV outputs (checkpoints/):
    training_log.csv  — loss, ppl, layers_run, skip_pct per step
    eval_log.csv      — val_loss, val_ppl, val_layers_run
    gate_log.csv      — per-layer gate values every log_every steps
    run_summary.csv   — one row per run
"""

import argparse
import torch
import torch.nn.functional as F

from utils.config import (SystemConfig, LoRAConfig, RouterConfig,
                           TrainingConfig, DataConfig, LayerDropConfig, MoDConfig,
                           QWEN_SPEC, LLAMA_SPEC, derive_middle_bounds)
from models.gated_qwen import GatedQwenLoRA
from models.gated_llama import GatedLlamaLoRA
from models.layerdrop_qwen import LayerDropQwenLoRA
from models.mod_qwen import MoDQwenLoRA

QWEN_SCALES = {
    "0.5b": {
        "hf_name": "Qwen/Qwen2.5-0.5B-Instruct", "n_layers": 24,
        "n_q_heads": 14, "n_kv_heads": 2, "hidden": 896,
        "head_dim": 64, "intermediate": 4864, "vocab": 151936,
    },
    "1.5b": {
        "hf_name": "Qwen/Qwen2.5-1.5B-Instruct", "n_layers": 28,
        "n_q_heads": 12, "n_kv_heads": 2, "hidden": 1536,
        "head_dim": 128, "intermediate": 8960, "vocab": 151936,
    },
}


def apply_qwen_scale(scale: str):
    """
    Mutate QWEN_SPEC's dict IN-PLACE to match the chosen scale (same pattern
    already used in run_layerroute_timed.py). models/gated_qwen.py imports
    QWEN_SPEC as a reference to this SAME dict object, so in-place mutation
    here is visible there too. Only relevant when --model_family qwen.
    """
    if scale not in QWEN_SCALES:
        raise ValueError(f"Unknown --model_scale '{scale}'. Choices: {list(QWEN_SCALES)}")
    QWEN_SPEC.clear()
    QWEN_SPEC.update(QWEN_SCALES[scale])
    print(f"  [model_scale={scale}] QWEN_SPEC set to {QWEN_SPEC['hf_name']} "
         f"({QWEN_SPEC['n_layers']} layers, hidden={QWEN_SPEC['hidden']})")


def _family_spec(model_family: str, method: str = "layerroute"):
    """Returns (model_cls, spec_dict, display_name) for the selected
    (model_family, method) combination. layerdrop/mod are qwen-only for now,
    matching this audit's core 2-task x 2-scale matrix -- requesting them
    with model_family="llama" is not supported and raises explicitly rather
    than silently falling back to a different method."""
    if method == "layerdrop":
        if model_family != "qwen":
            raise ValueError("LayerDropQwenLoRA only supports model_family='qwen' currently.")
        return LayerDropQwenLoRA, QWEN_SPEC, f"Qwen2.5 (LayerDrop, rate set via --layerdrop_rate)"
    if method == "mod":
        if model_family != "qwen":
            raise ValueError("MoDQwenLoRA only supports model_family='qwen' currently.")
        return MoDQwenLoRA, QWEN_SPEC, f"Qwen2.5 (Mixture-of-Depths, capacity set via --mod_capacity)"
    if model_family == "llama":
        return GatedLlamaLoRA, LLAMA_SPEC, "TinyLlama-1.1B-Chat-v1.0"
    return GatedQwenLoRA, QWEN_SPEC, "Qwen2.5-0.5B-Instruct"


def build_config(args) -> SystemConfig:
    # Middle-layer boundary for biased init (Eq. 7): ALWAYS derive
    # proportionally from the ACTUAL n_layers of whatever spec is currently
    # active (QWEN_SPEC or LLAMA_SPEC), rather than hardcoding (8,17) for
    # any Qwen run regardless of scale. FIXED: previously this hardcoded
    # (8,17) for every Qwen run -- correct by coincidence at 0.5B (24 layers,
    # since derive_middle_bounds(24)==(8,17) exactly), but WRONG for any
    # other Qwen scale (e.g. 1.5B/28 layers should be (9,20), not (8,17)).
    # This bug was discovered by inspecting a 1.5B checkpoint trained before
    # this fix existed -- see layerroute_errata_note.pdf and STAGE3_NOTES.
    if args.model_family == "llama":
        middle_start, middle_end = derive_middle_bounds(LLAMA_SPEC["n_layers"])
    else:
        middle_start, middle_end = derive_middle_bounds(QWEN_SPEC["n_layers"])

    return SystemConfig(
        lora     = LoRAConfig(
            r              = args.lora_r,
            alpha          = args.lora_alpha,
            dropout        = args.lora_dropout,
        ),
        router   = RouterConfig(
            init_bias_early  = args.router_init_early,
            init_bias_middle = args.router_init_middle,
            threshold        = args.router_threshold,
            gate_reg_weight  = args.gate_reg_weight,
            middle_start     = middle_start,
            middle_end       = middle_end,
        ),
        training = TrainingConfig(
            lr          = args.lr,
            batch_size  = args.batch_size,
            grad_accum  = args.grad_accum,
            max_steps   = args.max_steps,
            output_dir  = args.output_dir,
            max_seq_len = args.max_seq_len,
        ),
        data      = DataConfig(
            max_train_samples = args.max_train_samples,
        ),
        layerdrop = LayerDropConfig(
            rate                 = args.layerdrop_rate,
            inference_skip_ratio = args.layerdrop_inference_skip_ratio,
        ),
        mod       = MoDConfig(
            capacity = args.mod_capacity,
            gate_reg_weight = args.mod_gate_reg_weight,
        ),
        # FIX: build_config previously never passed model_family here, so
        # cfg.model_family always silently defaulted to "qwen" regardless of
        # --model_family, breaking data/loader.py's load_tokenizer(cfg.model_family)
        # dispatch for any non-qwen run. Now correctly threaded through.
        model_family = args.model_family,
        method       = args.method,
    )


def run_train(args):
    cfg   = build_config(args)
    model_cls, spec, name = _family_spec(args.model_family, args.method)

    print(f"\n[Model] Loading {name} + LoRA + Routers...")
    model = model_cls.from_pretrained(cfg)
    print(f"  Trainable : {model.trainable_params():,}")
    print(f"  Total     : {model.total_params():,}")

    from data.loader import build_dataloaders
    from utils.trainer import Trainer

    print("\n[Data] Building dataloaders...")
    train_l, val_l = build_dataloaders(cfg)

    trainer = Trainer(model, cfg)
    trainer.train(train_l, val_l)

    # Auto-run demo after training
    args.ckpt = f"{args.output_dir}/best_adapters.pt"
    run_demo(args)


def run_demo(args):
    from transformers import AutoTokenizer
    cfg   = build_config(args)
    model_cls, spec, name = _family_spec(args.model_family, args.method)
    model = model_cls.from_pretrained(cfg)

    if args.ckpt:
        model.load_adapters(args.ckpt)

    device = next(p for p in model.hf_model.parameters()).device
    if hasattr(model, "routers"):
        model.routers = model.routers.to(device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(spec["hf_name"], trust_remote_code=True)

    samples = [
        ("<tool_call> search_database(table='AUFK', filter='AUFNR=100023')",
         "TOOL CALL"),
        ("<tool_call> get_work_order_status(id=78234, priority=True)",
         "TOOL CALL"),
        ("Given the current backlog of 450 open work orders across 3 plants, "
         "develop a prioritization strategy that balances equipment criticality "
         "and crew capacity over the next 2 weeks.",
         "PLANNING"),
        ("Analyze the root cause of recurring pump failures in Plant B and "
         "propose a preventive maintenance schedule.",
         "PLANNING"),
    ]

    print("\n" + "═"*55)
    print(f"  CONFGATE v6 DEMO ({name})")
    print("  LoRA + Hard-Gated Skip Connections")
    print("═"*55)

    for text, label in samples:
        prompt = tok.apply_chat_template(
            [{"role":"user","content":text}],
            tokenize=False, add_generation_prompt=True
        )
        device = next(model.parameters()).device
        ids = tok.encode(prompt, return_tensors="pt",
                         max_length=cfg.training.max_seq_len, truncation=True).to(device)

        print(f"\n[{label}] {repr(text[:65])}")
        with torch.no_grad():
            logits, _, gate_stats = model(ids)

        print(f"  Layers run : {gate_stats['layers_run']:.0f}/{model.n_layers}  "
              f"(skip {gate_stats['skip_pct']:.1f}%)")
        print(f"  Gate values: {gate_stats['gate_values']}")

        probs = F.softmax(logits[0, -1, :], dim=-1)
        top5  = torch.topk(probs, 5)
        print("  Top-5 next tokens:")
        for p, idx in zip(top5.values, top5.indices):
            print(f"    {repr(tok.decode([idx.item()])):15s} p={p.item():.4f}")

    # Gate analysis -- only meaningful for methods with a per-layer,
    # sequence-level gate (LayerRoute). LayerDrop has no learned router at
    # all (fixed structural pruning); MoD routes per-token, not per-layer,
    # so a single scalar-per-layer view doesn't apply the same way.
    if hasattr(model, "routers") and hasattr(model.routers, "gate_values"):
        print("\n" + "\u2550"*55)
        print("  GATE VALUES (lower = layer more often skipped)")
        gvals = model.routers.gate_values()
        for i, g in enumerate(gvals):
            bar = "\u2588" * int(g * 20)
            print(f"  Layer {i:2d}: {g:.4f}  {bar}")
    elif args.method == "layerdrop":
        print("\n" + "\u2550"*55)
        print("  LAYERDROP: fixed inference-time drop set (input-independent)")
        print(f"  Dropped layers: {model.get_skip_layers()}")
    elif args.method == "mod":
        print("\n" + "\u2550"*55)
        print("  MOD: routing is per-token, not per-layer -- see skip_pct per sample above.")


def parse_args():
    p = argparse.ArgumentParser(description="ConfGate v6 — LoRA + Skip Gates")
    p.add_argument("--mode",           default="demo",
                   choices=["train","demo"])
    p.add_argument("--model_family",   default="qwen", choices=["qwen", "llama"],
                   help="qwen -> Qwen2.5-Instruct (scale set via --model_scale). "
                        "llama -> TinyLlama-1.1B-Chat-v1.0 (22 layers).")
    p.add_argument("--model_scale",    default="0.5b", choices=["0.5b", "1.5b"],
                   help="Only used when --model_family qwen. Sets QWEN_SPEC "
                        "in-place before model construction.")
    p.add_argument("--method",         default="layerroute",
                   choices=["layerroute", "layerdrop", "mod"],
                   help="layerroute -> learned, input-adaptive per-sequence gate. "
                        "layerdrop -> fixed, input-independent structural pruning "
                        "(Fan et al. 2020), qwen-only. "
                        "mod -> Mixture-of-Depths, learned per-token top-k routing "
                        "with true gather/scatter (Raposo et al. 2024), qwen-only.")

    # LayerDrop
    p.add_argument("--layerdrop_rate", type=float, default=0.2,
                   help="Train-time per-layer stochastic drop probability.")
    p.add_argument("--layerdrop_inference_skip_ratio", type=float, default=0.25,
                   help="Fixed, input-independent fraction of (prunable) layers "
                        "dropped at inference time.")

    # Mixture-of-Depths
    p.add_argument("--mod_capacity",   type=float, default=0.75,
                   help="Target AVERAGE fraction of tokens routed through each "
                        "layer's block, enforced via gate_reg_weight pressure "
                        "(STE per-token gate, not an exact per-step count).")
    p.add_argument("--mod_gate_reg_weight", type=float, default=1.0,
                   help="Regularization weight pushing the average per-token "
                        "gate value toward --mod_capacity. Increase if actual "
                        "skip_pct stays far below the target capacity.")

    # LoRA
    p.add_argument("--lora_r",         type=int,   default=8)
    p.add_argument("--lora_alpha",     type=float, default=16.0)
    p.add_argument("--lora_dropout",   type=float, default=0.05)

    # Router
    p.add_argument("--router_threshold",     type=float, default=0.5)
    p.add_argument("--router_init_early",    type=float, default=1.0,
                   help="Init bias for early/late layers (sigmoid(1.0)=0.73)")
    p.add_argument("--router_init_middle",   type=float, default=-1.0,
                   help="Init bias for middle layers (sigmoid(-1.0)=0.27). "
                        "Boundary is derived per model_family -- see build_config().")
    p.add_argument("--gate_reg_weight",      type=float, default=1.0,
                   help="Gate regularisation weight. Higher=more aggressive skipping.")

    # Training
    p.add_argument("--lr",             type=float, default=2e-4)
    p.add_argument("--batch_size",     type=int,   default=4)
    p.add_argument("--grad_accum",     type=int,   default=4)
    p.add_argument("--max_steps",      type=int,   default=1000)
    p.add_argument("--max_seq_len",    type=int,   default=512)
    p.add_argument("--output_dir",     default="./checkpoints")
    p.add_argument("--max_train_samples", type=int, default=5000)

    # Inference
    p.add_argument("--ckpt",           default=None)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.model_family == "qwen":
        apply_qwen_scale(args.model_scale)

    print("=" * 55)
    print("  CONFGATE v6 — LoRA + Hard-Gated Skip Connections")
    print("=" * 55)
    print(f"  mode         : {args.mode}")
    print(f"  model_family : {args.model_family}")
    print(f"  LoRA r       : {args.lora_r}  alpha={args.lora_alpha}")
    print(f"  Router thresh: {args.router_threshold}")
    print(f"  Max steps    : {args.max_steps}")

    if args.mode == "train":
        run_train(args)
    else:
        run_demo(args)