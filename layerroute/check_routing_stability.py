"""
check_routing_stability.py
============================
Empirical check for per-sequence (frozen) routing: does the router's gate
decision, computed from ONLY the prompt, stay consistent with what it would
decide later, deeper into generation (with more tokens in context)?

This determines whether an ALREADY-TRAINED checkpoint (trained with
continuous, full-growing-sequence re-routing) can be evaluated directly
under a frozen, prompt-only routing protocol, or whether it would need
retraining under that protocol to produce sensible/matching decisions.

Method: for each sample, generate normally (current, continuous-rerouting
behavior), and at several checkpoints during generation (10, 30, 60 tokens
in), snapshot the per-layer soft gate values using the CURRENT full
sequence. Compare each snapshot against the prompt-only snapshot:
  - Per-layer gate VALUE drift (mean absolute difference in soft_g)
  - Per-layer HARD decision flips (how many of 24 layers flip open<->closed)

Low drift / few flips -> prompt-only routing is a reasonable approximation,
safe to evaluate directly on the existing checkpoint.
High drift / many flips -> the router's decision genuinely depends on
generated content, not just the prompt -- freezing at the prompt would
change what the model computes, and retraining under the new protocol
would likely be needed before trusting results.
"""
import torch
from models.gated_qwen import GatedQwenLoRA
from utils.config import SystemConfig, QWEN_SPEC
from transformers import AutoTokenizer


def compute_gate_snapshot(model, ids):
    """Compute per-layer soft gate values for the CURRENT sequence, without
    running the full forward pass (cheap -- just the router computation)."""
    transformer = model.hf_model.model
    hidden = transformer.embed_tokens(ids)
    gate_vals = []
    with torch.no_grad():
        h = hidden
        T = ids.shape[1]
        position_ids = torch.arange(T, device=ids.device).unsqueeze(0)
        cos, sin = transformer.rotary_emb(h, position_ids)
        layer_kwargs = dict(attention_mask=None, position_ids=position_ids,
                            past_key_values=None, use_cache=False,
                            position_embeddings=(cos, sin))
        for i, layer in enumerate(transformer.layers):
            router = model.routers.routers[i]
            h_mean = h.mean(dim=1).to(router.linear.weight.dtype)
            score = router.linear(h_mean).squeeze(-1)
            soft_g = torch.sigmoid(score).item()
            gate_vals.append(soft_g)
            # advance hidden through the REAL gated computation so later
            # layers see a realistic input, matching actual generation
            hard_g = 1.0 if soft_g > router.threshold else 0.0
            out = layer(h, **layer_kwargs)
            h_out = out[0] if isinstance(out, tuple) else out
            h = (hard_g * h_out + (1.0 - hard_g) * h).to(hidden.dtype)
    return gate_vals


def main():
    # Explicit scale setting -- QWEN_SPEC's on-disk default may be left at
    # whatever scale a previous session used last (1.5B, in this project's
    # history). Match it explicitly to the 0.5B checkpoint being loaded.
    QWEN_SPEC.clear()
    QWEN_SPEC.update({
        "hf_name": "Qwen/Qwen2.5-0.5B-Instruct", "n_layers": 24,
        "n_q_heads": 14, "n_kv_heads": 2, "hidden": 896,
        "head_dim": 64, "intermediate": 4864, "vocab": 151936,
    })

    cfg = SystemConfig()
    model = GatedQwenLoRA.from_pretrained(cfg)
    state = torch.load("checkpoints_0.5B_backup/best_adapters.pt", map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    tok = AutoTokenizer.from_pretrained(QWEN_SPEC["hf_name"])

    prompts = [
        "What is 15 times 23?",
        "Explain the water cycle in one paragraph.",
        "If a train travels 60 miles in 45 minutes, what is its speed in mph?",
    ]

    checkpoints = [0, 10, 30, 60]  # tokens generated so far, when snapshot is taken

    for p_text in prompts:
        print(f"\n{'='*70}\nPrompt: {p_text[:60]}\n{'='*70}")
        prompt = tok.apply_chat_template([{"role": "user", "content": p_text}],
                                         tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").input_ids.to(model.hf_model.device)

        snapshots = {}
        cur = ids.clone()
        for target in checkpoints:
            while cur.shape[1] - ids.shape[1] < target:
                with torch.no_grad():
                    logits, _, _ = model(cur)
                    nxt = logits[0, -1, :].argmax(dim=-1, keepdim=True).unsqueeze(0)
                cur = torch.cat([cur, nxt], dim=1)
            snapshots[target] = compute_gate_snapshot(model, cur)

        prompt_only = snapshots[0]
        print(f"{'tokens_gen':<12}{'mean_abs_drift':<18}{'hard_flips (of 24)':<20}")
        for target in checkpoints:
            snap = snapshots[target]
            drift = sum(abs(a - b) for a, b in zip(prompt_only, snap)) / len(prompt_only)
            flips = sum(1 for a, b in zip(prompt_only, snap)
                       if (a > 0.5) != (b > 0.5))
            print(f"{target:<12}{drift:<18.4f}{flips:<20d}")


if __name__ == "__main__":
    main()