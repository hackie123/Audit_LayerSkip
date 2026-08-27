"""
run_layerroute_timed.py  (place in conf_gate6/)
------------------------------------------------
CORRECTED VERSION -- see STAGE3_NOTES for the full bug report. The previous
version of this script timed `model.hf_model.generate(...)` -- HF's stock
generate() on the RAW, unwrapped backbone. This NEVER exercises
GatedQwenLoRA._forward_layers() (the router/STE gating logic lives entirely
in that separate, hand-written method, which .generate() has no way to
call). Verified empirically: routed and force-open passes produced
BYTE-IDENTICAL output text under the old script, at both scales.

FIX: generation now goes through model(ids) -- GatedQwenLoRA's own
forward() -- in a manual, token-by-token greedy decoding loop. This
genuinely exercises the trained gates.

KNOWN, FLAGGED LIMITATION (separate from the bug above, not fixed here):
this manual loop has NO KV-CACHE (GatedQwenLoRA's _forward_layers() was
never built to support one), so it reprocesses the full sequence at every
new token (O(T^2)). ConfLayers/SWIFT's custom decode loops DO use real KV
caching. This means the corrected cost numbers below are now CORRECTLY
GATED, but still not a fully cache-fair speed comparison against
ConfLayers/SWIFT. Flagged explicitly in every output record
(`kv_cache: false`) rather than presented as a clean apples-to-apples
speed number.

Usage:
    python run_layerroute_timed.py \
        --ckpt checkpoints/best_adapters.pt \
        --dataset gsm8k --n_eval 100 \
        --out_dir results/layerroute_timed \
        --train_wall_clock_seconds <paste from your training log>
"""
import os, sys, json, time, argparse
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.gated_qwen import GatedQwenLoRA
from utils.config import SystemConfig, QWEN_SPEC

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


def apply_model_scale(scale: str):
    if scale not in QWEN_SCALES:
        raise ValueError(f"Unknown --model_scale '{scale}'. Choices: {list(QWEN_SCALES)}")
    QWEN_SPEC.clear()
    QWEN_SPEC.update(QWEN_SCALES[scale])
    print(f"  [model_scale={scale}] QWEN_SPEC set to {QWEN_SPEC['hf_name']} "
         f"({QWEN_SPEC['n_layers']} layers, hidden={QWEN_SPEC['hidden']})")


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_model(ckpt_path, lora_r, lora_alpha, router_threshold, max_seq_len):
    cfg = SystemConfig()
    cfg.lora.r = lora_r
    cfg.lora.alpha = lora_alpha
    cfg.router.threshold = router_threshold
    model = GatedQwenLoRA.from_pretrained(cfg)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg


def build_eval_samples_gsm8k(tok, n=100, seed=2024):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=seed).select(range(n))
    samples = []
    for row in ds:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": row["question"]}],
            tokenize=False, add_generation_prompt=True,
        )
        samples.append({"prompt": prompt, "gold": row["answer"]})
    return samples


def build_eval_samples_cnndm(tok, n=100, seed=2024):
    from datasets import load_dataset
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test").shuffle(seed=seed).select(range(n))
    samples = []
    for row in ds:
        instruction = (
            "Summarize the following article in 2-3 concise sentences "
            "(no more than 60 words total). Do not repeat the article or add commentary.\n\n"
            f"Article: {row['article']}"
        )
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False, add_generation_prompt=True,
        )
        samples.append({"prompt": prompt, "gold": row["highlights"]})
    return samples


def build_eval_samples(tok, dataset, n=100, seed=2024):
    if dataset == "gsm8k":
        return build_eval_samples_gsm8k(tok, n=n, seed=seed)
    elif dataset == "cnndm":
        return build_eval_samples_cnndm(tok, n=n, seed=seed)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Supported: gsm8k, cnndm.")


def time_query_gated(model, tokenizer, prompt, max_new_tokens, max_seq_len=512):
    """
    Manual, token-by-token greedy decoding through model(ids) -- GatedQwenLoRA's
    OWN forward(), which genuinely calls _forward_layers() and applies the
    trained (or force-opened) gates. NO KV cache (see module docstring) --
    reprocesses the full sequence-so-far at every step.
    """
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len)
    ids = enc["input_ids"].to(next(model.hf_model.parameters()).device)
    prompt_len = ids.shape[1]
    eos_id = tokenizer.eos_token_id

    _sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _, _ = model(ids)
            next_id = logits[0, -1, :].argmax(dim=-1, keepdim=True).unsqueeze(0)
            ids = torch.cat([ids, next_id], dim=1)
            if eos_id is not None and next_id.item() == eos_id:
                break
            # FIX: previously checked ids.shape[1] >= max_seq_len (the SAME
            # value used for PROMPT truncation), meaning any prompt already
            # truncated to max_seq_len had ZERO headroom to generate --
            # confirmed via measurement to cause new_tokens=1 on ~80% of
            # CNN/DM samples (long articles truncated to exactly max_seq_len,
            # loop's own cap then fires after a single generated token).
            # Now guarantees max_new_tokens of real headroom regardless of
            # prompt length, decoupled from the truncation limit.
            if ids.shape[1] - prompt_len >= max_new_tokens:
                break
    _sync()
    ms = (time.perf_counter() - t0) * 1000.0

    gen_text = tokenizer.decode(ids[0][prompt_len:], skip_special_tokens=True)
    return ms, gen_text, ids.shape[1] - prompt_len


# NOTE (second bug, caught by re-verifying against the REAL _forward_layers()
# code path rather than a simplified test double): GatedQwenLoRA._forward_
# layers() inlines its gate computation (score/sigmoid/hard_g) directly --
# it NEVER calls LayerRouter.forward(), which is the ONLY place `force_open`
# is checked. Setting router.force_open has therefore always been dead code
# in the actual forward pass. FIX: force layers open the same way
# evaluate.py's eval_perplexity correctly does -- overwrite router.linear.
# bias directly (sigmoid(10.0)~=1.0), which DOES feed into the real inline
# computation. Unlike evaluate.py's version, this saves and restores each
# router's EXACT original bias (not a blanket restore-to-init_bias_early),
# verified via torch.equal before/after in a structural test.

def force_all_gates_open(model):
    """Returns the saved original bias tensors, to be restored via
    restore_routing(model, saved). This IS the genuine vanilla-equivalent
    baseline: all layers active, no skipping -- verified structurally to
    actually change layers_run (3.0 -> 4.0 in a 4-layer test), unlike the
    old router.force_open mechanism which never took effect."""
    saved = [r.linear.bias.data.clone() for r in model.routers.routers]
    for r in model.routers.routers:
        r.linear.bias.data.fill_(10.0)  # sigmoid(10.0) ~= 1.0 -- forces hard_g=1 for every input
    return saved


def restore_routing(model, saved):
    for r, s in zip(model.routers.routers, saved):
        r.linear.bias.data.copy_(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/best_adapters.pt")
    p.add_argument("--model_scale", default="0.5b", choices=["0.5b", "1.5b"])
    p.add_argument("--dataset", default="gsm8k")
    p.add_argument("--n_eval", type=int, default=100)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--router_threshold", type=float, default=0.5)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--out_dir", default="results/layerroute_timed")
    p.add_argument("--train_wall_clock_seconds", type=float, required=True)
    args = p.parse_args()

    apply_model_scale(args.model_scale)

    os.makedirs(args.out_dir, exist_ok=True)
    model, cfg = build_model(args.ckpt, args.lora_r, args.lora_alpha,
                             args.router_threshold, args.max_seq_len)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(QWEN_SPEC["hf_name"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    samples = build_eval_samples(tok, args.dataset, n=args.n_eval, seed=args.seed)
    records = []
    identical_output_count = 0  # sanity counter -- if this stays high, gating still isn't taking effect
    for i, sample in enumerate(samples):
        prompt = sample["prompt"]

        # (1) LayerRoute timed pass: trained routers active, real skipping happens.
        #     (No restore_routing needed here -- biases are only ever modified
        #     inside the force_all_gates_open/restore_routing pair below, so
        #     the routers are already at their trained state entering this call.)
        routed_ms, routed_text, routed_tokens = time_query_gated(
            model, tok, prompt, args.max_new_tokens, args.max_seq_len)

        # (2) Vanilla-equivalent baseline: force all gates open via bias
        # manipulation (the ONLY mechanism confirmed to actually affect
        # _forward_layers()'s real inline gate computation), then restore
        # the EXACT original per-router bias afterward.
        saved_biases = force_all_gates_open(model)
        base_ms, base_text, base_tokens = time_query_gated(
            model, tok, prompt, args.max_new_tokens, args.max_seq_len)
        restore_routing(model, saved_biases)

        if routed_text == base_text:
            identical_output_count += 1

        records.append({
            "query_id": i,
            "method": "LayerRoute",
            "model_id": QWEN_SPEC["hf_name"],
            "pure_inference_ms": routed_ms,
            "search_overhead_ms": 0.0,
            "baseline_ms": base_ms,
            "new_tokens": routed_tokens,
            "baseline_new_tokens": base_tokens,  # ADDED: was previously discarded, needed to score baseline accuracy for comparison
            "generated_text": routed_text,
            "baseline_generated_text": base_text,  # ADDED: same reason
            "gold": sample["gold"],
            "kv_cache": False,  # flagged explicitly -- see module docstring
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] routed={routed_ms:.1f}ms  baseline={base_ms:.1f}ms  "
                 f"identical_so_far={identical_output_count}/{i+1}")

    out_path = os.path.join(args.out_dir, "layerroute_query_costs.json")
    with open(out_path, "w") as f:
        json.dump({
            "train_cost_seconds": args.train_wall_clock_seconds,
            "kv_cache": False,
            "records": records,
        }, f, indent=2)
    print(f"\nSaved {len(records)} query cost records -> {out_path}")
    print(f"Identical routed/baseline outputs: {identical_output_count}/{len(samples)} "
         f"({'SUSPICIOUS -- gating may not be taking effect, investigate before trusting results' if identical_output_count > len(samples)*0.5 else 'expected some overlap on short/easy prompts'})")
    print("NOTE: verify `generated_text` against gold answers separately for "
         "quality_score (exact_match/ROUGE) before feeding into cost_accounting.py")


if __name__ == "__main__":
    main()