"""
run_layerdrop_timed.py
========================
Cost + accuracy measurement for LayerDrop, matching run_layerroute_timed.py's
(corrected) protocol exactly: manual token-by-token generation through
model(ids) -- NOT hf_model.generate() -- so the actual fixed drop-set
pruning genuinely takes effect (same class of bug already found and fixed
in run_layerroute_timed.py; applying the same verification discipline here
from the start rather than discovering it later).

Baseline: LayerDropQwenLoRA has no learned router to force open -- its
"vanilla-equivalent" baseline is the SAME model with ZERO layers pruned,
implemented by temporarily clearing self._inference_drop_set (empty set =
every layer runs), NOT relying on hf_model.generate() (matches the
already-fixed LayerRoute pattern; the raw hf_model still lacks a way to
bypass LoRA/routing internals cleanly, so we go through the model's own
_forward_layers() with drop_set emptied instead).

Usage:
    python run_layerdrop_timed.py \
        --ckpt checkpoints_layerdrop_05b/best_adapters.pt \
        --model_scale 0.5b --dataset gsm8k --n_eval 100 \
        --train_wall_clock_seconds <paste from train_layerdrop_05b.log>
"""
import os, sys, json, time, argparse
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.layerdrop_qwen import LayerDropQwenLoRA
from utils.config import SystemConfig, QWEN_SPEC

QWEN_SCALES = {
    "0.5b": {"hf_name": "Qwen/Qwen2.5-0.5B-Instruct", "n_layers": 24,
             "n_q_heads": 14, "n_kv_heads": 2, "hidden": 896,
             "head_dim": 64, "intermediate": 4864, "vocab": 151936},
    "1.5b": {"hf_name": "Qwen/Qwen2.5-1.5B-Instruct", "n_layers": 28,
             "n_q_heads": 12, "n_kv_heads": 2, "hidden": 1536,
             "head_dim": 128, "intermediate": 8960, "vocab": 151936},
}


def apply_model_scale(scale: str):
    QWEN_SPEC.clear()
    QWEN_SPEC.update(QWEN_SCALES[scale])
    print(f"  [model_scale={scale}] QWEN_SPEC set to {QWEN_SPEC['hf_name']} "
         f"({QWEN_SPEC['n_layers']} layers, hidden={QWEN_SPEC['hidden']})")


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_model(ckpt_path, lora_r, lora_alpha, layerdrop_inference_skip_ratio, max_seq_len):
    cfg = SystemConfig()
    cfg.lora.r = lora_r
    cfg.lora.alpha = lora_alpha
    cfg.layerdrop.inference_skip_ratio = layerdrop_inference_skip_ratio
    model = LayerDropQwenLoRA.from_pretrained(cfg)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg


def build_eval_samples_gsm8k(tok, n=100, seed=2024):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=seed).select(range(n))
    return [{"prompt": tok.apply_chat_template(
                [{"role": "user", "content": row["question"]}],
                tokenize=False, add_generation_prompt=True),
            "gold": row["answer"]} for row in ds]


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
        samples.append({"prompt": tok.apply_chat_template(
                            [{"role": "user", "content": instruction}],
                            tokenize=False, add_generation_prompt=True),
                        "gold": row["highlights"]})
    return samples


def build_eval_samples(tok, dataset, n=100, seed=2024):
    if dataset == "gsm8k":
        return build_eval_samples_gsm8k(tok, n=n, seed=seed)
    elif dataset == "cnndm":
        return build_eval_samples_cnndm(tok, n=n, seed=seed)
    raise ValueError(f"Unknown dataset '{dataset}'.")


def time_query_gated(model, tokenizer, prompt, max_new_tokens, max_seq_len=512):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model_scale", default="0.5b", choices=["0.5b", "1.5b"])
    p.add_argument("--dataset", default="gsm8k")
    p.add_argument("--n_eval", type=int, default=100)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--layerdrop_inference_skip_ratio", type=float, default=0.25)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--out_dir", default="results/layerdrop_timed")
    p.add_argument("--train_wall_clock_seconds", type=float, required=True)
    args = p.parse_args()

    apply_model_scale(args.model_scale)

    os.makedirs(args.out_dir, exist_ok=True)
    model, cfg = build_model(args.ckpt, args.lora_r, args.lora_alpha,
                             args.layerdrop_inference_skip_ratio, args.max_seq_len)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(QWEN_SPEC["hf_name"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    samples = build_eval_samples(tok, args.dataset, n=args.n_eval, seed=args.seed)
    records = []
    identical_output_count = 0

    original_drop_set = set(model._inference_drop_set)  # save for restore after baseline pass

    for i, sample in enumerate(samples):
        prompt = sample["prompt"]

        # (1) LayerDrop timed pass: fixed drop set active, real pruning happens.
        model._inference_drop_set = original_drop_set
        routed_ms, routed_text, routed_tokens = time_query_gated(
            model, tok, prompt, args.max_new_tokens, args.max_seq_len)

        # (2) Vanilla-equivalent baseline: empty drop set, every layer runs.
        model._inference_drop_set = set()
        base_ms, base_text, base_tokens = time_query_gated(
            model, tok, prompt, args.max_new_tokens, args.max_seq_len)
        model._inference_drop_set = original_drop_set

        if routed_text == base_text:
            identical_output_count += 1

        records.append({
            "query_id": i,
            "method": "LayerDrop",
            "model_id": QWEN_SPEC["hf_name"],
            "pure_inference_ms": routed_ms,
            "search_overhead_ms": 0.0,
            "baseline_ms": base_ms,
            "new_tokens": routed_tokens,
            "baseline_new_tokens": base_tokens,
            "generated_text": routed_text,
            "baseline_generated_text": base_text,
            "gold": sample["gold"],
            "kv_cache": False,
            "drop_set": sorted(original_drop_set),
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] routed={routed_ms:.1f}ms  baseline={base_ms:.1f}ms  "
                 f"identical_so_far={identical_output_count}/{i+1}")

    out_path = os.path.join(args.out_dir, "layerdrop_query_costs.json")
    with open(out_path, "w") as f:
        json.dump({
            "train_cost_seconds": args.train_wall_clock_seconds,
            "kv_cache": False,
            "drop_set": sorted(original_drop_set),
            "records": records,
        }, f, indent=2)
    print(f"\nSaved {len(records)} query cost records -> {out_path}")
    print(f"Identical routed/baseline outputs: {identical_output_count}/{len(samples)} "
         f"({'SUSPICIOUS -- pruning may not be taking effect, investigate before trusting results' if identical_output_count > len(samples)*0.5 else 'expected some overlap on short/easy prompts'})")


if __name__ == "__main__":
    main()