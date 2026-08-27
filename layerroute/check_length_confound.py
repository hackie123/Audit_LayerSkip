"""
check_length_confound.py
=========================
Tests whether the inverted TinyLlama skip differential (planning skips MORE
than tool_call, opposite of Qwen2.5-0.5B) could be explained by a sequence-
length confound rather than genuine architecture-driven routing behavior.

Mechanism under test: the router computes h_mean = hidden.mean(dim=1), a
mean-pool over the SEQUENCE LENGTH dimension. If planning (GSM8K) samples
come out systematically longer under TinyLlama's tokenizer (32K vocab) than
under Qwen's (151K vocab), that length shift alone -- independent of actual
step-type semantics -- could be driving a different mean-pooled
representation and thus a different routing decision.

Test: tokenize the SAME 100 eval samples (50 Hermes tool_call + 50 GSM8K
planning, identical to evaluate.py's load_eval_samples) with BOTH
tokenizers, and compare the tool_call/planning length ratio under each.

  - If the ratio is SIMILAR under both tokenizers -> length-distribution-
    per-se is not a good explanation for the FLIP (the same relative length
    pattern existed for Qwen too, yet Qwen showed the opposite differential
    direction).
  - If TinyLlama's tokenizer inflates planning length MUCH more than Qwen's
    does -> real confound candidate, worth controlling for before trusting
    the inversion as purely architecture-driven.

Usage (run in layerroute_venv, same env used for TinyLlama training):
    python check_length_confound.py --n 100
"""

import argparse
from transformers import AutoTokenizer


def load_eval_samples(n=100):
    """Mirrors evaluate.py's load_eval_samples exactly -- same samples used
    for the routing-differential measurement, so the length check is on the
    EXACT data that produced the inverted result, not a fresh sample."""
    from datasets import load_dataset
    samples = []
    try:
        ds = load_dataset("NousResearch/hermes-function-calling-v1", split="train")
        ds = ds.select(range(min(n // 2, len(ds))))
        for row in ds:
            conv = row.get("conversations", [])
            turns = [{"role": ("assistant" if c.get("from") == "gpt" else "user"),
                     "content": c.get("value", "")} for c in conv]
            if turns:
                samples.append((turns, "tool_call"))
    except Exception as e:
        print(f"  [warn] Hermes load failed: {e}")

    try:
        ds = load_dataset("openai/gsm8k", "main", split="test")
        ds = ds.select(range(min(n // 2, len(ds))))
        for row in ds:
            turns = [{"role": "user", "content": row["question"]},
                     {"role": "assistant", "content": row["answer"]}]
            samples.append((turns, "planning"))
    except Exception as e:
        print(f"  [warn] GSM8K load failed: {e}")

    return samples


def token_lengths(samples, tok):
    lengths = {"tool_call": [], "planning": []}
    for turns, step_type in samples:
        prompt = tok.apply_chat_template(turns, tokenize=False, add_generation_prompt=False)
        ids = tok.encode(prompt, add_special_tokens=True)
        lengths[step_type].append(len(ids))
    return lengths


def summarize(name, lengths):
    tc = lengths["tool_call"]
    pl = lengths["planning"]
    tc_mean = sum(tc) / len(tc) if tc else 0
    pl_mean = sum(pl) / len(pl) if pl else 0
    ratio = pl_mean / tc_mean if tc_mean > 0 else float("nan")
    print(f"\n[{name}]")
    print(f"  tool_call : n={len(tc):3d}  mean_len={tc_mean:7.1f}  "
         f"min={min(tc) if tc else 0}  max={max(tc) if tc else 0}")
    print(f"  planning  : n={len(pl):3d}  mean_len={pl_mean:7.1f}  "
         f"min={min(pl) if pl else 0}  max={max(pl) if pl else 0}")
    print(f"  planning/tool_call length ratio: {ratio:.3f}")
    return ratio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    args = p.parse_args()

    print("Loading eval samples (same as evaluate.py)...")
    samples = load_eval_samples(n=args.n)
    print(f"  tool_call: {sum(1 for _,t in samples if t=='tool_call')} samples")
    print(f"  planning : {sum(1 for _,t in samples if t=='planning')} samples")

    print("\nLoading tokenizers...")
    qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    llama_tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", trust_remote_code=True)

    qwen_lengths = token_lengths(samples, qwen_tok)
    llama_lengths = token_lengths(samples, llama_tok)

    ratio_qwen = summarize("Qwen2.5 tokenizer (151,936 vocab)", qwen_lengths)
    ratio_llama = summarize("TinyLlama tokenizer (32,000 vocab)", llama_lengths)

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    diff = ratio_llama - ratio_qwen
    rel_diff = diff / ratio_qwen * 100 if ratio_qwen else float("nan")
    print(f"  Qwen tokenizer   planning/tool_call length ratio: {ratio_qwen:.3f}")
    print(f"  TinyLlama tokenizer planning/tool_call length ratio: {ratio_llama:.3f}")
    print(f"  Difference: {diff:+.3f} ({rel_diff:+.1f}% relative)")
    if abs(rel_diff) < 15:
        print("\n  -> Length ratio is SIMILAR across tokenizers.")
        print("     Sequence-length confound is UNLIKELY to explain the inverted")
        print("     differential -- the length pattern was comparable under Qwen's")
        print("     tokenizer too, yet Qwen showed the OPPOSITE routing direction.")
    else:
        print("\n  -> Length ratio DIFFERS meaningfully across tokenizers.")
        print("     Sequence-length is a PLAUSIBLE partial confound -- TinyLlama's")
        print("     tokenizer changes the relative length of planning vs tool_call")
        print("     inputs, which could interact with the mean-pooled router input.")
        print("     Recommend controlling for length before treating the inversion")
        print("     as purely architecture-driven.")


if __name__ == "__main__":
    main()