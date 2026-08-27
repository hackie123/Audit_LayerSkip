"""
build_audit_report.py  (place at repo root, alongside audit_harness/ and conf_gate6/)
----------------------------------------------------------------------------------
Reads the three methods' raw per-query cost records and produces the final
amortized-cost report via harness/cost_accounting.py. No manual number-copying.

Expects:
  --conflayers_json : output of Stage 2 (ConfLayers, frozen-replay instrumented)
  --swift_json       : output of Stage 2 (SWIFT, frozen-replay instrumented)
  --layerroute_json  : output of run_layerroute_timed.py (Stage 3)

Each input JSON is a list of QueryCostRecord dicts (ConfLayers/SWIFT) or the
{"train_cost_seconds":..., "records":[...]} shape (LayerRoute) -- see each
stage's script for the exact schema.
"""
import argparse, json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_harness"))
from harness.cost_accounting import MethodCostProfile, build_report, save_report


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


import re


def _extract_answer(text):
    m = re.findall(r'####\s*(-?[\d,]+(?:\.\d+)?)', text)
    return m[0].replace(',', '').strip() if m else None


def _load_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def _gsm8k_gold_answers(n=100, seed=2024):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test").shuffle(seed=seed).select(range(n))
    return [_extract_answer(ex["answer"]) for ex in ds]


def _cnndm_gold_summaries(n=100, seed=2024):
    """Matches ConfLayers' eval.py load_data() exactly: cnn_dailymail, config
    3.0.0, split='test', shuffle(seed).select(range(n))."""
    from datasets import load_dataset
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test").shuffle(seed=seed).select(range(n))
    return [ex["highlights"] for ex in ds]


_rouge_scorer = None


def _rouge_l(pred, gold):
    global _rouge_scorer
    if _rouge_scorer is None:
        from rouge_score import rouge_scorer
        _rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    if not pred or not gold:
        return 0.0
    return _rouge_scorer.score(gold, pred)["rougeL"].fmeasure


def _extract_summary(text, stop_marker="\nArticle:", max_words=100):
    """
    Two-stage truncation applied IDENTICALLY to every method's output before
    ROUGE scoring, so the comparison measures summary QUALITY, not which
    method rambles least:
      1. Cut at the first sign of bleeding into a new few-shot-style block
         (same non-stopping pathology observed on GSM8K).
      2. Cap at max_words (~2x the CNN/DM gold mean of ~53 words) -- observed
         raw generations run 200-400+ words vs a 53-word gold, which alone
         would dominate ROUGE-L regardless of any method difference. Capping
         uniformly (not per-method) keeps the comparison fair regardless of
         each method's native prompting style (instruct vs. few-shot).
    """
    idx = text.find(stop_marker)
    text = text[:idx].strip() if idx != -1 else text.strip()
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    return text


def _quality_scores(task, generations, golds):
    if task == "gsm8k":
        preds = [_extract_answer(g) for g in generations]
        correct = sum(1 for p, g in zip(preds, golds) if p == g and p is not None)
        return correct / len(generations)
    elif task == "cnndm":
        scores = [_rouge_l(_extract_summary(g), gold) for g, gold in zip(generations, golds)]
        return _mean(scores)
    else:
        raise ValueError(f"Unknown task '{task}'. Supported: gsm8k, cnndm.")


def profile_from_conflayers_style(json_path, name, model_id, baseline_path=None,
                                  seed=2024, task="gsm8k", cnndm_golds=None):
    """
    Reads ConfLayers/SWIFT's NATIVE output schema (JSONL, one {"choices":[{...}]}
    per line -- NOT the QueryCostRecord/timing_split schema). IMPORTANT LIMITATION:
    Stage 2 (frozen-replay timing split, see harness/timing_split.py) was never
    wired into ConfLayers' evaluation/eval.py, so wall_time here is the RAW,
    UNSPLIT cost -- search overhead and pure inference are still combined, exactly
    as ConfLayers'/SWIFT's own papers report it. search_overhead_ms is therefore
    set to 0 and pure_inference_ms carries the full (unsplit) wall_time. This is
    flagged explicitly in the report rather than silently presented as a true split.

    cnndm_golds: if provided, reuse these gold summaries instead of calling
    load_dataset("cnn_dailymail",...) again. Necessary because a second,
    separate load_dataset call for this legacy scripted dataset hits a
    datasets/huggingface_hub bug (HfUriError / OfflineModeIsEnabled) even
    though the FIRST call (in run_layerroute_timed.py) succeeds fine. Since
    all methods share the same seed/shuffle/n recipe, reusing one canonical
    load is both a workaround and better practice than redundant loads.
    """
    lines = _load_jsonl(json_path)
    per_query_ms = [sum(l["choices"][0]["wall_time"]) * 1000.0 for l in lines]  # sec -> ms

    baseline_ms = 0.0
    if baseline_path:
        base_lines = _load_jsonl(baseline_path)
        baseline_ms = _mean([sum(l["choices"][0]["wall_time"]) * 1000.0 for l in base_lines])

    generations = [l["choices"][0]["turns"] for l in lines]
    if task == "gsm8k":
        golds = _gsm8k_gold_answers(n=len(lines), seed=seed)
    elif task == "cnndm":
        if cnndm_golds is not None:
            golds = cnndm_golds[:len(lines)]
        else:
            golds = _cnndm_gold_summaries(n=len(lines), seed=seed)
    else:
        raise ValueError(f"Unknown task '{task}'.")
    quality = _quality_scores(task, generations, golds)

    return MethodCostProfile(
        name=name, model_id=model_id,
        train_cost_seconds=0.0,  # training-free by construction
        pure_inference_ms_per_query=_mean(per_query_ms),  # UNSPLIT: search+inference combined
        search_overhead_ms_per_query=0.0,  # NOT separately measured -- see docstring
        baseline_ms_per_query=baseline_ms,
        quality_score=quality,
    )


def profile_from_layerroute(json_path, model_id, seed=2024, task="gsm8k"):
    data = json.load(open(json_path))
    records = data["records"]
    generations = [r["generated_text"] for r in records]
    golds_raw = [r["gold"] for r in records]
    if task == "gsm8k":
        golds = [_extract_answer(g) for g in golds_raw]
    elif task == "cnndm":
        golds = golds_raw
    else:
        raise ValueError(f"Unknown task '{task}'.")
    quality = _quality_scores(task, generations, golds)
    return MethodCostProfile(
        name="LayerRoute", model_id=model_id,
        train_cost_seconds=data["train_cost_seconds"],
        pure_inference_ms_per_query=_mean([r["pure_inference_ms"] for r in records]),
        search_overhead_ms_per_query=0.0,
        baseline_ms_per_query=_mean([r["baseline_ms"] for r in records]),
        quality_score=quality,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--conflayers_json", required=True)
    p.add_argument("--swift_json", required=True)
    p.add_argument("--layerroute_json", required=True)
    p.add_argument("--vanilla_json", required=True,
                   help="Cleaned vanilla baseline jsonl (e.g. vanilla_clean.jsonl), "
                        "used as the wall-clock baseline for ConfLayers/SWIFT speedup.")
    p.add_argument("--model_id", default="Qwen2.5-1.5B")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--task", default="gsm8k", choices=["gsm8k", "cnndm"],
                   help="gsm8k -> exact-match on #### answer. "
                        "cnndm -> ROUGE-L F-measure against reference summary.")
    p.add_argument("--out", default="results/audit_report.json")
    p.add_argument("--query_volumes", type=int, nargs="+",
                   default=[1, 10, 100, 1000, 10000, 100000])
    args = p.parse_args()

    layerroute_profile = profile_from_layerroute(
        args.layerroute_json, args.model_id, seed=args.seed, task=args.task)

    # Reuse LayerRoute's already-loaded gold summaries for cnndm (avoids a
    # second load_dataset("cnn_dailymail",...) call, which hits a
    # datasets/huggingface_hub bug -- see profile_from_conflayers_style docstring).
    cnndm_golds = None
    if args.task == "cnndm":
        lr_data = json.load(open(args.layerroute_json))
        cnndm_golds = [r["gold"] for r in lr_data["records"]]

    profiles = [
        layerroute_profile,
        profile_from_conflayers_style(args.conflayers_json, "ConfLayers", args.model_id,
                                      baseline_path=args.vanilla_json, seed=args.seed,
                                      task=args.task, cnndm_golds=cnndm_golds),
        profile_from_conflayers_style(args.swift_json, "SWIFT", args.model_id,
                                      baseline_path=args.vanilla_json, seed=args.seed,
                                      task=args.task, cnndm_golds=cnndm_golds),
    ]

    report = build_report(profiles, query_volumes=args.query_volumes)
    save_report(report, args.out)

    print(f"\nSaved audit report -> {args.out}\n")
    print("=" * 70)
    print("GATE CHECK -- search_overhead_fraction (must be checked before writing anything):")
    for name, m in report["methods"].items():
        frac = m["search_overhead_fraction"]
        flag = "  <-- premise may be dead if this is small" if frac < 0.05 else ""
        print(f"  {name:12s}: {frac:6.1%}{flag}")
    print("=" * 70)
    print("CROSSOVERS:")
    for c in report["crossovers"]:
        print(f"  {c['method_a']} vs {c['method_b']}: N* = {c['crossover_query_volume']}"
              f"  -- {c['interpretation']}")
    print("=" * 70)


if __name__ == "__main__":
    main()