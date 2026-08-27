#!/usr/bin/env python3
"""
Compute ROUGE-2 for two methods against a baseline.

The baseline file's generated text is treated as the reference.
Each method is scored against it, entry-by-entry (matched by line order).

Prediction text is read from: choices[0].turns
  - `turns` may be a string or a list of strings (joined with spaces).

Files are JSONL (one JSON object per line). If a file is instead a single
JSON array, that is handled too.

Usage:
    python rouge.py baseline.jsonl method1.jsonl
    
"""

import argparse
import difflib
import json
import re
import sys
from collections import Counter


# ----------------------------- IO helpers ---------------------------------- #

def load_records(path):
    """Load a JSONL file (or a single JSON array) into a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    # Try JSONL first (the common case for these eval dumps).
    records = []
    looks_like_jsonl = False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
            looks_like_jsonl = True
        except json.JSONDecodeError:
            looks_like_jsonl = False
            break
    if looks_like_jsonl:
        return records
    # Fall back to a single JSON value (array or object).
    obj = json.loads(text)
    return obj if isinstance(obj, list) else [obj]


def extract_prediction(record):
    """Pull the generated text out of choices[0].turns."""
    choices = record.get("choices") or []
    if not choices:
        return ""
    turns = choices[0].get("turns", "")
    if isinstance(turns, list):
        return " ".join(str(t) for t in turns)
    return str(turns)


# --------------------------- ROUGE-2 core ---------------------------------- #

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text):
    """Lowercase word tokenization (unicode-aware, keeps umlauts etc.)."""
    return _TOKEN_RE.findall(text.lower())


def bigrams(tokens):
    return Counter(zip(tokens, tokens[1:]))


def rouge2(prediction, reference):
    """Return (precision, recall, f1) for ROUGE-2 (bigram overlap)."""
    pred_bg = bigrams(tokenize(prediction))
    ref_bg = bigrams(tokenize(reference))

    pred_total = sum(pred_bg.values())
    ref_total = sum(ref_bg.values())

    if pred_total == 0 or ref_total == 0:
        return 0.0, 0.0, 0.0

    overlap = sum((pred_bg & ref_bg).values())  # multiset intersection
    precision = overlap / pred_total
    recall = overlap / ref_total
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1

# --------------------------- Inline highlight ------------------------------- #
 
# ANSI colors, auto-disabled when output is not a terminal.
_COLOR = sys.stdout.isatty()
 
 
def _wrap(code, text):
    return f"\033[{code}m{text}\033[0m" if _COLOR else text
 
 
def inline_highlight(prediction, reference):
    """Return (ref_text, pred_text) with mismatching spans highlighted inline.
 
    Reference-only spans -> red, prediction-only spans -> green, shared -> dim.
    When not a terminal, falls back to [-...-] / [+...+] markers.
    """
    ref_words = reference.split()
    pred_words = prediction.split()
    sm = difflib.SequenceMatcher(a=ref_words, b=pred_words, autojunk=False)
 
    ref_out, pred_out = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        ref_chunk = " ".join(ref_words[i1:i2])
        pred_chunk = " ".join(pred_words[j1:j2])
        if tag == "equal":
            ref_out.append(_wrap("2", ref_chunk))
            pred_out.append(_wrap("2", pred_chunk))
        else:  # delete / insert / replace
            if ref_chunk:
                ref_out.append(_wrap("41;97", ref_chunk if _COLOR else f"[-{ref_chunk}-]"))
            if pred_chunk:
                pred_out.append(_wrap("42;30", pred_chunk if _COLOR else f"[+{pred_chunk}+]"))
    return " ".join(ref_out), " ".join(pred_out)
 
 
# ------------------------------ Driver ------------------------------------- #

def score_method(method_records, baseline_records, n):
    """Score one method's records against the baseline references."""
    m = len(method_records)
    b = len(baseline_records)
    results = []
    for i in range(n):
        # print("\nNew")
        ref = extract_prediction(baseline_records[b-i-1])
        # print("\nRef:\n", ref)
        pred = extract_prediction(method_records[m-i-2])
        # print("\nPred:\n",pred)
        p, r, f = rouge2(pred, ref)
        results.append((p, r, f))
        ref_hl, pred_hl = inline_highlight(pred, ref)
        print(f"\nEntry {i}  |  ROUGE-2 P/R/F1 = {p:.4f} / {r:.4f} / {f:.4f}")
        print(f"  REF : {ref_hl}")
        print(f"  PRED: {pred_hl}")
    return results


def average(results):
    if not results:
        return 0.0, 0.0, 0.0
    n = len(results)
    p = sum(r[0] for r in results) / n
    r_ = sum(r[1] for r in results) / n
    f = sum(r[2] for r in results) / n
    return p, r_, f


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline", help="JSONL file used as the reference")
    ap.add_argument("method1", help="First method JSONL file")
    ap.add_argument("--csv", help="Optional path to write per-entry scores as CSV")
    ap.add_argument("--n", type=int)

    args = ap.parse_args()

    baseline = load_records(args.baseline)
    m1 = load_records(args.method1)
    n = args.n

    print(f"Loaded: baseline={len(baseline)}  method1={len(m1)} entries")
    if not (len(baseline) == len(m1)):
        print("  ! File lengths differ; comparing up to the shortest by line order.\n")

    methods = [("method1", args.method1, m1)]
    all_results = {}

    for name, path, recs in methods:
        results = score_method(recs, baseline, n)
        all_results[name] = results
        p, r, f = average(results)
        print(f"\n{name}  ({path})")
        print(f"  entries scored : {len(results)}")
        print(f"  ROUGE-2 P / R / F1 (avg): {p:.4f} / {r:.4f} / {f:.4f}")

    # Optional per-entry CSV dump.
    if args.csv:
        n = min(len(all_results["method1"]), len(all_results["method2"]))
        with open(args.csv, "w", encoding="utf-8") as out:
            out.write("index,m1_p,m1_r,m1_f1,m2_p,m2_r,m2_f1\n")
            for i in range(n):
                a = all_results["method1"][i]
                b = all_results["method2"][i]
                out.write(f"{i},{a[0]:.6f},{a[1]:.6f},{a[2]:.6f},"
                          f"{b[0]:.6f},{b[1]:.6f},{b[2]:.6f}\n")
        print(f"\nWrote per-entry scores to {args.csv}")


if __name__ == "__main__":
    main()