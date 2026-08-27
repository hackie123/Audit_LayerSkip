# Audit: Matched-Cost Comparison of Trained vs. Heuristic Self-Speculative Decoding

## What this is

An audit of self-speculative-decoding / layer-skipping methods that fixes a
methodological gap confirmed by direct inspection of the ConfLayers/SWIFT
codebase: **every prior paper's reported "speedup" folds the online
Bayesian-optimization search cost into a single wall-clock number**, while a
trained router (LayerRoute) pays an equivalent cost entirely upfront, once,
at training time. No paper compares these on equal footing across query
volume. That's the gap this audit closes.

## Confirmed from the actual code (not just the papers)

- `ConfLayers/README.md`: "This codebase is built on SWIFT" -- one harness
  covers both baselines.
- `evaluation/inference_conflayer.py` line 11: `from bayes_opt import
  BayesianOptimization` -- the search runs live, inside generation.
- `evaluation/eval.py` lines 169-180: `wall_time` is measured as
  `torch.cuda.synchronize(); start=time.time(); <full generation incl.
  search>; wall_time = time.time()-start` -- search cost is never isolated.
- The search-active branch (`statistics["optimization"]`) can remain active
  for many decoding steps, or the whole generation, depending on the
  "Optimization Stopped" condition (`inference_conflayer.py` ~line 118) --
  meaning the hidden cost is INPUT-DEPENDENT, not a fixed overhead.

## Files

- `harness/timing_split.py` -- the core measurement. `measure_frozen_replay()`
  runs each prompt TWICE: once normally (search active, produces a final
  layer-skip config), once with that config FROZEN and search disabled. The
  wall-clock difference is the true search overhead. This is the only
  non-invasive way to split the two costs, because in the original code the
  BayesOpt candidate evaluations ARE real generation steps -- the search and
  the "useful" work are algorithmically interleaved, not separable by simply
  reading a flag.
- `harness/cost_accounting.py` -- turns per-query (pure_inference_ms,
  search_overhead_ms) measurements plus a one-time training cost into an
  amortized total-cost-per-query curve over query volume N, and finds the
  crossover N* where method rankings flip. Both modules are unit-tested here
  (CPU, mocked) -- see the assertions in each file's test block; they must be
  re-validated against real measurements on the pod.

## Reproduction plan (in order -- do not skip steps)

### Step 0: Environment
```bash
git clone https://github.com/WA225/ConfLayers.git
cd ConfLayers
conda env create -f environment.yml
conda activate conflayers
pip3 install torch torchvision   # use the CUDA index for your pod, not ROCm
hf auth login
```

### Step 1: Reproduce ConfLayers' own numbers FIRST (sanity gate)
Run `eval.sh` on Llama-2-13B or swap in TinyLlama-1.1B (edit `--model-path`)
on GSM8K. Confirm you get a speedup in the same ballpark as their paper
(~1.2-1.4x). **Do not proceed until this reproduces.** If it doesn't, the
audit's credibility is compromised before it starts -- this is the #1 failure
mode for audit papers (see "Learning Rate Matters" precedent: they explicitly
validated they could reproduce prior claims before re-evaluating them).

### Step 2: Wire in the timing split
Import `harness/timing_split.py` into ConfLayers' `evaluation/eval.py`. Wrap
each query's call to `swift_forward` with `measure_frozen_replay(...)`
instead of the single-pass call currently there. This doubles wall-clock time
per query (you generate twice) -- budget for that; it does not change the
generation itself for either method.

Output per query: a `QueryCostRecord` with `pure_inference_ms` and
`search_overhead_ms` populated from the frozen-replay difference.

### Step 3: Wire in LayerRoute on the same prompts
Run your existing LayerRoute inference path on the SAME GSM8K prompts used
in Step 1-2. Record: one-time router training wall-clock (you already have
this from training logs), and per-query `pure_inference_ms` (search_overhead
= 0 by construction, since the router makes its decision in one forward
pass, no iterative search).

### Step 4: Build the cost profiles and run the report
```python
from harness.cost_accounting import MethodCostProfile, build_report, save_report

layerroute = MethodCostProfile(
    name="LayerRoute", model_id="TinyLlama-1.1B",
    train_cost_seconds=<measured>,
    pure_inference_ms_per_query=<mean over queries>,
    search_overhead_ms_per_query=0.0,
    baseline_ms_per_query=<vanilla decoding mean>,
    quality_score=<exact_match or ROUGE>,
)
conflayers = MethodCostProfile(
    name="ConfLayers", model_id="TinyLlama-1.1B",
    train_cost_seconds=0.0,
    pure_inference_ms_per_query=<mean pure_inference_ms from Step 2>,
    search_overhead_ms_per_query=<mean search_overhead_ms from Step 2>,
    baseline_ms_per_query=<same vanilla baseline as above>,
    quality_score=<exact_match or ROUGE>,
)
report = build_report([layerroute, conflayers], query_volumes=[1, 10, 100, 1000, 10000, 100000])
save_report(report, "results/audit_report.json")
```

Repeat for SWIFT (same codebase, `inference_baseline.py` path) and optionally
DEL as the "known to underperform" sanity check.

### Step 5: Read the crossover
- If `crossover_query_volume` is `None`: one method dominates at all realistic
  deployment volumes -- report which, and why (this is still a valid,
  publishable finding: "amortization never matters in practice because X").
- If a crossover exists in a realistic range (compare against actual API
  deployment volumes -- thousands to millions of queries/day is typical):
  **this is the headline result.** Report the phase diagram: at low volume,
  training-free heuristics win (matches the field's current belief); at
  production volume, the cheaply-trained router wins, and nobody has shown
  this because nobody separated the two costs.

## What would make this NOT worth publishing (negative-result gate)

Check these before writing anything:
- If `search_overhead_ms_per_query` is negligible (<5% of total) once
  properly isolated, the entire premise dissolves -- the field's numbers were
  fine all along, and this becomes a short correctness note, not a paper.
- If quality_score for the frozen-replay pass differs meaningfully from the
  original (it shouldn't, since we freeze the config the search WOULD have
  converged to) -- that would mean the search is doing continuous adaptation
  across the query, not one-time configuration, and the whole measurement
  approach needs rethinking.
- If LayerRoute's per-query pure_inference_ms is NOT meaningfully lower than
  ConfLayers' (they should be similar, since ConfLayers/SWIFT skip a similar
  layer fraction) -- if it's much worse, LayerRoute may need retraining/
  retuning before it's a fair comparison point at all.

## Scale plan (single A5000)
- Discovery: TinyLlama-1.1B, GSM8K + CNN/DM (matches ConfLayers' own tasks),
  ~100 queries, 3 seeds.
- Confirmation: Qwen2.5-1.5B (you already have this pipeline from the
  reasoning experiment), same tasks.
- Volume sweep is ANALYTICAL (the cost_accounting formula), not simulated by
  actually running 100,000 queries -- you only need the PER-QUERY mean cost
  from a modest sample; the amortization curve is then computed exactly.
