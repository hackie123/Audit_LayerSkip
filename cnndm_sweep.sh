#!/bin/bash
# cnndm_sweep.sh
# ---------------------------------------------------------------------------
# Second-task check for the audit: reruns the 4-method comparison on CNN/DM
# summarization instead of GSM8K, to test whether the GSM8K finding
# (LayerRoute acc~0.4-0.46 vs ConfLayers/SWIFT~0.05-0.10) is task-general or
# a GSM8K-specific artifact. Quality metric is ROUGE-L (exact-match makes no
# sense for free-text summaries).
#
# Prereqs: build_audit_report.py and run_layerroute_timed.py must already be
# the CNN/DM-aware versions (see cnndm_fix/ files). eval.sh's SKIP_RATIO/
# MODEL_PATH/torch_dtype are left as-is (only SEED/OPT_METHOD/TASK_NAME
# change here) -- confirm they're still 0.4 / Qwen2.5-1.5B-Instruct / bfloat16
# before running.
#
# rouge-score must be installed in BOTH environments (system Python for
# ConfLayers, layerroute_venv for LayerRoute/build_audit_report):
#   pip3 install rouge-score                          (system)
#   source /root/layerroute_venv/bin/activate && pip install rouge-score
#
# Usage: bash cnndm_sweep.sh
# ---------------------------------------------------------------------------
set -e

SEEDS=(2024 42)   # start with 2 seeds for the second-task check; add more if promising
WORKSPACE=/workspace
CONFLAYERS_DIR=$WORKSPACE/ConfLayers
LAYERROUTE_DIR=$WORKSPACE/layerroute
RESULTS_DIR=$WORKSPACE/results/cnndm_sweep
mkdir -p "$RESULTS_DIR"

export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface

for SEED in "${SEEDS[@]}"; do
  echo "=============================================================="
  echo " CNN/DM SEED $SEED -- starting"
  echo "=============================================================="

  # -------------------------------------------------------------------
  # Stage A: ConfLayers/SWIFT/vanilla (system Python env)
  # -------------------------------------------------------------------
  cd "$CONFLAYERS_DIR"

  OUT_DIR="outputs/cnndm/cnndm_100/model_answer/qwen2.5-1.5b"
  TEST_DIR="test/cnndm/cnndm_100/model_answer/qwen2.5-1.5b"
  mkdir -p "$OUT_DIR" "$TEST_DIR"

  echo "--- [$SEED] vanilla baseline (cnndm) ---"
  python -m evaluation.inference_baseline \
    --model-path Qwen/Qwen2.5-1.5B-Instruct \
    --model-id qwen2.5-1.5b \
    --temperature 0.0 --top-p 0.85 --dtype bfloat16 \
    --task-name cnndm --data-num 100 --max-new-tokens 512 \
    --seed "$SEED" --num-gpus-per-model 1 --num-gpus-total 1

  VANILLA_RAW="$TEST_DIR/qwen2.5-1.5b-vanilla-bfloat16-temp-0.0-top-p-0.85-seed-${SEED}-max_new_tokens-512.jsonl"
  VANILLA_CLEAN="$TEST_DIR/vanilla_clean_seed${SEED}.jsonl"
  # NOTE: verify this filename against the actual "Output to ..." line printed
  # by inference_baseline.py -- task-name in the filename may differ; adjust
  # if the head -n 100 step below errors with "No such file".
  head -n 100 "$VANILLA_RAW" > "$VANILLA_CLEAN"

  echo "--- [$SEED] ConfLayers (cnndm) ---"
  sed -i -e "s|^SEED=.*|SEED=${SEED}|" \
         -e 's|OPT_METHOD=".*"|OPT_METHOD="conflayers"|' \
         -e 's|TASK_NAME="[^"]*"|TASK_NAME="cnndm"|' \
         eval.sh
  chmod +x eval.sh
  ./eval.sh
  CONFLAYERS_RAW="$OUT_DIR/qwen2.5-1.5b-conflayers-bfloat16-temp-0.0-top-p-0.85-seed-${SEED}-max_new_tokens-512-search_interval-30-max_opt-100-max_score-0.95-context_window-100-skip_ratio-0.4.jsonl"
  CONFLAYERS_CLEAN="$OUT_DIR/conflayers_clean_seed${SEED}.jsonl"
  head -n 100 "$CONFLAYERS_RAW" > "$CONFLAYERS_CLEAN"

  echo "--- [$SEED] SWIFT (cnndm) ---"
  sed -i 's|OPT_METHOD=".*"|OPT_METHOD="swift"|' eval.sh
  ./eval.sh
  SWIFT_RAW="$OUT_DIR/qwen2.5-1.5b-swift-bfloat16-temp-0.0-top-p-0.85-seed-${SEED}-max_new_tokens-512-search_interval-30-max_opt-100-max_score-0.95-context_window-100-skip_ratio-0.4.jsonl"
  SWIFT_CLEAN="$OUT_DIR/swift_clean_seed${SEED}.jsonl"
  head -n 100 "$SWIFT_RAW" > "$SWIFT_CLEAN"

  # -------------------------------------------------------------------
  # Stage B: LayerRoute (separate venv)
  # -------------------------------------------------------------------
  cd "$LAYERROUTE_DIR"
  echo "--- [$SEED] LayerRoute eval (cnndm) ---"
  source /root/layerroute_venv/bin/activate

  python run_layerroute_timed.py \
    --ckpt checkpoints/best_adapters.pt \
    --dataset cnndm \
    --n_eval 100 \
    --max_new_tokens 300 \
    --seed "$SEED" \
    --train_wall_clock_seconds 1015 \
    --out_dir "results/layerroute_timed_cnndm_seed${SEED}"

  deactivate

  # -------------------------------------------------------------------
  # Stage C: build the report for this seed
  # -------------------------------------------------------------------
  cd "$WORKSPACE"
  source /root/layerroute_venv/bin/activate

  python build_audit_report.py \
    --conflayers_json "$CONFLAYERS_DIR/$CONFLAYERS_CLEAN" \
    --swift_json "$CONFLAYERS_DIR/$SWIFT_CLEAN" \
    --layerroute_json "$LAYERROUTE_DIR/results/layerroute_timed_cnndm_seed${SEED}/layerroute_query_costs.json" \
    --vanilla_json "$CONFLAYERS_DIR/$VANILLA_CLEAN" \
    --seed "$SEED" \
    --task cnndm \
    --out "$RESULTS_DIR/audit_report_cnndm_seed${SEED}.json"

  deactivate

  echo "=============================================================="
  echo " CNN/DM SEED $SEED -- DONE. Report: $RESULTS_DIR/audit_report_cnndm_seed${SEED}.json"
  echo "=============================================================="
done

echo ""
echo "All CNN/DM seeds complete. Summary (quality_score = ROUGE-L F-measure):"
for SEED in "${SEEDS[@]}"; do
  echo "--- seed $SEED ---"
  python3 -c "
import json
r = json.load(open('$RESULTS_DIR/audit_report_cnndm_seed${SEED}.json'))
for name, m in r['methods'].items():
    print(f\"  {name:12s} rougeL={m['quality_score']:.3f}  pure_ms={m['pure_inference_ms_per_query']:.1f}\")
"
done