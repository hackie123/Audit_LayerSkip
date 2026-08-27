#!/bin/bash
# multi_seed_sweep.sh
# ---------------------------------------------------------------------------
# Reruns the full 4-method comparison (vanilla, ConfLayers, SWIFT, LayerRoute)
# across NEW seeds, to check whether the pilot result (LayerRoute acc=0.44 vs
# ConfLayers/SWIFT acc=0.05) survives beyond a single seed.
#
# NOTE: only the EVAL seed varies here -- the LayerRoute router checkpoint
# (best_adapters.pt) is reused as-is, since retraining per seed is expensive
# and out of scope for this quick multi-seed check. This tests "does the
# accuracy gap hold across different held-out question samples", not "is the
# router itself seed-sensitive" -- a fair, cheaper first check.
#
# Usage: bash multi_seed_sweep.sh
# Stops immediately on any error (set -e) so you see exactly where it broke.
# ---------------------------------------------------------------------------
set -e

SEEDS=(42 123)
WORKSPACE=/workspace
CONFLAYERS_DIR=$WORKSPACE/ConfLayers
LAYERROUTE_DIR=$WORKSPACE/layerroute
RESULTS_DIR=$WORKSPACE/results/multi_seed
mkdir -p "$RESULTS_DIR"

export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface

for SEED in "${SEEDS[@]}"; do
  echo "=============================================================="
  echo " SEED $SEED -- starting"
  echo "=============================================================="

  # -------------------------------------------------------------------
  # Stage A: ConfLayers/SWIFT/vanilla (system Python env)
  # -------------------------------------------------------------------
  cd "$CONFLAYERS_DIR"

  OUT_DIR="outputs/gsm8k/gsm8k_100/model_answer/qwen2.5-1.5b"
  TEST_DIR="test/gsm8k/gsm8k_100/model_answer/qwen2.5-1.5b"
  mkdir -p "$OUT_DIR" "$TEST_DIR"

  echo "--- [$SEED] vanilla baseline ---"
  python -m evaluation.inference_baseline \
    --model-path Qwen/Qwen2.5-1.5B-Instruct \
    --model-id qwen2.5-1.5b \
    --temperature 0.0 --top-p 0.85 --dtype bfloat16 \
    --task-name gsm8k --data-num 100 --max-new-tokens 512 \
    --seed "$SEED" --num-gpus-per-model 1 --num-gpus-total 1

  VANILLA_RAW="$TEST_DIR/qwen2.5-1.5b-vanilla-bfloat16-temp-0.0-top-p-0.85-seed-${SEED}-max_new_tokens-512.jsonl"
  VANILLA_CLEAN="$TEST_DIR/vanilla_clean_seed${SEED}.jsonl"
  head -n 100 "$VANILLA_RAW" > "$VANILLA_CLEAN"

  echo "--- [$SEED] ConfLayers ---"
  sed -i -e "s|^SEED=.*|SEED=${SEED}|" \
         -e 's|OPT_METHOD=".*"|OPT_METHOD="conflayers"|' \
         eval.sh
  chmod +x eval.sh
  ./eval.sh
  CONFLAYERS_RAW="$OUT_DIR/qwen2.5-1.5b-conflayers-bfloat16-temp-0.0-top-p-0.85-seed-${SEED}-max_new_tokens-512-search_interval-30-max_opt-100-max_score-0.95-context_window-100-skip_ratio-0.4.jsonl"
  CONFLAYERS_CLEAN="$OUT_DIR/conflayers_clean_seed${SEED}.jsonl"
  head -n 100 "$CONFLAYERS_RAW" > "$CONFLAYERS_CLEAN"

  echo "--- [$SEED] SWIFT ---"
  sed -i 's|OPT_METHOD=".*"|OPT_METHOD="swift"|' eval.sh
  ./eval.sh
  SWIFT_RAW="$OUT_DIR/qwen2.5-1.5b-swift-bfloat16-temp-0.0-top-p-0.85-seed-${SEED}-max_new_tokens-512-search_interval-30-max_opt-100-max_score-0.95-context_window-100-skip_ratio-0.4.jsonl"
  SWIFT_CLEAN="$OUT_DIR/swift_clean_seed${SEED}.jsonl"
  head -n 100 "$SWIFT_RAW" > "$SWIFT_CLEAN"

  # -------------------------------------------------------------------
  # Stage B: LayerRoute (separate venv -- newer transformers)
  # -------------------------------------------------------------------
  cd "$LAYERROUTE_DIR"
  echo "--- [$SEED] LayerRoute eval ---"
  source /root/layerroute_venv/bin/activate

  python run_layerroute_timed.py \
    --ckpt checkpoints/best_adapters.pt \
    --dataset gsm8k \
    --n_eval 100 \
    --seed "$SEED" \
    --train_wall_clock_seconds 1015 \
    --out_dir "results/layerroute_timed_seed${SEED}"

  deactivate

  # -------------------------------------------------------------------
  # Stage C: build the report for this seed
  # -------------------------------------------------------------------
  cd "$WORKSPACE"
  source /root/layerroute_venv/bin/activate   # build_audit_report only needs stdlib, either env works

  python build_audit_report.py \
    --conflayers_json "$CONFLAYERS_DIR/$CONFLAYERS_CLEAN" \
    --swift_json "$CONFLAYERS_DIR/$SWIFT_CLEAN" \
    --layerroute_json "$LAYERROUTE_DIR/results/layerroute_timed_seed${SEED}/layerroute_query_costs.json" \
    --vanilla_json "$CONFLAYERS_DIR/$VANILLA_CLEAN" \
    --seed "$SEED" \
    --out "$RESULTS_DIR/audit_report_seed${SEED}.json"

  deactivate

  echo "=============================================================="
  echo " SEED $SEED -- DONE. Report: $RESULTS_DIR/audit_report_seed${SEED}.json"
  echo "=============================================================="
done

echo ""
echo "All seeds complete. Summary:"
for SEED in "${SEEDS[@]}"; do
  echo "--- seed $SEED ---"
  python3 -c "
import json
r = json.load(open('$RESULTS_DIR/audit_report_seed${SEED}.json'))
for name, m in r['methods'].items():
    print(f\"  {name:12s} acc={m['quality_score']:.2f}  pure_ms={m['pure_inference_ms_per_query']:.1f}\")
"
done
