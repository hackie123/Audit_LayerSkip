
MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
MODEL_NAME="qwen2.5-1.5b"
TEMP=0.0
TOP_P=0.85
DATA_NUM=100
SEED=123
GPU_DEVICES=0
MAX_NEW_TOKENS=512
torch_dtype="bfloat16" # ["float32", "float64", "float16", "bfloat16"]
TASK_NAME="cnndm" # ["cnndm", "gsm8k", "wmt14_de-en", "alpaca"]

# Dynamic Hyperparameters
SEARCH_INTERVAL=30
MAX_OPT_ITER=100
MAX_SCORE=0.95
CONTEXT_WINDOW=100
SKIP_RATIO=0.4
OPT_METHOD="conflayers"

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_conflayer --model-path $MODEL_PATH --model-id ${MODEL_NAME} --temperature $TEMP --top-p ${TOP_P} --dtype $torch_dtype --task-name ${TASK_NAME} --data-num ${DATA_NUM} --max-new-tokens ${MAX_NEW_TOKENS} --seed $SEED --context-window ${CONTEXT_WINDOW} --search-interval ${SEARCH_INTERVAL} --max-opt-iter ${MAX_OPT_ITER} --max-score ${MAX_SCORE} --skip-ratio ${SKIP_RATIO} --optimization ${OPT_METHOD} #--cache-hit
