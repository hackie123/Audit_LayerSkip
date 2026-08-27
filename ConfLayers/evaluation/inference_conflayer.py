"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse

from fastchat.utils import str_to_torch_dtype

from evaluation.eval import run_eval

from transformers import AutoTokenizer
from bayes_opt import BayesianOptimization, UtilityFunction

from model.conflayers.utils import *
from model.conflayers.modeling_llama import LlamaForCausalLM
from model.conflayers.modeling_qwen2 import Qwen2ForCausalLM
from model.conflayers.kv_cache import initialize_past_key_values

def swift_forward(input_ids, model, tokenizer, max_new_tokens, statistics=None, optimizer=None, utility=None,
                  logits_processor=None, max_steps=512):
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    # Avoid modifying the input_ids in-place
    input_ids = input_ids.clone()
    accept_length_list = []

    # Initialize the past key and value states
    (
        past_key_values,
        past_key_values_data,
        current_length_data,
    ) = initialize_past_key_values(model.model)
    model.past_key_values = past_key_values
    model.past_key_values_data = past_key_values_data
    model.current_length_data = current_length_data

    input_len = input_ids.shape[1]
    cur_length = input_len
    reset_swift_mode(model)
    swift_logits, sample_token, top1_prob = initialize_swift(input_ids, model, max_new_tokens,
                                                             past_key_values, past_key_values_data,
                                                             current_length_data, logits_processor=logits_processor)

    # Clone the prefilled past key and value states for swift optimization
    input_past_key_values_data = []
    for i in range(len(past_key_values_data)):
        input_past_key_values_data.append(past_key_values_data[i].clone())
    input_current_length_data = current_length_data.clone()

    new_token_num = 0
    draft_token_num = 0
    total_acc_num = 0
    layers_skipped= model.get_skip_layers()
    skip_layer_num = int((model.config.num_hidden_layers - 2) * statistics["skip_ratio"])+1
    optimization = True
    for idx in range(max_steps):
        # drafted tokens + 1 bonus verified token
        draft_token_num += len(top1_prob)
        # Initialize the swift buffer
        swift_choices = eval(f"{get_choices_list(top1_prob, logits_processor=logits_processor)}")
        swift_buffers = generate_swift_buffers(swift_choices, device=model.model.layers[-1].self_attn.q_proj.weight.device)
        model.swift_buffers = swift_buffers
        model.swift_choices = swift_choices
        model.model.swift_mask = swift_buffers["swift_attn_mask"]

        candidates, cart_candidates_prob, tree_candidates = generate_candidates(
            swift_logits,
            swift_buffers["tree_indices"],
            swift_buffers["retrieve_indices"],
            sample_token,
            logits_processor
        )

        logits, outputs = tree_decoding(
            model,
            tree_candidates,
            past_key_values,
            swift_buffers["swift_position_ids"],
            input_ids,
            swift_buffers["retrieve_indices"],
        )

        best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor, cart_candidates_prob, swift_logits[2],
                swift_buffers["p_indices"], tree_candidates, swift_buffers["b_indices"]
            )

        input_ids, new_token_num, sample_token = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            swift_buffers["retrieve_indices"],
            logits_processor,
            new_token_num,
            past_key_values_data,
            current_length_data,
            sample_p
        )

        accept_length_tree = input_ids.shape[1] - cur_length
        cur_length = accept_length_tree + cur_length
        accept_length_list.append(accept_length_tree)
        total_acc_num += accept_length_tree - 1

        # layer set optimization
        if statistics["optimization"] is not None and new_token_num > (statistics["context_window"] + 1):
            statistics["origin_score"].append(((accept_length_tree - 1)/len(top1_prob)))
            statistics["origin_acc"].append(accept_length_tree-1)
            if statistics["opt_iter"] > 0 and len(statistics["origin_score"]) == statistics["search_interval"]/2 :
                if len(layers_skipped)>=skip_layer_num:
                    ret_score = np.mean(statistics["origin_score"])
                    acc_score = np.mean(statistics["origin_acc"])
                    if acc_score < 1:
                        statistics["origin_score"]=[]
                        statistics["origin_acc"] = []
                        optimization=True
            if statistics["opt_iter"] > 0 and len(statistics["origin_score"]) > statistics["search_interval"] :
                if len(layers_skipped)>=skip_layer_num:
                    ret_score = np.mean(statistics["origin_score"])
                    acc_score = np.mean(statistics["origin_acc"])
                    if (acc_score > 4 and ret_score > statistics["max_score"]): 
                        statistics["optimization"] = None
                        optimization = False  # stop optimization
                        logging.info("\n" + "=" * 30 + 'Optimization Stopped because the score reaches the expected number!' + "=" * 30 + "\n")
                        logging.info("Mean # Acc Tokens {}, Score = {} and Best Nbr of Layers Skipped: {}".format(acc_score, ret_score, len(layers_skipped)))
                        logging.info("Optimal Layers Set: {}".format(layers_skipped))
                    elif acc_score > statistics["best_acc"]: 
                        logging.info("\n" + "=" * 30 + 'New best found! Mean # Acc Tokens {}, Score = {} and Layers skipped = {}'.format(acc_score, ret_score, layers_skipped) + "=" * 30 + "\n")
                        statistics["best_score"] = ret_score
                        statistics["best_acc"] = acc_score
                        statistics["opt_set"] = layers_skipped
                        optimization = True
                    else:
                        optimization = True
                else:
                    optimization = True
                statistics["origin_score"]=[]
                statistics["origin_acc"]=[]

            if optimization or len(layers_skipped)<skip_layer_num:
                layers_skipped= dynamic_optimization(
                    model,
                    input_ids[:, input_len:],
                    input_past_key_values_data,
                    input_current_length_data,
                    statistics,
                    statistics["opt_set"])               
                statistics["origin_score"]=[]
                statistics["origin_acc"]=[]
                optimization = False

        # swift drafting
        swift_logits, top1_prob = swift_draft(
            model,
            input_ids=sample_token,
            new_token_num=new_token_num,
            past_key_values_data=past_key_values_data,
            current_length_data=current_length_data,
            max_new_tokens=max_new_tokens,
            logits_processor=logits_processor,
        )
        
        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token_num > max_new_tokens:
            break
    logging.info("token acceptance rate: {}".format(total_acc_num / draft_token_num))
    return input_ids, new_token_num, idx + 1, accept_length_list, draft_token_num


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for ConfLayers sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.85,
        help="The top-p for sampling.",
    )
    parser.add_argument(
        "--skip-ratio",
        type=float,
        default=0.45,
        help="The skipped layer ratio of ConfLayers.",
    )
    parser.add_argument(
        "--search-interval",
        type=int,
        default=25,
        help="The interval of optimization.",
    )
    parser.add_argument(
        "--max-opt-iter",
        type=int,
        default=1000,
        help="The maximum layer set optimization iteration.",
    )
    parser.add_argument(
        "--max-score",
        type=float,
        default=0.95,
        help="The early stop threshold of layer set search.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=32,
        help="The context window of ConfLayers.",
    )
    parser.add_argument(
        "--optimization",
        type=str,
        default="None",
        choices=["swift", "conflayers"],
        help="Layer set optimization method.",
    )
    parser.add_argument(
        "--cache-hit",
        action="store_true",
        default=False,
        help="Whether to use cached ConfLayers configuration.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--data-num",
        type=int,
        default=10,
        help="The number of samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="The sampling seed.",
    )

    args = parser.parse_args()

    args.model_name = (args.model_id + "-" + str(args.optimization) + "-" + str(args.dtype)+ "-temp-" + str(args.temperature)
                       + "-top-p-" + str(args.top_p) + "-seed-" + str(args.seed) + "-max_new_tokens-" + str(args.max_new_tokens)
                       + "-search_interval-" + str(args.search_interval) + "-max_opt-" + str(args.max_opt_iter) 
                       + "-max_score-" + str(args.max_score) + "-context_window-" + str(args.context_window) + "-skip_ratio-" + str(args.skip_ratio))
    answer_file = f"outputs/{args.task_name}/{args.task_name}_{args.data_num}/model_answer/{args.model_id}/{args.model_name}.jsonl"
    set_logger()

    print(f"Output to {answer_file}")

    torch.nn.Linear.reset_parameters = lambda x: None

    if "llama" in args.model_id:
        model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto")
    elif "qwen" in args.model_id:
        model = Qwen2ForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=str_to_torch_dtype(args.dtype),
            low_cpu_mem_usage=True,
            device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=args.temperature, top_p=args.top_p)
    else:
        logits_processor = None
    if args.cache_hit:
        # Load the cached layer set configuration
        args.optimization=False
        _skip_layer_id_set = get_cache_configuration(model_name=args.model_id,
                                                                                  task_name=args.task_name)
    else:
        # Unified layer set initialization
        _skip_layer_id_set = np.arange(1, model.config.num_hidden_layers - 1, 2)  # keep the first and last layer

    model.set_skip_layers(_skip_layer_id_set)

    # Bayes Optimization Settings
    pbounds = {f"x{i}": (0, 1) for i in range((model.config.num_hidden_layers - 2) * 2)} # keep the first and last layer
    optimizer = BayesianOptimization(f=None, pbounds=pbounds, random_state=1, verbose=1, allow_duplicate_points=True)
    optimizer.set_gp_params(alpha=1e-2)
    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

    statistics = {"model_id": args.model_id, "origin_score": [], "opt_iter": 0, 
                  "skip_ratio": args.skip_ratio, "acceptance_rate_list": [],
                  "search_interval": args.search_interval, "max_opt_iter": args.max_opt_iter,
                  "max_score": args.max_score,
                  "context_window": args.context_window, "optimization": args.optimization, 
                  "best_score": 0, "best_acc": 0, "opt_set": [], "lambda": 0, "origin_acc" : []}

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=swift_forward,
        model_id=args.model_id,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        task_name=args.task_name,
        data_num=args.data_num,
        seed=args.seed,
        optimizer=optimizer,
        utility=utility,
        statistics=statistics,
        logits_processor=logits_processor,
    )
