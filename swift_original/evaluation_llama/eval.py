"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
# adapted from fastchat: https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_model_answer.py

import copy
import json
import logging
import os
import time
import torch
import random
import numpy as np
import shortuuid

from tqdm import tqdm
from datasets import load_dataset
from human_eval.data import read_problems


def seed_everything(seed=64):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clip_input(tokenizer, prompt, task_name, max_new_tokens=512, tree_length=250, max_output_length=4096, prompt_shots=None):
    end_prompt = ''
    if task_name == 'cnndm':
        inputs = tokenizer(
            prompt_shots + 'Article: ' + prompt['article'] + '\nSummary:',
            return_tensors='pt').to("cuda")
        end_prompt = '\nSummary:'
    elif task_name == 'humaneval':
        prompt = prompt['prompt'].replace("    ", "\t")
        inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
    elif task_name == 'gsm8k':
        # Matches ConfLayers' eval.py exactly (verified against that repo):
        # prompt_shots + "Q: {question}\nA:" -- model completes the answer.
        inputs = tokenizer(
            prompt_shots + f"Q: {prompt['question'].strip()}\nA:",
            return_tensors='pt').to("cuda")
        end_prompt = '\nA:'
    else:
        logging.info("This task is not supported.")
    end_prompt_length = len(tokenizer(end_prompt,return_tensors='pt').input_ids[0])
    input_ids = inputs.input_ids
    if len(input_ids[0]) + max_new_tokens + tree_length >= max_output_length:
        sample_num = (len(input_ids[0]) + max_new_tokens + tree_length - max_output_length)
        input_ids = torch.cat((input_ids[0][:-(end_prompt_length+sample_num)], input_ids[0][-end_prompt_length:]), dim=0).unsqueeze(0)
    return input_ids

def load_data(task_name, seed,  data_num=10):
    data = []
    prompt_shots = ''
    if task_name == 'cnndm':
        n_shot = 1
        data = load_dataset('cnn_dailymail', name='3.0.0', split='test').shuffle(seed=seed).select(range(data_num))
        shots = load_dataset('cnn_dailymail', name='3.0.0', split='train').shuffle(seed=seed).select(range(n_shot))
        prompt_keys = ['article', 'highlights']
        instructions = ['Article: ', '\nSummary: ']
        for i in range(n_shot):
            prompt = instructions[0] + shots[i][prompt_keys[0]] + instructions[1] + shots[i][prompt_keys[1]].replace('\n', '') + '\n'
            prompt_shots += prompt
    elif task_name == 'humaneval':
        original_data = read_problems()
        for i, task_id in enumerate(original_data):
            if i >= data_num:
                break
            data.append(original_data[task_id])
    elif task_name == 'gsm8k':
        # Matches ConfLayers' eval.py exactly (verified against that repo,
        # traced earlier in this project): n_shot=5, same load_dataset call
        # using "gsm8k" (not "openai/gsm8k") -- confirmed both aliases
        # resolve to the identical dataset/shuffle order, so held-out data
        # stays consistent with LayerRoute/ConfLayers' existing results.
        n_shot = 5
        data = load_dataset("gsm8k", "main", split='test').shuffle(seed=seed).select(range(data_num))
        shots = load_dataset("gsm8k", "main", split='train').shuffle(seed=seed).select(range(n_shot))
        for i in range(n_shot):
            prompt_shots += f"Q: {shots[i]['question'].strip()}\nA: {shots[i]['answer'].strip()}\n" + "\n"
    else:
        logging.info("This task is not supported.")
    return data, prompt_shots


def run_eval(
        model,
        tokenizer,
        forward_func,
        model_id,
        answer_file,
        max_new_tokens,
        num_gpus_per_model,
        num_gpus_total,
        task_name,
        data_num,
        seed,
        **kwargs,
):
    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0

    seed_everything(seed)

    data, prompt_shots = load_data(task_name, seed, data_num=data_num)
    get_answers_func = get_model_answers

    get_answers_func(
        model,
        tokenizer,
        forward_func,
        model_id,
        data,
        prompt_shots,
        answer_file,
        max_new_tokens,
        task_name,
        **kwargs,
    )


@torch.inference_mode()
def get_model_answers(
        model,
        tokenizer,
        forward_func,
        model_id,
        data,
        prompt_shots,
        answer_file,
        max_new_tokens,
        task_name,
        **kwargs,
):
    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    accept_lengths_tree = []
    total_draft_num = 0
    for question in tqdm(data):
        choices = []
        input_ids = clip_input(tokenizer, question, task_name, max_new_tokens=max_new_tokens,
                               prompt_shots=prompt_shots, max_output_length=model.config.max_position_embeddings)
        cur_accept_lengths_tree = []
        cur_draft_num = 0
        steps = []
        new_tokens = []
        wall_time = []
        torch.cuda.synchronize()
        start_time = time.time()
        output_ids, new_token_num, step, accept_length_tree, draft_token_num = forward_func(
            input_ids,
            model,
            tokenizer,
            max_new_tokens,
            **kwargs,
        )
        torch.cuda.synchronize()
        total_time = time.time() - start_time

        # ---- FROZEN-REPLAY MEASUREMENT (search-overhead vs pure-inference
        # split) --------------------------------------------------------
        # Re-run the SAME query with search disabled: a deep copy of
        # `statistics` with ["optimization"] forced to None, using
        # whatever skip-layer configuration the model converged to during
        # the real call above (model.get_skip_layers() is read fresh at
        # the top of forward_func and is unchanged by this second call,
        # since the entire search-mutation block -- including
        # dynamic_optimization -- sits inside
        # `if statistics["optimization"] is not None:`, confirmed by
        # direct trace of the decoding loop). Forcing None on a COPY
        # genuinely skips that whole block without modifying the
        # algorithm itself.
        #
        # The frozen call's generated text is discarded -- only its
        # wall-clock is used. The real call's output above remains what
        # gets written to the answer file, so accuracy/text results are
        # completely unaffected by this addition.
        pure_inference_time = total_time  # fallback if frozen replay can't run
        search_overhead_time = 0.0
        if "statistics" in kwargs and kwargs["statistics"] is not None:
            frozen_kwargs = dict(kwargs)
            frozen_statistics = copy.deepcopy(kwargs["statistics"])
            frozen_statistics["optimization"] = None
            frozen_kwargs["statistics"] = frozen_statistics
            torch.cuda.synchronize()
            frozen_start = time.time()
            _ = forward_func(
                input_ids,
                model,
                tokenizer,
                max_new_tokens,
                **frozen_kwargs,
            )
            torch.cuda.synchronize()
            pure_inference_time = time.time() - frozen_start
            search_overhead_time = max(0.0, total_time - pure_inference_time)
        # -----------------------------------------------------------------

        cur_accept_lengths_tree.extend(accept_length_tree)
        cur_draft_num += draft_token_num
        output_ids = output_ids[0][len(input_ids[0]):]

        output = tokenizer.decode(
            output_ids,
            spaces_between_special_tokens=False,
        )
        for special_token in tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                for special_tok in special_token:
                    output = output.replace(special_tok, "")
            else:
                output = output.replace(special_token, "")
        output = output.strip()

        steps.append(int(step))
        new_tokens.append(int(new_token_num))
        wall_time.append(total_time)

        accept_lengths_tree.extend(cur_accept_lengths_tree)
        total_draft_num += cur_draft_num
        choices.append({"turns": output, "decoding_steps": steps, "new_tokens": new_tokens, "wall_time": wall_time,
                        "accept_lengths": cur_accept_lengths_tree,
                        "acceptance_rate": (sum(cur_accept_lengths_tree) - len(
                            cur_accept_lengths_tree)) / cur_draft_num,
                        "pure_inference_time": [pure_inference_time],
                        "search_overhead_time": [search_overhead_time]})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")
        # break
    mean_accepted_tokens = np.mean(accept_lengths_tree)
    if mean_accepted_tokens > 1:
        best_attn_skip_layer_id_set, best_mlp_skip_layer_id_set = model.get_skip_layers()
        best_skip_ratio = (len(best_mlp_skip_layer_id_set) + len(best_attn_skip_layer_id_set)) / ((model.config.num_hidden_layers - 2) * 2)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "Mean accepted tokens": np.mean(accept_lengths_tree),
                "Token acceptance rate": (sum(accept_lengths_tree) - len(accept_lengths_tree)) / total_draft_num,
                "Best Skip Ratio": best_skip_ratio,
                "Best Attn Layer Set": best_attn_skip_layer_id_set,
                "Best MLP Layer Set": best_mlp_skip_layer_id_set,
            }
            fout.write(json.dumps(ans_json) + "\n")
            print("#Mean accepted tokens:", np.mean(accept_lengths_tree))
            print("Token acceptance rate:", (sum(accept_lengths_tree) - len(accept_lengths_tree)) / total_draft_num)