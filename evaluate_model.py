# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import fire
import numpy as np

import vllm
from jinja2 import Template
from datasets import load_from_disk

from utils.math_grader import (
    answer_tag_reward_fn, 
    boxed_reward_fn, 
    answer_tag_reward_fn_for_orz,
)



def apply_qwen_math_template(question: str):
    return (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def apply_r1_template(question: str):
    return (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: "
        + question
        + "\nAssistant: <think>"
    )


def apply_new_r1_template(question: str):
    return (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> tags. And the final answer should be placed within \\boxed{}, i.e., <think> reasoning process here </think> \\boxed{answer} here."
        "\nUser: " + question
        + "\nAssistant: "
    )


def apply_qwen_r1_template(question: str):
    return (
        "<|im_start|>system\nA conversation between user and assistant. The user asks a question, and the assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. "
        "The assistant should put the final answer within \\boxed{}. The reasoning process is enclosed within <think> </think>. The answer is placed after </think>. I.e., <think> reasoning process here </think> \\boxed{answer} here.<|im_end|>\n"
        "<|im_start|>user\n" + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


# The following two templates are used to evaluate baselines from other projects.
def apply_prime_zero_template(question: str):
    """https://huggingface.co/PRIME-RL/Eurus-2-7B-PRIME-Zero"""
    question = question + "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"
    return f"A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {question}. Assistant:"


def apply_open_reasoner_zero_template(question: str):
    "https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/e008f6d95f0b9a0e992f6b8bac912515b50a4634/playground/zero_setting_base.py"
    prompt_template_jinja = """\
{{bos_token}}A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. \
The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {{prompt}}
Assistant: <think>\
"""
    prompt_instruction_template_jinja = """\
You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.
This is the problem:
{{prompt}}
"""
    prompt_instruction_template = Template(prompt_instruction_template_jinja)
    prompt_instruction = prompt_instruction_template.render(prompt=question)
    prompt_template = Template(prompt_template_jinja)
    return prompt_template.render(bos_token="", prompt=prompt_instruction)


def main(
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    tasks: list = ["aime", "amc", "math", "minerva", "olympiad_bench"],
    template: str = "qwen_math",
    dataset_name: str = "./datas/evaluation_suite",
    temperature: float = 0,
    top_p: float = 1,
    min_p: float = 0,
    max_tokens: int = 3000,
    max_model_len: int = 4096,  # VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 for longer ones.
    n_samples: int = 1,
    save: bool = False,
    seed: int = 0,
    output_dir: str = "./outputs",
    max_num_seqs: int = 256,
    start: int = 0,
    end: int = -1,
):
    sampling_params = vllm.SamplingParams(
        n=n_samples,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        max_tokens=max_tokens,
        logprobs=2,
        seed=seed,
        skip_special_tokens=False
    )

    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    model = vllm.LLM(
        model_name,
        max_model_len=max_model_len,
        dtype="bfloat16",
        enable_prefix_caching=True,
        trust_remote_code=True,
        tensor_parallel_size=len(available_gpus),
        gpu_memory_utilization=0.95,
        seed=seed,
        max_num_seqs=max_num_seqs,
    )

    if "prime" in model_name.lower():
        template = "prime-zero"
    if "open-reasoner-zero" in model_name.lower():
        template = "open-reasoner-zero"

    if "instruct" in model_name.lower() and "instruct" not in template:
        input(
            f"{model_name}\n{template}\ninstruct model but not instruct template! continue?"
        )

    print("Using template:", template)
    if template in ["qwen_math", "no"]:
        math_reward_fn = boxed_reward_fn
        if template == "qwen_math":
            apply_template = apply_qwen_math_template
        else:
            apply_template = lambda x: x
    elif template == "r1":
        math_reward_fn = answer_tag_reward_fn
        sampling_params.stop = ["</answer>"]
        sampling_params.include_stop_str_in_output = True
        apply_template = apply_r1_template
    elif template == "new_r1":
        math_reward_fn = boxed_reward_fn
        apply_template = apply_new_r1_template
    elif template == "prime-zero":
        math_reward_fn = boxed_reward_fn
        apply_template = apply_prime_zero_template
    elif template == "open-reasoner-zero":
        math_reward_fn = answer_tag_reward_fn_for_orz
        apply_template = apply_open_reasoner_zero_template
    elif template == "llama-instruct":
        from transformers import AutoTokenizer

        math_reward_fn = boxed_reward_fn
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def apply_template(question):
            return tokenizer.apply_chat_template(
                [
                    {
                        "content": f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\n",
                        "role": "user",
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

    elif template == "r1d":  # r1-distill
        from transformers import AutoTokenizer

        math_reward_fn = boxed_reward_fn
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def apply_template(question):
            return tokenizer.apply_chat_template(
                [{"content": question, "role": "user"}],
                tokenize=False,
                add_generation_prompt=True,
            )
        
    elif template == "qwen_r1":
        math_reward_fn = boxed_reward_fn
        apply_template = apply_qwen_r1_template

    else:
        raise ValueError

    results = {}
    avg_lens = {}
    max_lens = {}
    formatted = {}
    for task_name, dataset in load_from_disk(dataset_name).items():
        to_be_saved = []

        if task_name not in tasks:
            continue

        if end != -1:
            prompts = dataset["problem"][start:end]
            targets = dataset["answer"][start:end]
        else:
            prompts = dataset["problem"][start:]
            targets = dataset["answer"][start:]

        prompts = list(map(apply_template, prompts))

        print("=" * 50)
        print(f"{task_name}: {len(prompts)}")
        print(prompts[0])

        outputs = model.generate(prompts, sampling_params)
        batch_scores = []
        batch_formatted = []
        batch_lengths = []
        for k in range(len(outputs)):
            output = outputs[k]
            gt_repeated = [targets[k]] * sampling_params.n
            rewards, infos = [], []
            for model_output, gt in zip([o.text for o in output.outputs], gt_repeated):
                info, r = math_reward_fn(model_output, gt, fast=False)
                rewards.append(r)
                infos.append(info)
            rewards = np.array(rewards)
            batch_lengths.append([len(o.token_ids) for o in output.outputs])
            batch_scores.append(rewards.mean())

            if infos[0] is not {}:
                batch_formatted.append(np.mean([i["formatted"] for i in infos]))

            to_be_saved.append(
                {
                    "task_name": task_name,
                    "prompt": output.prompt,
                    "gt": targets[k],
                    "model_output": [o.text for o in output.outputs],
                    #"model_output_token_ids": [o.token_ids for o in output.outputs],
                    "reward": [r for r in rewards],
                    "formatted": [i["formatted"] for i in infos],
                }
            )

        results[task_name] = np.mean(batch_scores)
        avg_lens[task_name] = np.mean(batch_lengths)
        if batch_formatted:
            formatted[task_name] = np.mean(batch_formatted)
        max_lens[task_name] = np.max(batch_lengths)

        print(f"acc:{results[task_name]}, avg length: {avg_lens[task_name]}, max length: {max_lens[task_name]}")
        if task_name in formatted:
            print(f"formatted: {formatted[task_name]}")

        if save:
            fn = os.path.join(output_dir, model_name.split("/")[-1], task_name)
            os.makedirs(fn, exist_ok=True)

            fn = f"{fn}/template_{template}_temp{temperature}topp{top_p}minp{min_p}_n{n_samples}_seed{seed}_start{start}end{end}.json"
            print(f"saving model outputs for {task_name} at {fn}")
            json.dump(to_be_saved, open(fn,"w",), indent=4)

    print("=" * 25, "Summary", "=" * 25)
    print(results)
    print("avg acc:", np.mean(list(results.values())))
    print("avg_lens:", avg_lens)
    print("max_lens:", max_lens)
    print("formatted:", formatted)

    
if __name__ == "__main__":
    fire.Fire(main)