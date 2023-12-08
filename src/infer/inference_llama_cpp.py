"""
This script support vllm batch inference with cot/pal/tora prompt.
Also sopport inference of fine-tuned models like WizardMath/ToRA.
Code based on: https://github.com/microsoft/ProphetNet/tree/master/CRITIC
"""
import random
import os
import argparse
import time
# from vllm import LLM, SamplingParams
import llama_cpp
from llama_cpp import Llama
from datetime import datetime
from tqdm import tqdm
import pickle

from eval.evaluate import evaluate
from utils.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from utils.parser import *
from utils.data_loader import load_data
from utils.python_executor import PythonExecutor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="gsm8k", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tora", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int) # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=1024, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_train_prompt_format", action="store_true")
    args = parser.parse_args()
    args.top_p = 1 if args.temperature == 0 else args.top_p # top_p must be 1 when using greedy sampling (vllm)
    return args


def prepare_data(args):
    examples = load_data(args.data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        examples = random.sample(examples, args.num_test_sample)
    elif args.num_test_sample == -1:
        args.num_test_sample = len(examples)
    
    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    if args.end == -1:
        args.end = len(examples)
    examples = examples[args.start:args.end]

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    out_file_prefix = f'{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}'
    out_file = f'{args.output_dir}/{model_name}/{args.data_name}/{out_file_prefix}_s{args.start}_e{args.end}_{dt_string}.jsonl'
    os.makedirs(f'{args.output_dir}/{model_name}/{args.data_name}', exist_ok=True)

    # load all processed samples
    processed_files = [f for f in os.listdir(f"{args.output_dir}/{model_name}/{args.data_name}/") if f.endswith(".jsonl") and f.startswith(out_file_prefix)]    
    processed_samples = []
    for f in processed_files:
        processed_samples.extend(list(load_jsonl(f"{args.output_dir}/{model_name}/{args.data_name}/{f}")))

    # dedepulicate
    processed_samples = {sample['idx']: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    total_examples = len(examples)
    examples = [example for example in examples if example['idx'] not in processed_idxs]
    print(f"Idx {args.start} - {args.end}: Remain {len(examples)}/{total_examples} samples.")
    if len(examples) == 0:
        pass
    else:
        print(examples[0])
    return examples, processed_samples, out_file


def main(args):
    examples, processed_samples, out_file = prepare_data(args)

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr='solution()')
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    # load model
    if len(examples) > 0:
        available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        # llm = LLM(model=args.model_name_or_path, tensor_parallel_size=len(available_gpus))
        print('n_gpu_layers', 300)
        llm = Llama(model_path=args.model_name_or_path, n_gpu_layers=300) # Offloading all layers to GPU
    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example['idx']

        # parse question and answer
        example['question'] = parse_question(example, args.data_name)
        gt_cot, gt_ans = parse_ground_truth(example, args.data_name)
        full_prompt = construct_prompt(args, example)

        sample = {'idx': idx, 'question': example['question'], 'gt_cot': gt_cot, 'gt': gt_ans, 'prompt': full_prompt}

        # add remain fields
        for key in ['level', 'type', 'unit', 'solution_type', 'choices', 'solution', 'ques_type', \
            'ans_type', 'answer_type', 'dataset', 'subfield', 'filed', 'theorem', 'answer']:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)  

    print("dataset:", args.data_name, "samples:", len(samples))
    if len(samples) > 0:
        print("-" * 50)
        print("sample:", samples[0]['prompt'])
        print("-" * 50)

    # repeat n times
    remain_prompts = [sample['prompt'] for sample in samples for _ in range(args.n_sampling)]
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    max_func_call = 1 if args.prompt_type in ['cot', 'pal'] else 4
    stop_tokens = ["</s>", "```output"]

    if args.prompt_type in ['cot']:
        stop_tokens.append("\n\n")
    elif args.prompt_type in ['wizard_zs', 'platypus_fs']:
        stop_tokens.extend(["Instruction", "Response"])

    # start inference
    # measure time use
    start_time = time.time()
    for epoch in range(max_func_call):
        print("=" * 50, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        # get all outputs
        prompts = [item[1] for item in current_prompts]
        # outputs = llm.generate(prompts, SamplingParams(
        #                 temperature=args.temperature,
        #                 top_p=args.top_p,
        #                 max_tokens=args.max_tokens_per_call,
        #                 n=1,
        #                 stop=stop_tokens,
        # ))
        outputs = [] 
        responses = []
        for prompt in tqdm(prompts):
            response = llm(prompt,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens_per_call,
                    # n=1, # THis is just number of outputs to return (For self-consistency?)
                    stop=stop_tokens,
            )
            responses.append(response)
            output = response['choices'][0]['text']
            outputs.append(output)

        # outputs = sorted(outputs, key=lambda x: int(x.request_id)) # sort outputs by request_id (Due to batched inference)
        # outputs = [output.outputs[0].text for output in outputs]
        # assert len(outputs) == len(current_prompts)

        # process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            if args.prompt_type == "pal":
                remain_prompts.append((i, query))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query))
            elif ("boxed" not in output and output.endswith("```")):
                program = extract_program(query)
                remain_prompts.append((i, query))
                remain_codes.append(program)
            else:
                end_prompts.append((i, query))

        # execute the remain prompts
        # remain_results = executor.batch_apply(remain_codes)

        remain_results = []
        code_times = []
        for code in remain_codes:
            code_start = time.time()
            result = executor.apply(code)
            code_end = time.time()

            code_exec_time = code_end - code_start
            # code_time_str = f"{int(code_exec_time // 60)}:{int(code_exec_time % 60):02d}"
            code_times.append(str(code_exec_time))
            code_times.append('\n')

            remain_results.append(result)
        
        # Write the code execution times
        with open(out_file.replace('.jsonl', f'_code_times_{epoch}.txt'), 'w') as f:
            f.writelines(code_times)

        with open(out_file.replace('.jsonl', f'_responses_{epoch}.pkl'), 'wb') as res_f:
            pickle.dump(responses, res_f)

        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])
    ans_split = "<|assistant|>" if args.use_train_prompt_format else "Question:"
    codes = [prompt.split(ans_split)[-1].strip() for _, prompt in end_prompts]

    # extract preds
    results = [run_execute(executor, code, args.prompt_type) for code in codes]
    time_use = time.time() - start_time

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i*args.n_sampling: (i+1)*args.n_sampling]
        result = results[i*args.n_sampling: (i+1)*args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]

        sample.pop('prompt')
        sample.update({'code': code, 'pred': preds, 'report': reports})
        all_samples.append(sample)

    # add processed samples
    all_samples.extend(processed_samples)
    save_jsonl(all_samples, out_file)

    result_str = evaluate(samples=all_samples, data_name=args.data_name, prompt_type=args.prompt_type, execute=True)
    result_str += f"\nTime use: {time_use:.2f}s"
    time_str = f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    result_str += f"\nTime use: {time_str}"

    with open(out_file.replace(".jsonl", f"_{args.prompt_type}.metrics"), "w") as f:
        f.write(result_str)

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)