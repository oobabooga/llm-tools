import argparse
import sys, os
import uuid
import time
from tqdm import tqdm

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)
from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler
)

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', required=True, type=str, help="Path to the model dir")
parser.add_argument('-c', '--length', default=0, required=False, type=int, help="Context length")
parser.add_argument('-n', '--samples', default=1, required=False, type=int, help="Number of samples per task")
parser.add_argument('-s', '--scale', default=1.0, required=False, type=float, help="Linear scale")
parser.add_argument('-A', '--ntk_scale', default=1.0, required=False, type=float, help="NTK scale")
parser.add_argument('-p', '--plus', action='store_true', help="Use HumanEvalPlus instead of original HumanEval")
parser.add_argument('-o', '--output', default='samples.jsonl', required=False, type=str, help="Output file name")
parser.add_argument('-t', '--temperature', default=0.1, required=False, type=float, help="Temperature")
parser.add_argument('--max_new_tokens', default = 384, required=False, type=int, help="Max tokens to generate")
parser.add_argument('--top_k', default=40, required=False, type=int, help="Top K")
parser.add_argument('--top_p', default=0.75, required=False, type=float, help="Top P")
parser.add_argument('--repetition_penalty', default=1, required=False, type=float, help="Repetition penalty")
args = parser.parse_args()

config = ExLlamaV2Config()
config.model_dir = args.model
config.prepare()

if args.length != 0:
    config.max_seq_len = args.length

if args.scale != 1.0:
    config.scale_pos_emb = args.scale

if args.ntk_scale != 1.0:
    config.scale_alpha_value = args.ntk_scale

model = ExLlamaV2(config)
print("Loading model: " + args.model)
model.load()
tokenizer = ExLlamaV2Tokenizer(config)
cache = ExLlamaV2Cache(model)

generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
generator.warmup()

def generate_one_completion(prompt: str):

    settings = ExLlamaV2Sampler.Settings()
    settings.top_k = args.top_k
    settings.top_p = args.top_p
    settings.temperature = args.temperature
    settings.token_repetition_penalty = args.repetition_penalty
 
    input_ids = tokenizer.encode(prompt, add_bos = True)
 
    generator.set_stop_conditions([tokenizer.eos_token_id])
    generator.begin_stream(input_ids, settings)
    generated_tokens = 0
    new_text = ""
    while True:
        chunk, eos, _ = generator.stream()
        generated_tokens += 1
        new_text += chunk
        if eos or generated_tokens == args.max_new_tokens:
            break
    return new_text


if not args.plus:
    from human_eval.data import write_jsonl, read_problems
    problems = read_problems()
else:
    from evalplus.data import get_human_eval_plus, write_jsonl
    problems = get_human_eval_plus()

num_samples_per_task = args.samples
samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    for task_id in tqdm(problems)
    for _ in range(num_samples_per_task)
]

write_jsonl(args.output, samples)
