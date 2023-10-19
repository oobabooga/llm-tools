import argparse
import torch
from tqdm import tqdm
from human_eval.data import write_jsonl, read_problems
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', required=True, type=str, help="Grasping Model")
parser.add_argument('-f', '--eight', action='store_true', help="Load in INT8 instead of NF4")
parser.add_argument('-o', '--output', default='samples.jsonl', required=False, type=str, help="Output file name")
parser.add_argument('-r', '--remote', required=False, type=bool, default=True, help="Trust remote code (default is True)")
args = parser.parse_args()

model_id = args.model
tokenizer = AutoTokenizer.from_pretrained(model_id)

if not args.eight:
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', quantization_config=nf4_config, trust_remote_code=args.remote)
else:
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', load_in_8bit=True, trust_remote_code=args.remote, use_flash_attention_2=True)

print(model.generation_config)

def generate_one_completion(prompt: str):
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)

    # Generate
    generate_ids = model.generate(inputs.input_ids.to("cuda"), max_new_tokens=384, do_sample=True, top_p=0.75, top_k=40, temperature=0.1)
    completion = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    completion = completion.replace(prompt, "").split("\n\n\n")[0]

    return completion

problems = read_problems()

num_samples_per_task = 1
samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    for task_id in tqdm(problems)
    for _ in range(num_samples_per_task)
]
write_jsonl(args.output, samples)
