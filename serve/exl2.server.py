import argparse
import sys, os
import uuid
import time
import bottle
from bottle import Bottle, run, route, request
bottle.BaseRequest.MEMFILE_MAX = 1024 * 1024 * 10

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    ExLlamaV2Lora,
)
from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler
)

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', required=True, type=str, help="Path to the model dir")
parser.add_argument('-l', '--lora', required=False, type=str, default=None, help="Path to the lora dir")
parser.add_argument('-a', '--model_name', required=True, type=str, help="Model alias or ID")
parser.add_argument('-c', '--length', default=0, required=False, type=int, help="Context length")
parser.add_argument('-s', '--scale', default=1.0, required=False, type=float, help="Linear scale")
parser.add_argument('-A', '--ntk_scale', default=1.0, required=False, type=float, help="NTK scale")
parser.add_argument('--port', default=8013, required=False, type=int, help="Port to listen on")
parser.add_argument('--ip', default='127.0.0.1', required=False, type=str, help="IP to listen on")
args = parser.parse_args()

# Initialize model and cache
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

lora = None
if args.lora is not None:
    lora = ExLlamaV2Lora.from_directory(model, args.lora)
# Initialize generator
generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
# Make sure CUDA is initialized so we can measure performance
generator.warmup()

conversations = {}
app = Bottle()

def full_conversation(idx):
    chat = ''
    for message in conversations[idx]['messages']:
        if message['role'] == 'system':
            chat += message['content']
        if message['role'] == 'user':
            chat += conversations[idx]['prefix'] + message['content'] + conversations[idx]['infix']
        if message['role'] == 'assistant':
            chat += conversations[idx]['suffix'] + message['content'] + '\n'
 
    if conversations[idx]['messages'][-1]['role'] == 'user':
        chat += conversations[idx]['suffix']
 
    return chat

@app.route('/prompt', method='PUT')
def set_prompt():
    data = request.json
    conversation_uuid = data.get('uuid', str(uuid.uuid4()))
    messages = data.get('messages', [{'role':'system', 'content':''}])
    prefix = data.get('prefix', 'USER: ')
    suffix = data.get('suffix', 'ASSISTANT:')
    infix = data.get('infix', '\n')
    conversations[conversation_uuid] = {
        "messages": messages,
        "prefix": prefix,
        "suffix": suffix,
        "infix": infix
    }
    return {"message": "Prompt set", "uuid": conversation_uuid}

@app.route('/chat', method='POST')
def chat():
    data = request.json
    conversation_uuid = data['uuid']
    if conversation_uuid not in conversations:
        return {"uuid":conversation_uuid, "message": "not found"}
 
    temperature = data.get('temperature', 0.5)
    top_k = data.get('top_k', 40)
    top_p = data.get('top_p', 0.9)
    typical = data.get('typical', 0)
    repetition_penalty = data.get('repetition_penalty', 1.15)
    max_new_tokens = data.get('max_length', 256)
    query = data.get('query')
 
    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = temperature
    settings.top_k = top_k
    settings.top_p = top_p
    settings.typical = typical
    settings.token_repetition_penalty = repetition_penalty
    conversations[conversation_uuid]['messages'].append({'role':'user','content':query})
 
    full_ctx = full_conversation(conversation_uuid)
    input_ids = tokenizer.encode(full_ctx, add_bos = True)
    prompt_tokens = input_ids.shape[-1]
 
    start_time = time.time_ns()

    generator.set_stop_conditions([tokenizer.eos_token_id, conversations[conversation_uuid]['prefix'].rstrip(), "<|im_end|>"])
    generator.begin_stream(input_ids, settings, loras = lora)
    generated_tokens = 0
    new_text = ""
    while True:
        chunk, eos, _ = generator.stream()
        generated_tokens += 1
        new_text += chunk
        if eos or generated_tokens == max_new_tokens:
            break

    end_time = time.time_ns()
    secs = (end_time - start_time) / 1e9

    conversations[conversation_uuid]['messages'].append({'role':'assistant','content':new_text})
    return {
        "uuid": conversation_uuid,
        "text": new_text,
        "tokens": generated_tokens,
        "rate": generated_tokens / secs,
        "model": args.model_name,
        "type" : 'exllama',
        "ctx" : prompt_tokens + generated_tokens
    }

@app.route('/complete', method='POST')
def complete():
    data = request.json
    temperature = data.get('temperature', 0.5)
    max_new_tokens = data.get('max_length', 256)
    add_bos = data.get('add_bos', False)
    top_k = data.get('top_k', 40)
    top_p = data.get('top_p', 0.9)
    typical = data.get('typical', 0)
    repetition_penalty = data.get('repetition_penalty', 1.15)
    encode_special = data.get('encode_special', True)
    query = data.get('query')

    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = temperature
    settings.top_k = top_k
    settings.top_p = top_p
    settings.typical = typical
    settings.token_repetition_penalty = repetition_penalty

    input_ids = tokenizer.encode(query, add_bos = add_bos, encode_special_tokens = encode_special)
    prompt_tokens = input_ids.shape[-1]

    start_time = time.time_ns()
    generator.set_stop_conditions([tokenizer.eos_token_id])
    generator.begin_stream(input_ids, settings, loras=lora)
    generated_tokens = 0
    new_text = ""
    while True:
        chunk, eos, _ = generator.stream()
        generated_tokens += 1
        new_text += chunk
        if eos or generated_tokens == max_new_tokens:
            break

    end_time = time.time_ns()
    secs = (end_time - start_time) / 1e9
    return {
        "text": new_text,
        "ctx": generated_tokens + prompt_tokens,
        "tokens": generated_tokens,
        "rate": generated_tokens / secs,
        "type": 'exllama',
        "model": args.model_name
    }

run(app, host=args.ip, port=args.port)
