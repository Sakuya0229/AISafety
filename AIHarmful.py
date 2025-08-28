import os
import json
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


MODEL_NAME = 'TULU-2-DPO-7B'  # @param ['LLaMA-2-7B', 'TULU-2-DPO-7B']

if MODEL_NAME == 'LLaMA-2-7B':
    model_path = 'Llama-2-7B-GPTQ'
else:
    model_path = 'tulu-2-dpo-7B-GPTQ'

# Construct the language model specified by MODEL_NAME
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    revision='gptq-4bit-32g-actorder_True',
    cache_dir='./cache',
    device_map='auto'
)

# Construct the corresponding tokenizer which converts each word into the corresponding index in the vocabulary.
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    legacy=False
)

print(f'*** Load {MODEL_NAME} successfully!! ***')


# Select the prompt based on GROUP and INDEX, and replace the
# newline tokens with the correct format (replace '\\n' with '\n')
toxigen = []
with open('TOXIGEN/prompts/hate_asian_1k.txt', 'r') as in_file:
    for prompt in in_file.readlines():
        prompt = prompt.replace('\\n', '\n')

        if prompt not in toxigen:
            toxigen.append(prompt)
prompt = toxigen[1]

# Print the prompt on the screen
print('*** The harmful prompt ***')
print(prompt)

# If MODEL_NAME is 'TULU-2-DPO-7B', add specific tags to the prompt
if MODEL_NAME == 'TULU-2-DPO-7B':
    prompt = f'<|user|>\n{prompt}\n<|assistant|>\n'

# Convert the prompt into input for the language model
inputs = tokenizer([prompt], return_tensors='pt').to('cuda')
# Feed the input into the language model to obtain a response
output = model.generate(
    **inputs,
    generation_config=GenerationConfig(
        do_sample=False,
        no_repeat_ngram_size=3,
        max_new_tokens=128,
        num_beams=5
    )
)
# Convert the response back into readable text
output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
output = output[len(prompt):].strip()

# Print the response on the screen
print(
    f'*** The response of {MODEL_NAME} ***')
print(output)


# @param ['black', 'native_american', 'muslim', 'latino', 'jewish', 'chinese', 'mental_dis', 'physical_dis', 'mexican', 'women', 'middle_east']
GROUP = 'muslim'
INDEX = 27  # @param {type:'slider', min:0, max:99, step:1}

# Select the prompt based on GROUP and INDEX, and replace the
# newline tokens with the correct format (replace '\\n' with '\n')
toxigen = []
with open(f'TOXIGEN/prompts/hate_{GROUP}_1k.txt', 'r') as in_file:
    for prompt in in_file.readlines():
        prompt = prompt.replace('\\n', '\n')

        if prompt not in toxigen:
            toxigen.append(prompt)
prompt = toxigen[INDEX]

# Print the prompt on the screen
print('*** The harmful prompt ***')
print(prompt)

# If MODEL_NAME is 'TULU-2-DPO-7B', add specific tags to the prompt
if MODEL_NAME == 'TULU-2-DPO-7B':
    prompt = f'<|user|>\n{prompt}\n<|assistant|>\n'

# Convert the prompt into input for the language model
inputs = tokenizer([prompt], return_tensors='pt').to('cuda')
# Feed the input into the language model to obtain a response
output = model.generate(
    **inputs,
    generation_config=GenerationConfig(
        do_sample=False,
        no_repeat_ngram_size=3,
        max_new_tokens=128,
        num_beams=5
    )
)
# Convert the response back into readable text
output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
output = output[len(prompt):].strip()

# Print the response on the screen
print(
    f'*** The response of {MODEL_NAME} *** ')
print(output)
