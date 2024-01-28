import discord
import asyncio
import torch
import transformers
import re

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
    BitsAndBytesConfig,  
)

transformers.logging.set_verbosity_info()
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

base_model_dir = "" # Path to your base model folder, ie "./models/TheBloke_Wizard-Vicuna-7B-Uncensored-HF"
lora_dir = "" # Path to your LoRA folder, ie "./loras/discord-chat-tuned-LoRA"

global prompt_memory
global temperature 
global repetition_penalty
global top_p
global top_k
global max_new_tokens
global system_prompt
temperature = 0.2 # Parameters for generation, read more here: https://huggingface.co/docs/transformers/main_classes/text_generation
repetition_penalty = 1.3
top_p = 0.85
top_k = 20
max_new_tokens = 70 # Parameters for generation, read more here: https://huggingface.co/docs/transformers/main_classes/text_generation

prompt_memory = 5 # How many messages the AI will store in it's internal chatlog. This will become very resource intensive with high numbers!!

system_prompt = "" # !!HUGE IMPACT ON RESULTS!! Set the stage for your AI to respond chat-like, "This is a conversation between User and DiscordGPT. DiscordGPT is very dumb and often answers questions wrong, sometimes intentionally..."
user_name = "" # The name you want the AI to acknowledge the most recent message sender as. Change the reference of this in the generate function to message.author to have it be the discord user's name!
agent_name = "" # What do you want your AI to think it's name is? 

host_discord_username = "" # Your discord unqiue username, not server nickname. This will give you access to discord chat commands to edit parameters of the text generation in real-time. 
bot_token = "" # Important! The bot's token. Do not share with others!

params = {
    'low_cpu_mem_usage': True,
    'torch_dtype': torch.bfloat16,
    'use_safetensors': True,
    "load_in_4bit": True,
    "do_sample": True
}

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(base_model_dir, padding_side="left")
print("\nTokeizer Loaded...\n")
model = AutoModelForCausalLM.from_pretrained(base_model_dir, **params, device_map="auto", quantization_config=quantization_config)
print("Model loaded...\n")
model.load_adapter(lora_dir) #Comment this out if you don't have a LoRA!
print("LoRA Applied...\n")
streamer = TextStreamer(tokenizer)

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

context = []

@client.event
async def on_ready():
    print(f'{client.user} is now online :)')

async def generate(message):
    prompt = f"{system_prompt}\n{''.join(context) if context else ''}\n{user_name}: {message.content[11:]}\n{agent_name}: "
    model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda:0")
    msg = await message.channel.send('Hm, let me think...')

    generated_ids = model.generate(**model_inputs, streamer = streamer, max_new_tokens = max_new_tokens, temperature = temperature, repetition_penalty = repetition_penalty, top_p = top_p, top_k = top_k).to("cuda:0")
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    regex = re.search("(.*)(\.|\?|\!)", output[len(prompt):])
    if regex != None:
        await msg.edit(content=regex.group())
        await add_to_context(message.content[11:], regex.group())
    else:
        await msg.edit(content=output[len(prompt):])
        await add_to_context(message.content[11:], output[len(prompt):])
    

async def add_to_context(message, response):
    if len(context) > prompt_memory:
        context.pop(0)
    context.append(f"\n{user_name}: {message}\n{agent_name}: {response}")


@client.event
async def on_message(message):
    global system_prompt
    if message.content.lower().startswith(f'Hey {agent_name} '.lower()):
        await generate(message)

    if message.content.lower().startswith(f'{agent_name}.temp='.lower()):
        if message.author == f"{host_discord_username}":
            global temperature
            temperature = float(message.content[(len(agent_name)+6):])
    
    if message.content.lower().startswith(f'{agent_name}.forget'.lower()):
        global context
        context.clear()

    if message.content.lower().startswith(f'{agent_name}.topp='.lower()):
        if message.author == f"{host_discord_username}":
            global top_p
            top_p = float(message.content[(len(agent_name)+6):])

    if message.content.lower().startswith(f'{agent_name}.topk='.lower()):
        if message.author == f"{host_discord_username}":
            global top_k
            top_k = float(message.content[(len(agent_name)+6):])

    if message.content.lower().startswith(f'{agent_name}.reppen='.lower()):
        if str(message.author) == f"{host_discord_username}":
            global repetition_penalty
            repetition_penalty = float(message.content[(len(agent_name)+8):])

    if message.content.lower().startswith(f'{agent_name}.maxtok='.lower()):
        if str(message.author) == f"{host_discord_username}":
            global max_new_tokens
            max_new_tokens = int(message.content[(len(agent_name)+8):])

    if message.content.lower().startswith(f'{agent_name}.pprompt'.lower()):
        if str(message.author) == f"{host_discord_username}":
            await message.channel.send(system_prompt)

    if message.content.lower().startswith(f'{agent_name}.eprompt='.lower()):
        if str(message.author) == f"{host_discord_username}":
            system_prompt = message.content[(len(agent_name)+9):]


client.run(bot_token)
