# DiscordSPT (Single Purpose Transformer)
Simple implimentation of local language model within a Discord bot. Includes chat commands for on-the-fly generation parameter training

# Warning 
This was made with support for modern Nvidia GPUs, if you run arcane hardware, this will not work as-is!

# Setup
After cloning, 
```pip install -r requirements.txt```

Make sure Torch is the right version. If ```python main.py``` throws torch errors, try ```pip uninstall torch``` and ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121```

Open main.py in your least-favorite text editor. 

On lines 18 and 19 provide paths from main.py to the folder containing your base model, and your LoRA (comment out if none) respectively. 

You will notice a number of comments along lines 36-38, fill out the strings accordingly. These will have a great deal of influence over the output, so try to be thoughtful/creative.
The final format that will be passed to the AI after interpolation will be something like: 
>This is a chat between User and DiscordGPT. DiscordGPT is a sentient AI mastermind who will always answer a Users question with extreme >confidence. DiscordGPT is never wrong about anything, and knows everything. DiscordGPT often makes jokes about the Users messages. 
>
>[context]
>
>User: {user's discord message will go here}
>
>DiscordGPT: {generated text will go here}

If you change the ```prompt_memory``` variable on line 34, you can increase the number of "User: ... \n Agent: ..." pairs that will appear in [context] as users converse with the machine. 

Save these changes to main.py.

# Usage

Run the bot with ```python main.py```

From Discord, simply send a message:
>Hey {agent_name} [silly message]

The AI will swiftly respond with a placeholder message, and once generation is complete the placeholder message will be replaced.

If the AI gets stuck in a crappy loop and output quality degrades
>{agent_name}.forget

will clear the chat context, or the "memory" allowing for a fresh start on the next prompt

If you read further down in main.py there are various chat commands for editing the parameters during runtime. Very good for tuning your output! They are fairly self-explanitory as to their purpose, and documentation on the parameters is linked further up in the file. 
These commands are only available for the host of the bot, or rather whichever discord username is declared in ```host_discord_username = ""```

# Enjoy!



