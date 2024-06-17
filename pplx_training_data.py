import pandas as pd
import numpy as np 
from openai import OpenAI
from dotenv import load_dotenv
import os
import time

load_dotenv()
api_key = os.getenv('PPLX_API_KEY')
client = OpenAI(
    api_key = api_key, 
    base_url = "https://api.perplexity.ai"
)

# Load prompts here
model = "llama-3-sonar-small-32k-chat"
prompts = pd.read_csv('./prompts.csv', header=0)
prompts = prompts['Prompts'].to_list()

message = []
model_list = [] 

for prompt in prompts:
    
    completion = client.chat.completions.create(
        model = model, 
        messages = [
            {"role": "system", "content": "You are a helpful assistant."}, 
            {"role":"user", "content":prompt}
        ], 
        max_tokens = 150
    )
    message.append(completion.choices[0].message.content)
    model_list.append(model)
    # To avoid getting rate limited
    if prompts.index(prompt) % 19 == 0:
        time.sleep(60)
    


d = {'model': model_list, 'prompt': prompts, 'message': message}
df = pd.DataFrame(data = d)

df.to_csv('./training_data_pplx.csv')