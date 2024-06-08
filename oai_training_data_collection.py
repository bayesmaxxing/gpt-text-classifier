import pandas as pd
import numpy as np 
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(
    api_key = api_key
)

# Load prompts here
models = ["gpt-4o", "gpt-3.5-turbo"]
prompts = pd.read_csv('./prompts').to_list()

message = []
model_list = [] 

for model in models:
    for prompt in prompts:
    
        # Call to OAI API
        completion = client.chat.completions.create(
            model = model, 
            messages = [
                {"role": "system", "content": "You are a helpful assistant."}, 
                {"role":"user", "content":prompt}
            ]
        )
        message.append(completion.choices[0].message.content)
        model_list.append(model)

prompts.append(prompts)
d = {'model': model_list, 'prompt': prompts, 'message': message}
df = pd.DataFrame(data = d)

df.to_csv('training_data.csv')