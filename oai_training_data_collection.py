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
prompts = ['Can you explain what a lambda function is?']

message = []
model_list = [] 

for model in models:
    for prompt in prompts:
    
        # Call to OAI API
        completion = client.chat.completions.create(
            model = "gpt-4o", 
            messages = [
                {"role": "system", "content": "You are a helpful assistant."}, 
                {"role":"user", "content":prompt}
            ]
        )
        message.append(completion.choices[0].message)
        model_list.append(model)

d = {'model': model_list, 'prompt': prompts, 'message': message}
df = pd.DataFrame(data = d)