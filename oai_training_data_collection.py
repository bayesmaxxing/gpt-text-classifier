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
prompts = pd.read_csv('./test_prompts.csv', header=0)
prompts = prompts['prompts'].to_list()

message = []
model_list = [] 
max_token_choices = [100,150,200,250,300]
rng = np.random.default_rng()

for model in models:
    for prompt in prompts:
    
        rand_index = rng.integers(0,5)
        # Call to OAI API
        completion = client.chat.completions.create(
            model = model, 
            messages = [
                {"role": "system", "content": "You are a helpful assistant."}, 
                {"role":"user", "content":prompt}
            ], 
            max_tokens = max_token_choices[rand_index]
        )
        message.append(completion.choices[0].message.content)
        model_list.append(model)

prompts = prompts + prompts
d = {'model': model_list, 'prompt': prompts, 'message': message}
df = pd.DataFrame(data = d)

df.to_csv('./test_data_gpts_variable.csv')