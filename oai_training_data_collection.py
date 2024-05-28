import pandas as pd
import numpy as np 
from openai import OpenAI

client = OpenAI()
project_id = 'proj_5JF0o5O7ktRRByRy75gKMxf9'

# Load prompts here
models = ["gpt-4o", "gpt-3.5-turbo"]
prompts = np.read_csv()

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


