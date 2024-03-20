
import pandas as pd

input_dir = "../input"
df = pd.read_csv(f"{input_dir}/nbroad/gemma-rewrite-nbroad/nbroad-v1.csv")
    
rewrite_prompts = df["rewrite_prompt"].unique()
print(f"Number of unique rewrite prompts: {len(rewrite_prompts)}")
for i, prompt in enumerate(rewrite_prompts):
    print(f"Prompt {i}: {prompt}")