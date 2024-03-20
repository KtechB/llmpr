
import pandas as pd
from vllm import LLM, SamplingParams

USER_CHAT_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

TOTAL_PROMPT_TEMPLATE = '{rewrite_prompt} :"{original_text}"'

df = pd.read_csv("data/llmpr_prompt_text_5.csv" )
print(df.head())
df["total prompt"] = df.apply(lambda x:  TOTAL_PROMPT_TEMPLATE.format(rewrite_prompt=x["rewrite_prompt"], original_text=x["original_text"]), axis=1)

llm = LLM(model="google/gemma-7b-it", max_model_len=4096)
sampling_params = SamplingParams(temperature=0.01, top_p=0.95, max_tokens=1000 )

batch_size= 8
# batch infer

results = []
"""
for i in tqdm(range(0, len(df), batch_size)):
    prompts = df["total prompt"][i:i+batch_size].tolist()
    prompts = [USER_CHAT_TEMPLATE.format(prompt=prompt) for prompt in prompts]
    outputs = llm.generate(prompts, sampling_params)
    results += [output.outputs[0].text for output in outputs]
"""
prompts = df["total prompt"].tolist()
prompts = [USER_CHAT_TEMPLATE.format(prompt=prompt) for prompt in prompts]
outputs = llm.generate(prompts, sampling_params)
results += [output.outputs[0].text for output in outputs]
df["rewritten_text"] = results
df.to_csv("data/llmpr_dataset_5.csv", index=False)

