
from typing import Tuple

import pandas as pd
import torch
from peft.auto import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import AutoTokenizer

from llmpr.eda import get_train_val
from llmpr.metric import Metric
from llmpr.prompt import infer_prompt, preprocess_text


def batch_to_prompts(row)->Tuple[list[str], list[str]]:
    original_texts = [preprocess_text(v) for v in row["original_text"]]
    rewritten_texts = [preprocess_text(v) for v in row["rewritten_text"]]
    rewrite_prompts = [v for v in row["rewrite_prompt"]]
    chats = [[{"role": "user", "content": infer_prompt.format(og_text=og, rewritten_text=rewritten)}] for og, rewritten in zip(original_texts, rewritten_texts)]
    prompts = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) for chat in chats]
    return prompts ,rewrite_prompts

    
if __name__ == "__main__":
    # model_path = "./output/gemma-2b-sft-3-add-eos"
    model_path = "./output/gemma-2b-sft-4-kfold"
    model_path = "./output/gemma-2b-sft-4-kfold"
    dtype = torch.bfloat16
    metric = Metric()
    input_dir = "../input"
    df = pd.read_csv(f"{input_dir}/nbroad/gemma-rewrite-nbroad/nbroad-v1.csv")
    # df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_test = get_train_val()
    df  = df_test.iloc[:50]
    # df  = df_train.iloc[:50]
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoPeftModelForCausalLM.from_pretrained(model_path,
        device_map="cuda",
        torch_dtype=dtype,
    )
    scores = []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        original_text = preprocess_text(row["original_text"])
        rewritten_text = preprocess_text(row["rewritten_text"])
        rewrite_prompt = row["rewrite_prompt"]

        chat = [
            {
                "role": "user",
                "content": infer_prompt.format(
                    og_text=original_text,
                    rewritten_text=rewritten_text
                ),
            }
        ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        # print("prompt:\n", prompt, "\n")
        # print("---------")
        inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        inputs_length = len(inputs[0])

        outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=300)
        print("output:")
        output_text = tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)
        print(output_text)
        print("-----------------")

        print('actual:')
        print(rewrite_prompt)


        score = metric.calc_score(output_text, rewrite_prompt)
        print("score:", score)
        print("=============")
        scores.append(score)

    print("mean score:", sum(scores) / len(scores))

    