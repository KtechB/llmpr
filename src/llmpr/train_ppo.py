import os
from typing import Tuple

import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from llmpr.eda import get_train_val
from llmpr.infer import batch_to_prompts
from llmpr.metric import Metric
from llmpr.prompt import infer_prompt, preprocess_text

load_dotenv()

model_name = "gemma-2b-it-ppo"

model_id = "./output/merged/gemma-2b-it-sft-4-kfold-epoch1"
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_id,
    peft_config=lora_config,
    load_in_4bit=True,
)
ppo_config = PPOConfig(
    model_name=model_name,
    task_name=model_name,
    log_with="wandb",
    tracker_project_name = os.environ.get("WANDB_PROJECT", "trl"),
    tracker_kwargs={"wandb": { "name": f"{model_name}"}},
    # batch_size=2,
    mini_batch_size = 1,
    gradient_accumulation_steps= 16,
    remove_unused_columns=False,
    learning_rate=1.41e-5,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
df_train, df_test = get_train_val()
dataset_train = Dataset.from_pandas(df_train)
dataset_val = Dataset.from_pandas(df_test)
# tokenizer.padding_side = "right"  # to prevent warnings
tokenizer.padding_side = "left"  # to prevent warnings
ref_model = None
def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


ppo_trainer = PPOTrainer(
    ppo_config,
    model,
    ref_model,
    tokenizer,
    dataset=dataset_train,
    data_collator=collator,
)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 32,
}
metric= Metric()
def batch_to_prompts(row)->Tuple[list[str], list[str]]:
    original_texts = [preprocess_text(v) for v in row["original_text"]]
    rewritten_texts = [preprocess_text(v) for v in row["rewritten_text"]]
    rewrite_prompts = [v for v in row["rewrite_prompt"]]
    chats = [[{"role": "user", "content": infer_prompt.format(og_text=og, rewritten_text=rewritten)}] for og, rewritten in zip(original_texts, rewritten_texts)]
    prompts = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) for chat in chats]
    return prompts ,rewrite_prompts
for _epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    prompts, rewrite_prompts = batch_to_prompts(batch)
    batch["query"] = prompts
    
    query_tensors = tokenizer(prompts, add_special_tokens=False, return_tensors="pt", padding=True, truncation=True, max_length=2000)["input_ids"]
    query_tensors = [q for q in query_tensors]
    


    # Get response from gpt2
    response_tensors, ref_response_tensors = ppo_trainer.generate(
        query_tensors,
        return_prompt=False,
        generate_ref_response=True,
        **generation_kwargs,
    )
    batch["response"] = tokenizer.batch_decode(response_tensors)
    batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

    # Compute sentiment score
    # pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    scores = [metric.calc_score(response, rewrite_prompt) for response, rewrite_prompt in zip(batch["response"], rewrite_prompts)]
    rewards = [torch.tensor(score) for score in scores]
    batch["rewards"] = rewards
    ref_scores = [metric.calc_score(response, rewrite_prompt) for response, rewrite_prompt in zip(batch["ref_response"], rewrite_prompts)]
    ref_rewards = [torch.tensor(score) for score in scores]
    batch["ref_rewards"] = ref_rewards

    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(
        stats,
        batch,
        rewards,
        columns_to_log=["query", "response", "ref_response", "ref_rewards"],
    )
