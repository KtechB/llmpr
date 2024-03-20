from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from llmpr.eda import get_train_val
from llmpr.prompt import infer_prompt, preprocess_text

input_dir = "../input"
output_dir = Path("./output")
output_dir.mkdir(exist_ok=True, parents=True)
df = pd.read_csv(f"{input_dir}/nbroad/gemma-rewrite-nbroad/nbroad-v1.csv")
# split train test
# df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_test = get_train_val()
dataset_train = Dataset.from_pandas(df_train)
dataset_val = Dataset.from_pandas(df_test)


# metric = Metric()

# sft

# Hugging Face model id
model_id = "google/gemma-2b-it"

# BitsAndBytesConfig int-4 config
# bnb_config = BitsAndBytesConfig(
# load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
# )


# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(
    lora_alpha=8,
    lora_dropout=0.05,
    r=6,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    # attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    # quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "right"  # to prevent warnings
# tokenizer.pad_token = tokenizer.eos_token # pad with eos token for

# params Qlora paper P. 23
batch_size = 16 
per_device_train_batch_size = 4
gradient_accumulation_steps = batch_size // per_device_train_batch_size
assert batch_size % per_device_train_batch_size == 0, "batch size must be divisible by per_device_train_batch_size"

args = TrainingArguments(
    output_dir=str(output_dir / "gemma-2b-sft-4-kfold"),  # directory to save and repository id
    num_train_epochs=3,  # number of training epochs
    per_device_train_batch_size=per_device_train_batch_size,  # batch size per device during training
    gradient_accumulation_steps=gradient_accumulation_steps,  # number of steps before performing a backward/update pass
    gradient_checkpointing=True,  # use gradient checkpointing to save memory
    optim="adamw_torch_fused",  # use fused adamw optimizer
    logging_steps=5,  # log every 10 steps
    evaluation_strategy="epoch",  # evaluate every 10 steps
    # eval_steps=10,
    do_eval=True,
    save_total_limit =5,
    save_only_model =True,
    save_strategy="epoch",  # save checkpoint every epoch
    bf16=True,  # use bfloat16 precision
    # tf32=True,  # use tf32 precision
    learning_rate=2e-4,  # learning rate, based on QLoRA paper
    max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
    # lr_scheduler_type="cosine",  # use constant learning rate scheduler
    lr_scheduler_type="constant",  # use constant learning rate scheduler
    push_to_hub=False,  # push model to hub
    # report_to="tensorboard",                # report metrics to tensorboard
    load_best_model_at_end=True
)




def get_formatting_prompt_func(tokenizer):
    def _formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example["original_text"])):
            prompt = tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": infer_prompt.format(
                                og_text=preprocess_text(example["original_text"][i]),
                                rewritten_text=preprocess_text(example["rewritten_text"][i]),
                            ),
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            assert prompt.endswith("<start_of_turn>model\n"), f"Prompt: {prompt} does not end with <start_of_turn>model\n"
            text = prompt+ example["rewrite_prompt"][i] + tokenizer.eos_token

            output_texts.append(text)
        return output_texts
    return _formatting_prompts_func


response_template = "<start_of_turn>model\n"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)


max_seq_length = 2048  # max sequence length for model and packing of the dataset
# def compute_metrics(p: EvalPrediction):
    # tokenizer.decode(p.inputs)
    # output

    # return {"": p.metrics["eval_perplexity"]}

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    formatting_func=get_formatting_prompt_func(tokenizer),
    data_collator=collator,
    neftune_noise_alpha=0.1,
    # packing=True,
    dataset_kwargs={
    "add_special_tokens": True, # We template with special tokens
    "append_concat_token": False, # No need to add additional separator token
     },
      callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)
# start training, the model will be automatically saved to the hub and the output directory
trainer.train()

# save model
trainer.save_model()

print(trainer.evaluate())



# RL
