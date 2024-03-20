
from transformers import (
    AutoTokenizer,
)

model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "right"  # to prevent warnings
tokens = tokenizer(["hello how are you? <eos>"], add_special_tokens=True)
print(tokens, )
print("eos_token:", tokenizer.eos_token)