import pandas as pd

# forum_messsages_df = pd.read_csv('/kaggle/input/meta-kaggle/ForumMessages.csv')
# forum_messsages_df.head()

# %%
# Let's grab the first 5 messages to test our generation pipeline:

original_texts = ["hello, this is compute world . how do you think?"] #forum_messsages_df['Message'][:5]

# %%
rewrite_prompts = [
    'Explain this to me like I\'m five.',
    'Convert this into a sea shanty.',
    'Make this rhyme.',
]

# %% [markdown]
# ## Generating `rewritten_text` with Gemma
# Now for the fun part! We can use gemma to rewrite our original text samples
# using the rewrite prompts we created.
# The code in this cell is borrowed from [the model card](https://www.kaggle.com/models/google/gemma/frameworks/pyTorch/variations/7b-it-quant).
# The important things to know:
# 
# We're using the 7B parameter instruction tuned quantized model, which means:
# 
# - 7B Parameter: this is the larger of the two Gemma models (the other has 2 billion parameters).
#     In general we expect the larger model to perform better on complex tasks, but
#     it's more resource intensive. You can see exactly how Gemma 7B compares to to Gemma 2B [here](https://ai.google.dev/gemma).
# - Instruction Tuned: instruction tuning is an extra training step that results in a model that
#     can follow user instructions better. Our rewrite prompt is a kind of instruction, so this is what we want!
# - Quantized: quantization is a way of shrinking the size of a model by reducing the precision of each
#     parameter; so while our model still has 7 billion parameters, it's easier to run on limited
#     hardware.
# 
# At the end of this cell, we'll have a `model` we can call `generate` on with a specially formatted prompt.

# %%
from gemma.config import GemmaConfig, get_config_for_7b, get_config_for_2b
from gemma.model import GemmaForCausalLM
from gemma.tokenizer import Tokenizer
import contextlib
import os
import torch

# Load the model
VARIANT = "7b-it-quant" 
MACHINE_TYPE = "cuda" 
# weights_dir = '../input/gemma/pytorch/gemma-7b-it-quant' 
weights_dir = '../input/gemma/pytorch/7b-it/2' 

@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
  """Sets the default torch dtype to the given dtype."""
  torch.set_default_dtype(dtype)
  yield
  torch.set_default_dtype(torch.float)

# Model Config.
model_config = get_config_for_2b() if "2b" in VARIANT else get_config_for_7b()
model_config.tokenizer = os.path.join(weights_dir, "tokenizer.model")
model_config.quant = "quant" in VARIANT

# Model.
device = torch.device(MACHINE_TYPE)
with _set_default_tensor_type(model_config.get_dtype()):
  model = GemmaForCausalLM(model_config)
  ckpt_path = os.path.join(weights_dir, f'gemma-{VARIANT}.ckpt')
  model.load_weights(ckpt_path)
  model = model.to(device).eval()


# %%
# Now we can loop through our input texts, randomly select a rewrite prompt, and see Gemma in action:

import random
random.seed(0)
# This is the prompt format the model expects
USER_CHAT_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

rewrite_data = []

for original_text in original_texts:
    rewrite_prompt = random.choice(rewrite_prompts)
    prompt = f'{rewrite_prompt}\n{original_text}'
    rewritten_text = model.generate(
        USER_CHAT_TEMPLATE.format(prompt=prompt),
        device=device,
        output_len=100,
    )
    rewrite_data.append({
        'original_text': original_text,
        'rewrite_prompt': rewrite_prompt,
        'rewritten_text': rewritten_text,
    })
    
print(rewrite_data)

# %%
# Let's turn our generated data into a dataframe, and spot check the first rewrite to see if it makes sense.
rewrite_data_df = pd.DataFrame(rewrite_data)
rewrite_data_df[:1].values

# %% [markdown]
# # Next Steps
# 
# Huzzah! We have a dataset with original texts, rewrite prompts, and rewritten text. Here are a couple of suggestions of next steps you could take to generate a larger, more diverse dataset:
# 1. Add more original text data sources; besides just using all of the forum messages (instead of just the first 5), Kaggle has tons of datasets that would make reasonable input text. Here are few random datasets you could use:
#     - The `Plot` column from the [Wikipedia Movie Plots dataset](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots).
#     - The `text` column from the [Emotions dataset](https://www.kaggle.com/datasets/nelgiriyewithana/emotions).
#     - The `body_text` and `abstract` columns of the [Wikibooks Dataset](https://www.kaggle.com/datasets/dhruvildave/wikibooks-dataset).
#     
#     Note that each of these may need different preprocessing; for example, Gemma has a context length of 8192 tokens, so if the text is long, you'll need to truncate it.
# 2. Use gemma to generate original text.
# 3. Expand the list of rewrite prompts. You can come up with them manually, or explore having Gemma write rewrite prompts.
# 4. Play around with the generation of `rewritten_text`:
#    - How does changing `output_len` affect the length and quality of rewrites?
#    - Do rewrites with the 2B parameter model differ substantially from the 7B model?
#    - Can you use [few shot prompting](https://www.promptingguide.ai/techniques/fewshot) to get higher quality rewrites?

# %%



