#%%
import polars as pl
from pathlib import Path

input_dir=Path("/home/ryota/programs/kaggle/llm-prompt-recovery/input/llm-prompt-recovery")
def load_data() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    # Load the data
    train_path = input_dir / "train.csv"
    test_path = input_dir / "train.csv"
    sample_path = input_dir / "train.csv"
    df_train = pl.read_csv(train_path)
    df_test = pl.read_csv(test_path)
    df_sample = pl.read_csv(sample_path)    
    return df_train, df_test, df_sample


#%%
df_train, df_test, df_sample =load_data()


def print_row(df: pl.DataFrame, idx: int ):
    original_text = df["original_text"]
    rewrite_prompt = df["rewrite_prompt"] 
    rewritten_text = df["rewritten_text"]
    sep = "========================================"
    print(f"{sep}\noriginal_text: \n{original_text[idx]}\n{sep}")
    print(f"{sep}\nrewrite_prompt: \n{rewrite_prompt[idx]}\n{sep}")
    print(f"{sep}\nrewritten_text: \n{rewritten_text[idx]}\n{sep}")
# %%
print_row(df_train, 0)

