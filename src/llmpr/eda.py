import pandas as pd

input_dir = "../input"
# check duplication os rewrite_prompt
# count of unique rewrite_prompt



# Speed up the data frame using the groupby method


# df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder


# PublisherをLabelEncoderで数値に置き換える
def get_punlisher_groups(train: pd.DataFrame, group_column):
    le = LabelEncoder()
    le.fit(train[group_column])
    groups = le.transform(train[group_column])
    return groups


def get_train_val():
    df = pd.read_csv(f"{input_dir}/nbroad/gemma-rewrite-nbroad/nbroad-v1.csv")
    groups = get_punlisher_groups(df, "rewrite_prompt")

    group_kf = GroupKFold(n_splits=5)  # n_splitは任意の数で
    for i, (tr_idx, va_idx) in enumerate(group_kf.split(df, None,groups, )):
        # trainとvalidに分ける
        df_train, df_val = df.iloc[tr_idx], df.iloc[va_idx]
        break
    return df_train, df_val

df_train, df_val = get_train_val()

unique_values = df_train["rewrite_prompt"].unique()
unique_values_val = df_val["rewrite_prompt"].unique()
print(f"both included value in train and val, {set(unique_values) & set(unique_values_val)}")
print(len(df_train), unique_values[-1])
print(len(df_val), unique_values_val[-1])