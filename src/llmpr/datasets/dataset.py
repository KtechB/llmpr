import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from llmpr.metric import Metric

logger = logging.getLogger(__name__)

INPUT_DIR = "../input"
# NLTKの必要な部分をインポート
import nltk
from nltk.tokenize import sent_tokenize

# NLTKのデータとトークナイザをダウンロード（初回のみ必要）
nltk.download('punkt')

def get_sentences_within_limit(text, char_limit=600):
    """
    指定された文字数以内に収まるようにテキストから文章を取得する関数。
    :param text: 分割したいテキスト。
    :param char_limit: 文章の合計の最大文字数。
    :return: 合計文字数がchar_limit以内に収まる最初のいくつかの文章のリスト。
    """
    # テキストを文章ごとに分割
    sentences = sent_tokenize(text)
    
    # 累積的にchar_limit文字以内に収まるように文章を取得
    accumulated_sentences = []
    total_length = 0
    
    for sentence in sentences:
        # 追加後の合計長を計算
        new_total_length = total_length + len(sentence)
        
        # 合計がchar_limit文字以下の場合のみ、文章を追加
        if new_total_length <= char_limit:
            accumulated_sentences.append(sentence)
            total_length = new_total_length
        else:
            # char_limit文字を超えるとループを終了
            break
    
    if len(accumulated_sentences) == 0:
        return np.nan
    return " ".join(accumulated_sentences)

def load_original_texts(metric: Optional[Metric] = None) -> list[str]:
    paths = [
        # 4000 text
        # 1
        # "nbroad/gemma-rewrite-nbroad/nbroad-v1.csv",  # 0.4871862232685089,
        # "nbroad/gemma-rewrite-nbroad/nbroad-v2.csv",  #  0.509476900100708
        # 2 Supplementary Texts Rewritten by GenAI Models By host pu
        # なし
        # 3  only 100 prompt , paass
        # 4 4343 Samples Dataset containing : 2541 samples from Gemma and 1802 from Gemini By NEWTON BABA
        # Dataset: https://www.kaggle.com/datasets/newtonbaba12345/llm-prompt-recovery-data-gemini-and-gemma/data
        # Discussion: https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/481151
        #  4000 unique prompt
        "newtonbaba12345/llm-prompt-recovery-data-gemini-and-gemma/gemma_data_set_prompt_recover_1.csv",  # 0.475982129573822
        "newtonbaba12345/llm-prompt-recovery-data-gemini-and-gemma/gemma_data_set_prompt_recover_2.csv",  # 0.4747753441333771
        "newtonbaba12345/llm-prompt-recovery-data-gemini-and-gemma/gemini_data_set_prompt_recover_3.csv",  # 0.5489454865455627
        # 500 unique prompt
        # "dipamc77/3000-rewritten-texts-prompt-recovery-challenge/prompts_0_500_wiki_first_para_3000.csv",
        # "dipamc77/3000-rewritten-texts-prompt-recovery-challenge/rewrite_prompts.csv",
        "dipamc77/3000-rewritten-texts-prompt-recovery-challenge/prompts_0_500_wiki_first_para_3000.csv",  # 0.5432605743408203
        # Rewrite Prompts Dataset(400 rewrite prompts using ChatGPT) By ILAN MEISSONNIER
        # Dataset: https://www.kaggle.com/datasets/ilanmeissonnier/chatgpt-rewrite-promts/data
        # Discussion: https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/480771
        # "ilanmeissonnier/chatgpt-rewrite-promts/prompts.csv",  # rewrite prompt only 0.5176106691360474
        # https://www.kaggle.com/datasets/juanmerinobermejo/rewritten-texts-with-gemma-2b
        # 5000 rewriten prompt original text
        # "juanmerinobermejo/rewritten-texts-with-gemma-2b/rewritten_texts_csv.csv",  # 0.6173520684242249  original_text,rewritten_text,prompt,final_prompt
    ]
    column  = "original_text"
    dfs = []
    for path in paths:
        # try:
        # df = pd.read_csv(f"{input_dir}/{path}", encoding="utf-8")
        df = pd.read_csv(f"{INPUT_DIR}/{path}")
        # except:
        # df = pd.read_csv(f"{input_dir}/{path}", encoding="CP932")

        if column not in df.columns:
            print(f"rewrite_prompt not in {path}")
            print(df.columns)
            raise ValueError()
        else:
            df_prompt = df[column]
        print("==============================")
        print(Path(path))
        print(df_prompt.nunique())
        if df_prompt.isna().any():
            original_length = len(df_prompt)
            df_prompt = df_prompt.dropna()
            print(f"drop na: {original_length - len(df_prompt)}")


        # calc length of text statistics
        print(df_prompt.str.len().describe())

        dfs.append(df_prompt)
    
    df_all:pd.Series = pd.concat(dfs)
    df_all = df_all.apply(lambda x: get_sentences_within_limit(x, 1000)).dropna()
    
    print("=================")
    print("data num:", len(df_all))
    print("length mean:\n",df_all.str.len().describe())
    return list(set(df_all.values))


def remove_prefixes(strings:list[str])->list[str]:
    # 新しい文字列のリストを保持するための空リストを初期化します。
    new_strings = []
    # 正規表現パターン: 数字に続くピリオドとスペース
    pattern = re.compile(r'^\d+\.\s+')
    # 各文字列に対してループを実行します。
    for s in strings:
        # 正規表現を使用してプレフィックスを削除します。
        new_string = re.sub(pattern, '', s)
        # 新しい文字列をリストに追加します。
        new_strings.append(new_string)
    return new_strings
def load_prompts(metric: Optional[Metric] = None) -> list[str]:
    paths = [
        # 4000 text
        # 1
        ## "nbroad/gemma-rewrite-nbroad/nbroad-v1.csv"  # 0.4871862232685089,
        # "nbroad/gemma-rewrite-nbroad/nbroad-v2.csv",  #  0.509476900100708
        # 2 Supplementary Texts Rewritten by GenAI Models By host pu
        # なし
        # 3  only 100 prompt , paass
        # 4 4343 Samples Dataset containing : 2541 samples from Gemma and 1802 from Gemini By NEWTON BABA
        # Dataset: https://www.kaggle.com/datasets/newtonbaba12345/llm-prompt-recovery-data-gemini-and-gemma/data
        # Discussion: https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/481151
        #  4000 unique prompt
        # "newtonbaba12345/llm-prompt-recovery-data-gemini-and-gemma/gemma_data_set_prompt_recover_1.csv",  # 0.475982129573822
        # "newtonbaba12345/llm-prompt-recovery-data-gemini-and-gemma/gemma_data_set_prompt_recover_2.csv",  # 0.4747753441333771
        "newtonbaba12345/llm-prompt-recovery-data-gemini-and-gemma/gemini_data_set_prompt_recover_3.csv",  # 0.5489454865455627
        # 500 unique prompt
        # "dipamc77/3000-rewritten-texts-prompt-recovery-challenge/prompts_0_500_wiki_first_para_3000.csv",
        # "dipamc77/3000-rewritten-texts-prompt-recovery-challenge/rewrite_prompts.csv",
        "dipamc77/3000-rewritten-texts-prompt-recovery-challenge/prompts_0_500_wiki_first_para_3000.csv",  # 0.5432605743408203
        # Rewrite Prompts Dataset(400 rewrite prompts using ChatGPT) By ILAN MEISSONNIER
        # Dataset: https://www.kaggle.com/datasets/ilanmeissonnier/chatgpt-rewrite-promts/data
        # Discussion: https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/480771
        "ilanmeissonnier/chatgpt-rewrite-promts/prompts.csv",  # rewrite prompt only 0.5176106691360474
        # https://www.kaggle.com/datasets/juanmerinobermejo/rewritten-texts-with-gemma-2b
        # 5000 rewriten prompt original text
        "juanmerinobermejo/rewritten-texts-with-gemma-2b/rewritten_texts_csv.csv",  # 0.6173520684242249  original_text,rewritten_text,prompt,final_prompt
    ]

    # 1. Load the data
    column = "rewrite_prompt"
    dfs = []
    for path in paths:
        # try:
        # df = pd.read_csv(f"{input_dir}/{path}", encoding="utf-8")
        df = pd.read_csv(f"{INPUT_DIR}/{path}")
        # except:
        # df = pd.read_csv(f"{input_dir}/{path}", encoding="CP932")

        if "rewrite_prompt" not in df.columns:
            print(f"rewrite_prompt not in {path}")
            df_prompt = df["prompt"]
        else:
            df_prompt = df[column]
        print(Path(path))
        print(df_prompt.nunique())
        baseline_text = "Improve the text to this."
        if metric:
            metrics = metric.calc_scores(
                list(df_prompt.values), [baseline_text] * len(df_prompt)
            )
            print(f"simirarity mean: {metrics.mean()}")

        dfs.append(df_prompt)

    df_all = pd.concat(dfs).dropna()
    return list(set(df_all.values))


def show_metrics_of_dataset():
    metric = Metric()
    prompts = load_prompts(metric)

    baseline_text = "Improve the text to this."
    # print(df_all["rewrite_prompt"].values)
    metrics = metric.calc_scores(list(prompts), [baseline_text] * len(prompts))
    print(f"unique prompt: {len(prompts)}")
    print(metrics.mean())


def create_dataset(epoch: int = 5 , seed=42):

    np.random.seed(seed)
    rewrite_prompts_original = load_prompts()
    rewrite_prompts_original = remove_prefixes(rewrite_prompts_original)
    logger.info(f"loaded rewrite_prompts_original: {len(rewrite_prompts_original)}")

    original_texts_original= load_original_texts()
    logger.info(f"loaded original_texts_original: {len(original_texts_original)}")
    rewrite_prompts = []
    original_texts = []
    original_texts = []
    for rewrite_prompt in rewrite_prompts_original:
        rewrite_prompts += [rewrite_prompt] * epoch 
        original_texts += np.random.choice(original_texts_original, size=epoch, replace=False).tolist()
    assert len(rewrite_prompts) == len(original_texts), f"text length wrong {len(rewrite_prompts)} != {len(original_texts)}"


    df = pd.DataFrame({"original_text": original_texts, "rewrite_prompt": rewrite_prompts})
    # deduplicate
    df = df.drop_duplicates().reset_index(drop=True)
    return df

if __name__ == "__main__":
    #logger config
    logging.basicConfig(level=logging.INFO)
    # load_original_texts()
    # show_metrics_of_dataset()
    num = 5
    df = create_dataset(num ,seed=43)
    print(f"created data num: {len(df)}")
    # save to csv
    df.to_csv(f"data/llmpr_prompt_text_{num}.csv", index=False)
    

    
