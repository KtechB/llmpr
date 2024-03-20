# cite : https://www.kaggle.com/code/yeoyunsianggeremie/llm-prompt-recovery-metric-computation
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class Metric():
    def __init__(self):
        # self._model = SentenceTransformer('/kaggle/input/sentence-t5-base-hf/sentence-t5-base') # TODO: not official TF model
        self._model = SentenceTransformer('sentence-transformers/sentence-t5-base') # TODO: not official TF model
        self.scs = lambda x, y: abs((cosine_similarity(x, y)) ** 3)
    def calc_score(self, text1:str, text2:str):
        emb1 = self._model.encode(text1, normalize_embeddings=True, show_progress_bar=False).reshape(1, -1)
        emb2 = self._model.encode(text2, normalize_embeddings=True, show_progress_bar=False).reshape(1, -1)
        return self.scs(emb1, emb2)[0][0]

    def calc_scores(self, texts1:list[str], texts2:list[str]):
        emb1 = self._model.encode(texts1, normalize_embeddings=True, show_progress_bar=False)# .reshape(1, -1)
        emb2 = self._model.encode(texts2, normalize_embeddings=True, show_progress_bar=False)# .reshape(1, -1)
        return np.diag(self.scs(emb1,emb2))
    
def CVScore(test):
    scs = lambda row: abs((cosine_similarity(row["actual_embeddings"], row["pred_embeddings"])) ** 3)
    model = SentenceTransformer('/kaggle/input/sentence-t5-base-hf/sentence-t5-base')
    test["actual_embeddings"] = test["rewrite_prompt"].progress_apply(lambda x: model.encode(x, normalize_embeddings=True, show_progress_bar=False).reshape(1, -1))
    test["pred_embeddings"] = test["pred"].progress_apply(lambda x: model.encode(x, normalize_embeddings=True, show_progress_bar=False).reshape(1, -1))
    test["score"] = test.apply(scs, axis=1)
    return np.mean(test['score'])[0][0]