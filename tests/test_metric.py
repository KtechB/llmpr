
import numpy as np
from llmpr.metric import Metric


def test_calc_scores():
    metric = Metric()
    texts1 = ["hello world", "goodbye world"]
    texts2 = ["hello world", "goodbye world"]
    expected = np.array([1.0, 1.0])
    actual = metric.calc_scores(texts1, texts2)
    print(actual)
    assert expected.shape == actual.shape
    assert (expected == actual).all()

    