from llmpr.datasets.dataset import create_dataset


def test_create_dataset():
    df = create_dataset(5)
    assert len(df) == 2761*5
