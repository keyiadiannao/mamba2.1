from src.navigator.base import merge_path_summaries


def test_merge_path_summaries_mean_then_tail() -> None:
    prev = [2.0, 4.0]
    cur = [4.0, 8.0, 1.0]
    out = merge_path_summaries(prev, cur)
    assert out[:2] == [3.0, 6.0]
    assert out[2] == 1.0
