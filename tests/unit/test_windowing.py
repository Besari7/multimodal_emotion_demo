from src.data.windowing import WindowConfig, iter_windows


def test_windowing_padded_tail() -> None:
    windows = list(iter_windows(duration_sec=6.0, config=WindowConfig(length_sec=5.0, overlap=0.5, stride_sec=2.5)))
    assert len(windows) == 3
    assert windows[-1].is_padded is True
    assert abs(windows[-1].pad_right_sec - 1.5) < 1e-6
