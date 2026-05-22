import numpy as np
import pytest
from batch_processor import crop600

def test_crop600_less_than_600_rows():
    arr = np.ones((500, 10))
    result = crop600(arr)
    assert result.shape == (500, 10)
    np.testing.assert_array_equal(result, arr)

def test_crop600_exactly_600_rows():
    arr = np.ones((600, 10))
    result = crop600(arr)
    assert result.shape == (600, 10)
    np.testing.assert_array_equal(result, arr)

def test_crop600_greater_than_600_rows_2d():
    arr = np.ones((700, 10))
    result = crop600(arr)
    assert result.shape == (600, 10)
    np.testing.assert_array_equal(result, arr[:600, :])

def test_crop600_greater_than_600_rows_3d():
    arr = np.ones((800, 10, 3))
    result = crop600(arr)
    assert result.shape == (600, 10, 3)
    np.testing.assert_array_equal(result, arr[:600, :, :])

def test_crop600_1d_less_than_600_rows():
    arr = np.ones(500)
    result = crop600(arr)
    assert result.shape == (500,)
    np.testing.assert_array_equal(result, arr)

def test_crop600_1d_greater_than_600_rows_raises():
    arr = np.ones(700)
    with pytest.raises(IndexError):
        crop600(arr)
