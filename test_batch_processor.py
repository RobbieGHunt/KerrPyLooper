import numpy as np
import pytest
from batch_processor import crop600

def test_crop600_2d_less_than_600():
    arr = np.zeros((500, 100))
    cropped = crop600(arr)
    assert cropped.shape == (500, 100)
    assert np.array_equal(cropped, arr)

def test_crop600_2d_exactly_600():
    arr = np.zeros((600, 100))
    cropped = crop600(arr)
    assert cropped.shape == (600, 100)
    assert np.array_equal(cropped, arr)

def test_crop600_2d_greater_than_600():
    arr = np.zeros((700, 100))
    # Give it some data to ensure we keep the correct rows
    arr[600, 0] = 1
    cropped = crop600(arr)
    assert cropped.shape == (600, 100)
    assert np.array_equal(cropped, arr[:600, :])

def test_crop600_3d_less_than_600():
    arr = np.zeros((500, 100, 3))
    cropped = crop600(arr)
    assert cropped.shape == (500, 100, 3)
    assert np.array_equal(cropped, arr)

def test_crop600_3d_exactly_600():
    arr = np.zeros((600, 100, 3))
    cropped = crop600(arr)
    assert cropped.shape == (600, 100, 3)
    assert np.array_equal(cropped, arr)

def test_crop600_3d_greater_than_600():
    arr = np.zeros((700, 100, 3))
    # Give it some data to ensure we keep the correct rows
    arr[600, 0, 0] = 1
    cropped = crop600(arr)
    assert cropped.shape == (600, 100, 3)
    assert np.array_equal(cropped, arr[:600, :, :])
