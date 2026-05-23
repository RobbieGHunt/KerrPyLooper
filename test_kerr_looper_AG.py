import numpy as np
import pytest
from kerr_looper_AG import normalize_image

def test_normalize_image_normal():
    # Test case with non-zero standard deviation
    img = np.array([[1.0, 2.0], [3.0, 4.0]])
    res = normalize_image(img)

    # Check that mean is approximately 0
    assert np.isclose(np.mean(res), 0.0)
    # Check that std is approximately 1
    assert np.isclose(np.std(res), 1.0)

    # Hand-calculate expected for this matrix
    # mean = 2.5, std = ~1.11803
    m = np.mean(img)
    s = np.std(img)
    expected = (img - m) / s
    np.testing.assert_array_almost_equal(res, expected)

def test_normalize_image_zero_std():
    # Test case with zero standard deviation (all elements the same)
    img = np.array([[5.0, 5.0], [5.0, 5.0]])
    res = normalize_image(img)

    # Fallback should return img - mean, which should be all zeros
    expected = np.zeros_like(img)
    np.testing.assert_array_almost_equal(res, expected)

def test_normalize_image_single_pixel():
    # Test case with a single pixel (zero standard deviation)
    img = np.array([[10.0]])
    res = normalize_image(img)
    expected = np.array([[0.0]])
    np.testing.assert_array_almost_equal(res, expected)
