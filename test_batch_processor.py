import numpy as np
import pytest
from shared_utils.image_processing import crop600
from batch_processor import compute_hc_hr

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

def test_compute_hc_hr_ideal_symmetric_loop():
    # Field: 10, 5, 0, -5, -10, -5, 0, 5, 10
    field = np.array([10, 5, 0, -5, -10, -5, 0, 5, 10], dtype=float)
    # ycorr: ideal symmetric interpolation. desc crosses 0 at -2.5.
    ycorr = np.array([1, 1, 0.5, -0.5, -1, -1, -0.5, 0.5, 1], dtype=float)

    res = compute_hc_hr(field, ycorr)

    assert np.isclose(res['hc_pos'], 2.5)
    assert np.isclose(res['hc_neg'], -2.5)
    assert np.isclose(res['hc_avg'], 2.5)
    assert np.isclose(res['hr_asc'], -0.5)
    assert np.isclose(res['hr_desc'], 0.5)
    assert np.isclose(res['hr_abs'], 0.5)

def test_compute_hc_hr_asymmetric_loop():
    field = np.array([10, 5, 0, -5, -10, -5, 0, 5, 10], dtype=float)
    # desc crosses zero at -4, asc crosses zero at +2
    ycorr = np.array([1, 1, 0.8, -0.2, -1, -1, -0.4, 0.6, 1], dtype=float)

    res = compute_hc_hr(field, ycorr)

    assert np.isclose(res['hc_pos'], 2.0)
    assert np.isclose(res['hc_neg'], -4.0)
    assert np.isclose(res['hc_avg'], 3.0)
    assert np.isclose(res['hr_asc'], -0.4)
    assert np.isclose(res['hr_desc'], 0.8)

def test_compute_hc_hr_inverted_loop():
    field = np.array([10, 5, 0, -5, -10, -5, 0, 5, 10], dtype=float)
    # ycorr inverted
    ycorr = np.array([-1, -1, -0.5, 0.5, 1, 1, 0.5, -0.5, -1], dtype=float)

    res = compute_hc_hr(field, ycorr)

    assert np.isclose(res['hc_pos'], 2.5)
    assert np.isclose(res['hc_neg'], -2.5)
    assert np.isclose(res['hc_avg'], 2.5)
    assert np.isclose(res['hr_asc'], 0.5)
    assert np.isclose(res['hr_desc'], -0.5)

def test_compute_hc_hr_no_crossing():
    field = np.array([10, 5, 0, -5, -10, -5, 0, 5, 10], dtype=float)
    ycorr = np.ones_like(field)

    res = compute_hc_hr(field, ycorr)

    # When there's no crossing, fallback logic picks argmin(abs(y - mid)), where mid is 1.
    # All are 1, so it picks the first element in sorted array, which for field is -10.
    assert res['hc_pos'] == -10.0
    assert res['hc_neg'] == -10.0
    assert res['hc_avg'] == 10.0
    assert res['hr_asc'] == 1.0
    assert res['hr_desc'] == 1.0

def test_compute_hc_hr_nearest_point_remanence():
    # Field points don't include exactly 0
    field = np.array([10, 4, 1, -2, -6, -10, -5, -1, 3, 10], dtype=float)
    ycorr = np.array([1, 1, 0.5, -0.5, -1, -1, -0.5, 0.5, 1, 1], dtype=float)

    res = compute_hc_hr(field, ycorr)

    # Nearest points to 0 are 1 (desc) and -1 (asc)
    # desc branch: nearest 0 is 1. hr_desc = y(1) = 0.5
    # asc branch: nearest 0 is -1. hr_asc = y(-1) = 0.5
    assert np.isclose(res['hr_asc'], 0.5)
    assert np.isclose(res['hr_desc'], 0.5)
    assert np.array_equal(res['hr_fields'], [-1.0, 1.0])
    assert np.array_equal(res['hr_vals'], [0.5, 0.5])
