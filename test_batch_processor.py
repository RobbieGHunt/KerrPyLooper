import numpy as np
import pytest
from shared_utils.image_processing import crop600

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

def calculate_local_hc_reference(img_stack, field, bin_size=4):
    import concurrent.futures
    from shared_utils.image_processing import extract_hc_from_loop
    N, H, W = img_stack.shape
    binned_stack = img_stack[:, ::bin_size, ::bin_size]
    b_H, b_W = binned_stack.shape[1], binned_stack.shape[2]
    hc_map = np.zeros((b_H, b_W), dtype=np.float32)
    def process_pixel(r, c):
        intensity = binned_stack[:, r, c]
        zero_idx = np.argmin(np.abs(field))
        intensity = intensity - intensity[zero_idx]
        return r, c, extract_hc_from_loop(field, intensity)
    coords = [(r, c) for r in range(b_H) for c in range(b_W)]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda args: process_pixel(*args), coords))
    for r, c, hc in results:
        hc_map[r, c] = hc
    hc_map_full = np.zeros((H, W), dtype=np.float32)
    for r in range(b_H):
        for c in range(b_W):
            hc_map_full[r*bin_size:(r+1)*bin_size, c*bin_size:(c+1)*bin_size] = hc_map[r, c]
    return hc_map_full

def test_calculate_local_hc_equivalence():
    from shared_utils.image_processing import calculate_local_hc
    # Create random dummy image stack (N=20, H=24, W=32)
    np.random.seed(42)
    img_stack = np.random.uniform(10, 255, (20, 24, 32)).astype(np.float32)
    field = np.linspace(-10, 10, 20).astype(np.float32)
    
    bin_size = 4
    res_opt = calculate_local_hc(img_stack, field, bin_size=bin_size, apply_corrections=False)
    res_ref = calculate_local_hc_reference(img_stack, field, bin_size=bin_size)
    
    np.testing.assert_array_almost_equal(res_opt, res_ref)

def calculate_local_hr_reference(img_stack, field, bin_size=4):
    from batch_processor import compute_hc_hr
    N, H, W = img_stack.shape
    binned_stack = img_stack[:, ::bin_size, ::bin_size]
    b_H, b_W = binned_stack.shape[1], binned_stack.shape[2]
    hr_map = np.zeros((b_H, b_W), dtype=np.float32)
    zero_idx = np.argmin(np.abs(field))
    for r in range(b_H):
        for c in range(b_W):
            intensity = binned_stack[:, r, c]
            intensity = intensity - intensity[zero_idx]
            res = compute_hc_hr(field, intensity)
            hr_map[r, c] = res["hr_abs"]
    hr_map_full = np.zeros((H, W), dtype=np.float32)
    for r in range(b_H):
        for c in range(b_W):
            hr_map_full[r*bin_size:(r+1)*bin_size, c*bin_size:(c+1)*bin_size] = hr_map[r, c]
    return hr_map_full

def test_calculate_local_hr_equivalence():
    from shared_utils.image_processing import calculate_local_hc
    np.random.seed(42)
    img_stack = np.random.uniform(10, 255, (20, 24, 32)).astype(np.float32)
    field = np.linspace(-10, 10, 20).astype(np.float32)
    
    bin_size = 4
    _, res_opt = calculate_local_hc(img_stack, field, bin_size=bin_size, return_hr=True, apply_corrections=False)
    res_ref = calculate_local_hr_reference(img_stack, field, bin_size=bin_size)
    
    np.testing.assert_array_almost_equal(res_opt, res_ref)
