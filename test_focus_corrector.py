import numpy as np
from focus_corrector import find_defect_roi

def test_find_defect_roi_edge_case():
    # An image smaller than patch_size + 2 * margin (128 + 40 = 168)
    # This will result in len(r_range) == 0 or len(c_range) == 0
    img = np.zeros((100, 100))
    result = find_defect_roi(img)
    assert result == (150, 150), f"Expected (150, 150), but got {result}"
