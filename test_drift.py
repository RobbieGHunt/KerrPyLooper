import numpy as np
import drift_corrector

np.random.seed(42)
ref = np.random.rand(100, 100).astype(np.float32)
target = np.random.rand(400, 400).astype(np.float32)

roi_r, roi_c = 150, 150
patch_h, patch_w = 100, 100
sw = 30

ref_patch = target[roi_r:roi_r+patch_h, roi_c:roi_c+patch_w]

dy, dx, ncc = drift_corrector.estimate_shift_sqdiff(ref_patch, target, roi_r, roi_c, patch_h, patch_w, sw)
print(f'Fast: dy={dy}, dx={dx}, min={np.min(ncc)}')
