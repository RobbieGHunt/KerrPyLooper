import sys
import time
from PyQt5.QtWidgets import QApplication
from drift_corrector import DriftCorrectorWindow
import numpy as np
import pandas as pd
from PIL import Image
import os
import shutil

# Setup output dir
out_dir = "example3_vector/x_drift_corrected"
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir, exist_ok=True)

app = QApplication(sys.argv)
dc = DriftCorrectorWindow()
dc.img_dir = "example3_vector/x"
df = pd.read_csv("example3_vector/x/test_bgd40mT.txt", sep="\t")
df.rename(columns={"File_Name": "File", "File_Original": "File_Original"}, inplace=True)
if "File" not in df.columns and len(df.columns) >= 3:
    df.columns = ["Field", "Intensity", "File"]
if "File" in df.columns and "File_Original" not in df.columns:
    df["File_Original"] = df["File"]
dc.txt_data = df

from shared_utils.image_processing import crop600
def to_gray(arr):
    if arr.ndim == 3:
        return np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
    return arr

arr = np.array(Image.open("example3_vector/x/image0004_unproccessed.png"))
dc.ref_gray = to_gray(crop600(arr))

# Mock UI and methods
class MockROI:
    def __init__(self):
        self.roi = (100, 100, 50, 50)
dc.img_label = MockROI()
class MockSpin:
    def value(self):
        return 10
dc.spin_search = MockSpin()

class MockChk:
    def isChecked(self):
        return False
dc.chk_use_sobel = MockChk()

dc._log = lambda *args: None
dc._plot_drift_curves = lambda: None
dc._recompute_crop_bounds = lambda: None
dc._refresh_display = lambda: None
class MockTabs:
    def setCurrentIndex(self, i): pass
dc.preview_tabs = MockTabs()

# Mock crop bounds which are computed by _recompute_crop_bounds
dc.crop_top = 0
dc.crop_bottom = 0
dc.crop_left = 0
dc.crop_right = 0

# Override QFileDialog
from PyQt5.QtWidgets import QFileDialog
def mock_getExistingDirectory(*args, **kwargs):
    return out_dir
QFileDialog.getExistingDirectory = mock_getExistingDirectory

print("Running estimation...")
start = time.time()
dc._run_estimation()
print(f"Estimation time: {time.time() - start:.2f}s")

print("Running save...")
start = time.time()
dc._save_corrected()
print(f"Save time: {time.time() - start:.2f}s")

files = os.listdir(out_dir)
print(f"Saved {len(files)} files to {out_dir}")
if len(files) > 0:
    print(f"First file: {files[0]}")
    arr_saved = np.array(Image.open(os.path.join(out_dir, files[0])))
    print(f"Image shape: {arr_saved.shape}")
