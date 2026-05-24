import sys
import time
from PyQt5.QtWidgets import QApplication
from drift_corrector import DriftCorrectorWindow
import numpy as np
import pandas as pd
from PIL import Image

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

start = time.time()
dc._run_estimation()
end = time.time()
print(f"Estimation time: {end - start:.2f}s")
