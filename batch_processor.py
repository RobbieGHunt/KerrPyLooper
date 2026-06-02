# -*- coding: utf-8 -*-
"""
batch_processor.py
==================
Headless batch processor for Kerr MOKE hysteresis loops.

Scans a parent directory for data sub-directories (each containing images and
a .txt data file), applies the same automatic corrections used in the
interactive tool, computes Hc and Hr, and saves per-loop results plus summary
plots to an "Analysis" folder inside the parent directory.

If sub-directory names contain a step number (e.g. "step0_AD", "Step2",
"run_step3_fine") the results are additionally plotted as Hc vs. step number
and Hr vs. step number.

Usage
-----
    py batch_processor.py <parent_directory>

or, if no argument is supplied, a Tk folder-picker dialog appears.

Reuses pure-Python logic from kerr_looper_AG.py (no PyQt5 required).
"""

import sys
import os
import re
import argparse
import datetime
import time
from shared_utils.image_processing import crop_batch, compute_subtracted_mean
import threading

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")

# Add script directory to sys.path to ensure gui_styles import works
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import multiprocessing

# Avoid importing PyQt5 inside worker processes to reduce startup overhead
_is_child = False
try:
    if hasattr(multiprocessing, "parent_process") and multiprocessing.parent_process() is not None:
        _is_child = True
except Exception:
    pass
if not _is_child and multiprocessing.current_process().name != "MainProcess":
    _is_child = True

if _is_child:
    HAS_PYQT5 = False
else:
    try:
        from PyQt5.QtWidgets import (
            QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
            QLabel, QLineEdit, QCheckBox, QFileDialog, QTextEdit, QGroupBox,
            QComboBox, QSpinBox
        )
        from PyQt5.QtCore import Qt, QThread, pyqtSignal
        from PyQt5.QtGui import QTextCursor
        from gui_styles import apply_theme
        HAS_PYQT5 = True
    except ImportError:
        HAS_PYQT5 = False

# ---------------------------------------------------------------------------
# Utility helpers (mirror of kerr_looper_AG.py, no GUI dependency)
# ---------------------------------------------------------------------------

class SeriesWrapper:
    def __init__(self, values):
        self.values = values
    def to_numpy(self, dtype=None):
        arr = np.array(self.values)
        if dtype:
            return arr.astype(dtype)
        return arr
    def tolist(self):
        return list(self.values)
    @property
    def str(self):
        class StrAccessor:
            def __init__(self, vals):
                self.vals = vals
            def strip(self):
                return SeriesWrapper([v.strip() for v in self.vals])
        return StrAccessor(self.values)
    def __len__(self):
        return len(self.values)
    def __iter__(self):
        return iter(self.values)


class SimpleDataFrame:
    def __init__(self, fields, intensities, files):
        self.fields = np.array(fields, dtype=np.float32)
        self.intensities = np.array(intensities, dtype=np.float32)
        self.files = files
        
    def __getitem__(self, key):
        if key == "Field":
            return SeriesWrapper(self.fields)
        elif key == "Intensity":
            return SeriesWrapper(self.intensities)
        elif key == "File":
            return SeriesWrapper(self.files)
        raise KeyError(key)

    class Row:
        def __init__(self, field, intensity, file):
            self.Field = field
            self.Intensity = intensity
            self.File = file

    def itertuples(self, index=False):
        return [self.Row(f, i, fn) for f, i, fn in zip(self.fields, self.intensities, self.files)]

    def __len__(self):
        return len(self.files)


def _parse_txt_file(txt_file: str) -> "SimpleDataFrame | None":
    """
    Attempt to read and parse a single text file into a structured DataFrame.
    Returns the formatted DataFrame if valid, otherwise None.
    """
    # Skip empty files
    if os.path.getsize(txt_file) == 0:
        return None

    # Quick check: a valid data file must reference ".png" files
    with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
        head = f.read(1024 * 1024)  # Read up to 1MB
    if ".png" not in head.lower():
        return None

    fields = []
    intensities = []
    files = []
    
    with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Split by whitespace/tabs
            parts = [p.strip() for p in line.split() if p.strip()]
            if len(parts) >= 3 and parts[2].lower().endswith('.png'):
                try:
                    f_val = float(parts[0])
                    i_val = float(parts[1])
                    fields.append(f_val)
                    intensities.append(i_val)
                    files.append(parts[2])
                except ValueError:
                    continue

    if len(files) >= 3:
        return SimpleDataFrame(fields, intensities, files)

    return None


def load_txt_data(data_dir: str, print_func=print):
    """
    Find a valid .txt data file in *data_dir* and return a DataFrame with columns
    Field, Intensity, File. Returns None if no valid file is found.
    If multiple .txt files exist, attempts to identify the correct one
    by checking for structured data.
    """
    txt_files = [fn for fn in os.listdir(data_dir) if fn.lower().endswith(".txt")]
    
    if not txt_files:
        return None

    for fn in txt_files:
        txt_file = os.path.join(data_dir, fn)
        try:
            df = _parse_txt_file(txt_file)
            if df is not None:
                return df
        except Exception as exc:
            print_func(f"  [warn] Could not parse {txt_file}: {exc}")
            continue

    return None


def find_background_image(data_dir: str, image_files: list) -> str | None:
    """
    Heuristic: use 'mask.png' if present, otherwise the first image
    alphabetically.
    """
    lower = [f.lower() for f in image_files]
    if "mask.png" in lower:
        idx = lower.index("mask.png")
        return image_files[idx]
    return image_files[0] if image_files else None


def pil_to_numpy_fast(img) -> np.ndarray:
    """
    Fast conversion of PIL Image to numpy array using direct buffer access when possible.
    """
    mode = img.mode
    if mode == "I;16":
        return np.frombuffer(img.tobytes(), dtype=np.uint16).reshape(img.height, img.width)
    elif mode == "L":
        return np.frombuffer(img.tobytes(), dtype=np.uint8).reshape(img.height, img.width)
    return np.array(img)


def run_subtraction_loop(data_dir: str, txt_df: "pd.DataFrame",
                         background_array: np.ndarray,
                         print_func=print,
                         cancel_event=None) -> np.ndarray:
    """
    Subtract *background_array* from each image listed in *txt_df* and return
    the mean intensity of the difference as a 1-D float32 array.
    """
    means = []

    # ⚡ Bolt: Pre-crop and cast the background array to float32
    # This avoids O(N) redundant memory allocations and conversions inside the loop.
    bg_cropped_f32 = crop_batch(background_array).astype(np.float32)

    for row in txt_df.itertuples(index=False):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Batch run cancelled by user")
        img_file = row.File.strip()
        img_path = os.path.join(data_dir, img_file)
        try:
            img_arr = pil_to_numpy_fast(Image.open(img_path))
            mean_val = compute_subtracted_mean(img_arr, bg_cropped_f32, crop_func=crop_batch)
            means.append(mean_val)
        except FileNotFoundError:
            means.append(np.nan)
        except Exception as exc:
            print_func(f"  [warn] Error processing {img_file}: {exc}")
            means.append(np.nan)
    return np.array(means, dtype=np.float32)


def auto_correct_coeffs(field: np.ndarray, intensity: np.ndarray,
                        drift_corr: bool = True, linear_corr: bool = True,
                        quad_corr: bool = True, print_func=print) -> dict:
    """
    Compute drift, linear Faraday, quadratic Faraday, and quad_offset
    corrections using a branch-aware, shelf-independent algorithm.

    This ensures that the MOKE hysteresis contrast (vertical step height) does
    not contaminate the background slope estimation.
    """
    idx           = np.arange(len(field), dtype=np.float32)
    idx_off       = idx - idx.mean()
    field_off     = field - np.mean(field)
    field_abs_max = float(np.max(np.abs(field_off)))

    # ------------------------------------------------------------------
    # Pass 1 – endpoint drift alignment
    # ------------------------------------------------------------------
    drift1     = float((intensity[0] - intensity[-1]) / len(intensity))
    intensity1 = intensity + drift1 * idx_off

    # ------------------------------------------------------------------
    # Branch separation (ascending vs descending field sweep)
    # ------------------------------------------------------------------
    i_min = int(np.argmin(field))
    i_max = int(np.argmax(field))
    if i_min < i_max:
        asc_idx  = np.arange(i_min, i_max + 1)
        desc_idx = np.concatenate([np.arange(i_max, len(field)),
                                   np.arange(0, i_min + 1)])
    else:
        desc_idx = np.arange(i_max, i_min + 1)
        asc_idx  = np.concatenate([np.arange(i_min, len(field)),
                                   np.arange(0, i_max + 1)])

    # ------------------------------------------------------------------
    # Quick Hc pre-estimate to guard against high-coercivity loops
    # ------------------------------------------------------------------
    mid_rough = 0.5 * (float(np.max(intensity1)) + float(np.min(intensity1)))
    shifted = intensity1 - mid_rough
    sign_diff = np.diff(np.sign(shifted))
    crossing_idx = np.nonzero(sign_diff)[0]

    asc_crossings = []
    desc_crossings = []
    if len(crossing_idx) > 0:
        for ci in crossing_idx:
            y0, y1 = shifted[ci], shifted[ci + 1]
            f0, f1 = field_off[ci], field_off[ci + 1]
            if y0 != y1:
                frac = -y0 / (y1 - y0)
                f_cross = f0 + frac * (f1 - f0)
                if ci in asc_idx:
                    asc_crossings.append(f_cross)
                elif ci in desc_idx:
                    desc_crossings.append(f_cross)

    hc_pos_rough = float(np.mean(asc_crossings)) if asc_crossings else 0.0
    hc_neg_rough = float(np.mean(desc_crossings)) if desc_crossings else 0.0

    # Determine adaptive sat_threshold for high field ranges
    sat_threshold = 0.85 * field_abs_max
    rough_hc_abs = max(abs(hc_pos_rough), abs(hc_neg_rough))
    if rough_hc_abs > 0.65 * field_abs_max:
        sat_threshold = min(0.95 * field_abs_max, rough_hc_abs + 0.10 * field_abs_max)
        print_func(f"  [info] High coercivity detected (Hc ~ {rough_hc_abs:.2f} mT), trying sat threshold {sat_threshold:.2f} mT")

    # Calculate 4 regions (before-coercivity and high-field for each sweep)
    H_min_off = float(np.min(field_off))
    H_max_off = float(np.max(field_off))

    # 'Before coercivity' thresholds (80% of the range from absolute max starting field to coercivity)
    thresh_asc = H_min_off + 0.8 * (hc_pos_rough - H_min_off)
    thresh_desc = H_max_off - 0.8 * (H_max_off - hc_neg_rough)

    # Initialize shelves for ascending branch
    sat_pos_asc = field_off[asc_idx] > sat_threshold
    sat_neg_asc = field_off[asc_idx] < thresh_asc

    # Initialize shelves for descending branch
    sat_pos_desc = field_off[desc_idx] > thresh_desc
    sat_neg_desc = field_off[desc_idx] < -sat_threshold

    # Fallback to standard shelves if we don't have enough points
    if np.sum(sat_pos_asc) < 2 or np.sum(sat_neg_asc) < 2 or np.sum(sat_pos_desc) < 2 or np.sum(sat_neg_desc) < 2:
        sat_pos_asc = field_off[asc_idx] > sat_threshold
        sat_neg_asc = field_off[asc_idx] < -sat_threshold
        sat_pos_desc = field_off[desc_idx] > sat_threshold
        sat_neg_desc = field_off[desc_idx] < -sat_threshold

    # ------------------------------------------------------------------
    # Linear Faraday: fit positive and negative saturation shelves
    # independently on each branch, then average.
    # ------------------------------------------------------------------
    slopes = []
    # Ascending branch
    f_asc = field_off[asc_idx]
    y_asc = intensity1[asc_idx]
    branch_slopes = []
    if np.sum(sat_pos_asc) >= 2:
        p_pos = np.polyfit(f_asc[sat_pos_asc], y_asc[sat_pos_asc], 1)
        branch_slopes.append(p_pos[0])
    if np.sum(sat_neg_asc) >= 2:
        p_neg = np.polyfit(f_asc[sat_neg_asc], y_asc[sat_neg_asc], 1)
        branch_slopes.append(p_neg[0])
    if branch_slopes:
        slopes.append(np.mean(branch_slopes))

    # Descending branch
    f_desc = field_off[desc_idx]
    y_desc = intensity1[desc_idx]
    branch_slopes = []
    if np.sum(sat_pos_desc) >= 2:
        p_pos = np.polyfit(f_desc[sat_pos_desc], y_desc[sat_pos_desc], 1)
        branch_slopes.append(p_pos[0])
    if np.sum(sat_neg_desc) >= 2:
        p_neg = np.polyfit(f_desc[sat_neg_desc], y_desc[sat_neg_desc], 1)
        branch_slopes.append(p_neg[0])
    if branch_slopes:
        slopes.append(np.mean(branch_slopes))

    linear_val = -float(np.mean(slopes)) if slopes else 0.0
    intensity2 = intensity1 + linear_val * field_off

    # ------------------------------------------------------------------
    # Residual quadratic (Cotton–Mouton): fit on the step-subtracted background
    # ------------------------------------------------------------------
    sat_pos_all = np.zeros(len(field), dtype=bool)
    sat_pos_all[asc_idx] = sat_pos_asc
    sat_pos_all[desc_idx] = sat_pos_desc

    sat_neg_all = np.zeros(len(field), dtype=bool)
    sat_neg_all[asc_idx] = sat_neg_asc
    sat_neg_all[desc_idx] = sat_neg_desc

    sat_all = sat_pos_all | sat_neg_all

    quad1           = 0.0
    quad_offset_val = 0.0

    if np.sum(sat_pos_all) >= 2 and np.sum(sat_neg_all) >= 2:
        # Subtract shelf means to get background-only signal at saturation
        M_pos = np.mean(intensity2[sat_pos_all])
        M_neg = np.mean(intensity2[sat_neg_all])
        y_bg = intensity2.copy()
        y_bg[sat_pos_all] -= M_pos
        y_bg[sat_neg_all] -= M_neg

        try:
            p2 = np.polyfit(field_off[sat_all], y_bg[sat_all], 2)
            a2, b2 = float(p2[0]), float(p2[1])
            if abs(a2) > 0:
                candidate_offset = -b2 / (2.0 * a2)
                if abs(candidate_offset) <= field_abs_max:
                    # Physically plausible vertex position – use full quadratic
                    quad1           = -a2
                    quad_offset_val = candidate_offset
                else:
                    # Vertex far outside field range – absorb only linear part
                    linear_val -= b2
        except Exception as exc:
            print_func(f"Error in residual quadratic fit: {exc}")

    # ------------------------------------------------------------------
    # Secondary shelf linear fit: calculate and remove residual slope
    # directly from the shelves to ensure the final gradient is zero.
    # ------------------------------------------------------------------
    intensity_temp = intensity1 + linear_val * field_off + quad1 * (field_off - quad_offset_val) ** 2
    slopes_res = []
    if np.sum(sat_pos_all) >= 2:
        try:
            p_pos_res = np.polyfit(field_off[sat_pos_all], intensity_temp[sat_pos_all], 1)
            slopes_res.append(p_pos_res[0])
        except Exception:
            pass
    if np.sum(sat_neg_all) >= 2:
        try:
            p_neg_res = np.polyfit(field_off[sat_neg_all], intensity_temp[sat_neg_all], 1)
            slopes_res.append(p_neg_res[0])
        except Exception:
            pass
    if slopes_res:
        slope_res = float(np.mean(slopes_res))
        linear_val -= slope_res
        print_func(f"  [info] Secondary linear adjustment: subtracted residual slope {slope_res:.6f}")

    # ------------------------------------------------------------------
    # Pass 2 – second drift correction after all shape corrections
    # ------------------------------------------------------------------
    intensity3 = intensity1 + linear_val * field_off + quad1 * (field_off - quad_offset_val) ** 2
    drift2     = float((intensity3[0] - intensity3[-1]) / len(intensity3))

    final_drift = drift1 + drift2
    final_linear = linear_val
    final_quad = quad1
    final_quad_offset = quad_offset_val

    if not drift_corr:
        final_drift = 0.0
    if not linear_corr:
        final_linear = 0.0
    if not quad_corr:
        final_quad = 0.0
        final_quad_offset = 0.0

    return dict(drift=final_drift, linear=final_linear,
                quad=final_quad, quad_offset=final_quad_offset)


def apply_correction(field: np.ndarray, intensity: np.ndarray,
                     coeffs: dict, normalize: bool = True) -> np.ndarray:
    """
    Apply drift / linear / quadratic Faraday corrections and (optionally)
    normalize to [-1, 1].  Mirrors LoopCorrectionPanel.correct_intensity().
    """
    arr      = np.asarray(intensity, dtype=np.float32).copy()
    idx      = np.arange(len(arr))
    qo       = coeffs.get("quad_offset", 0.0)
    arr_corr = (arr
                + coeffs["drift"]  * (idx - idx.mean())
                + coeffs["linear"] * (field - np.mean(field))
                + coeffs["quad"]   * ((field - np.mean(field) - qo) ** 2))
    if normalize:
        ptp = np.ptp(arr_corr)
        if ptp > 0:
            arr_corr = (arr_corr - np.min(arr_corr)) / ptp * 2 - 1
        else:
            arr_corr[:] = 0.0
    return arr_corr


def compute_hc_hr(field: np.ndarray,
                  ycorr: np.ndarray) -> dict:
    """
    Compute coercive fields Hc+ / Hc- and remanence Hr, reporting Hr separately
    for each field-sweep branch.

    Hr_asc  = intensity at H≈0 on the ascending branch  (after negative sat.).
    Hr_desc = intensity at H≈0 on the descending branch (after positive sat.).

    Averaging Hr_asc and Hr_desc directly yields near-zero for a symmetric loop
    (one is positive, one negative), so they are stored and reported individually.
    The magnitude |Hr| = 0.5*(|Hr_asc| + |Hr_desc|) gives the squareness ratio.

    Returns a dict with keys:
        hc_pos, hc_neg, hc_avg,
        hr_asc, hr_desc, hr_abs,
        hr_fields, hr_vals  (legacy, for plot markers)
    """
    # Split into ascending and descending field branches
    i_min = int(np.argmin(field))
    i_max = int(np.argmax(field))

    if i_min < i_max:
        asc_idx  = np.arange(i_min, i_max + 1)
        desc_idx = np.concatenate([np.arange(i_max, len(field)),
                                   np.arange(0, i_min + 1)])
    else:
        desc_idx = np.arange(i_max, i_min + 1)
        asc_idx  = np.concatenate([np.arange(i_min, len(field)),
                                   np.arange(0, i_max + 1)])

    def sorted_branch(idxs):
        f_b = field[idxs]
        y_b = ycorr[idxs]
        order = np.argsort(f_b)
        return f_b[order], y_b[order]

    f_asc,  y_asc  = sorted_branch(asc_idx)
    f_desc, y_desc = sorted_branch(desc_idx)

    # Saturation-based midpoint
    field_off     = field - np.mean(field)
    field_abs_max = float(np.max(np.abs(field_off)))
    sat_mask_pos  = field_off >  0.8 * field_abs_max
    sat_mask_neg  = field_off < -0.8 * field_abs_max
    sat_pos = float(np.mean(ycorr[sat_mask_pos])) if np.any(sat_mask_pos) else float(ycorr[i_max])
    sat_neg = float(np.mean(ycorr[sat_mask_neg])) if np.any(sat_mask_neg) else float(ycorr[i_min])
    mid = 0.5 * (sat_pos + sat_neg)

    def find_crossings(f_branch, y_branch):
        crossings = []
        for i in range(len(f_branch) - 1):
            y0, y1 = y_branch[i], y_branch[i + 1]
            f0, f1 = f_branch[i], f_branch[i + 1]
            if (y0 - mid) * (y1 - mid) <= 0 and y0 != y1:
                frac    = (mid - y0) / (y1 - y0)
                f_cross = f0 + frac * (f1 - f0)
                crossings.append((f_cross, abs(y1 - y0)))
        return crossings

    crossings_asc  = find_crossings(f_asc,  y_asc)
    crossings_desc = find_crossings(f_desc, y_desc)

    hc_pos = None
    if crossings_asc:
        crossings_asc.sort(key=lambda x: x[1], reverse=True)
        hc_pos = crossings_asc[0][0]
    elif len(f_asc) > 0:
        hc_pos = float(f_asc[np.argmin(np.abs(y_asc - mid))])

    hc_neg = None
    if crossings_desc:
        crossings_desc.sort(key=lambda x: x[1], reverse=True)
        hc_neg = crossings_desc[0][0]
    elif len(f_desc) > 0:
        hc_neg = float(f_desc[np.argmin(np.abs(y_desc - mid))])

    hc_avg = None
    if hc_pos is not None and hc_neg is not None:
        magnitude = 0.5 * (abs(hc_pos) + abs(hc_neg))
        polarity = 1.0 if (hc_pos - hc_neg) >= 0 else -1.0
        hc_avg = polarity * magnitude

    # ------------------------------------------------------------------
    # Remanence: intensity at the field point nearest zero on EACH branch.
    # These have opposite signs for a symmetric loop; averaging them would
    # give ~0 which is physically meaningless.
    # ------------------------------------------------------------------
    iz_asc  = int(np.argmin(np.abs(f_asc)))
    iz_desc = int(np.argmin(np.abs(f_desc)))
    hr_asc  = float(y_asc[iz_asc])    # remanence after negative saturation
    hr_desc = float(y_desc[iz_desc])  # remanence after positive saturation
    hr_abs  = 0.5 * (abs(hr_asc) + abs(hr_desc))   # unsigned squareness ratio

    # Legacy arrays kept for plot markers (both H≈0 points, one per branch)
    hr_fields = np.array([f_asc[iz_asc], f_desc[iz_desc]])
    hr_vals   = np.array([hr_asc, hr_desc])

    return dict(hc_pos=hc_pos, hc_neg=hc_neg, hc_avg=hc_avg,
                hr_asc=hr_asc, hr_desc=hr_desc, hr_abs=hr_abs,
                hr_fields=hr_fields, hr_vals=hr_vals)


# ---------------------------------------------------------------------------
# Step-number extraction
# ---------------------------------------------------------------------------

_STEP_RE = re.compile(r"step\s*(\d+)", re.IGNORECASE)


def extract_step_number(dir_name: str):
    """
    Return the integer step number from a directory name, or None if not found.
    Examples that match: "step0_AD", "Step1", "run_step3_fine", "STEP10".
    """
    m = _STEP_RE.search(dir_name)
    return int(m.group(1)) if m else None


def discover_data_dirs(parent_dir: str) -> list:
    """
    Recursively scan parent_dir (up to a depth of 3 subfolders) to find all
    directories that contain a valid MOKE dataset (i.e. at least one .txt file).
    Returns a sorted list of absolute directory paths.
    """
    valid_dirs = []
    parent_dir = os.path.abspath(parent_dir)
    
    for root, dirs, files in os.walk(parent_dir):
        # Calculate current depth relative to parent_dir
        rel_path = os.path.relpath(root, parent_dir)
        if rel_path == ".":
            depth = 0
        else:
            depth = len(rel_path.split(os.sep))
            
        # Limit traversal depth to 3
        if depth > 3:
            dirs[:] = []
            continue
            
        # Ignore the "Analysis" directory
        if os.path.basename(root).lower() == "analysis":
            dirs[:] = []
            continue
            
        # Check if the directory has a valid MOKE txt data file
        if any(f.lower().endswith(".txt") for f in files):
            if load_txt_data(root) is not None:
                valid_dirs.append(root)
                # Do not descend further into subfolders of a valid data folder
                dirs[:] = []
                
    return sorted(valid_dirs)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def save_loop_plot(field: np.ndarray, ycorr: np.ndarray,
                   hc_hr: dict, title: str, save_path: str):
    """Save a hysteresis loop figure with Hc/Hr annotations."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(field, ycorr, "o-", color="#1F4E79", lw=1.5, markersize=3,
            label="MOKE intensity")

    mid_y = 0.5 * (max(ycorr) + min(ycorr)) if len(ycorr) > 0 else 0.0

    hc_pos    = hc_hr.get("hc_pos")
    hc_neg    = hc_hr.get("hc_neg")
    hr_asc    = hc_hr.get("hr_asc")
    hr_desc   = hc_hr.get("hr_desc")
    hr_fields = hc_hr.get("hr_fields", [])

    if hc_pos is not None:
        ax.plot(hc_pos, mid_y, "o", ms=8, mec="#2E7D32", mfc="white", mew=1.5,
                label=f"Hc+ = {hc_pos:.2f} mT")
        ax.axvline(x=hc_pos, color="#2E7D32", linestyle=":", alpha=0.7, lw=1.0)
    if hc_neg is not None:
        ax.plot(hc_neg, mid_y, "o", ms=8, mec="#C62828", mfc="white", mew=1.5,
                label=f"Hc- = {hc_neg:.2f} mT")
        ax.axvline(x=hc_neg, color="#C62828", linestyle=":", alpha=0.7, lw=1.0)

    # Plot Hr on ascending branch (after negative saturation)
    if hr_asc is not None and len(hr_fields) >= 1:
        ax.plot(hr_fields[0], hr_asc, "s", ms=8, mec="#E65100", mfc="white", mew=1.5,
                label=f"Hr(asc) = {hr_asc:.3f}")
        ax.axhline(y=hr_asc, color="#E65100", linestyle=":", alpha=0.5, lw=0.8)
    # Plot Hr on descending branch (after positive saturation)
    if hr_desc is not None and len(hr_fields) >= 2:
        ax.plot(hr_fields[1], hr_desc, "D", ms=7, mec="#BF360C", mfc="white", mew=1.5,
                label=f"Hr(desc) = {hr_desc:.3f}")
        ax.axhline(y=hr_desc, color="#BF360C", linestyle=":", alpha=0.5, lw=0.8)

    ax.set_xlabel("Field (mT)", fontsize=12)
    ax.set_ylabel("MOKE Intensity (Normalized)", fontsize=12)
    ax.set_title(title, fontsize=11)
    ax.grid(True, color="#e0e0e0", linestyle="--", linewidth=0.8)
    ax.legend(loc="best", fontsize=9, frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_loop_data(field: np.ndarray, raw_intens: np.ndarray,
                   ycorr: np.ndarray, hc_hr: dict,
                   coeffs: dict, dir_name: str, save_path: str):
    """Save corrected loop data and Hc/Hr values to a tab-separated text file."""
    hc_pos  = hc_hr.get("hc_pos")
    hc_neg  = hc_hr.get("hc_neg")
    hc_avg  = hc_hr.get("hc_avg")
    hr_asc  = hc_hr.get("hr_asc")
    hr_desc = hc_hr.get("hr_desc")
    hr_abs  = hc_hr.get("hr_abs")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"# KerrPyLooper Batch Processor – Loop Export\n")
        f.write(f"# Export Date  : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"# Source Dir   : {dir_name}\n")
        f.write(f"#\n")
        f.write(f"# --- Correction Coefficients ---\n")
        f.write(f"# Drift             : {coeffs['drift']:.6e}\n")
        f.write(f"# Linear Faraday    : {coeffs['linear']:.6e}\n")
        f.write(f"# Quadratic Faraday : {coeffs['quad']:.6e}\n")
        f.write(f"# Quad Field Offset : {coeffs['quad_offset']:.4f} mT\n")
        f.write(f"# Normalized        : True\n")
        f.write(f"#\n")
        f.write(f"# --- Magnetic Parameters ---\n")
        f.write(f"# Hc+  (mT) : {hc_pos:.4f}\n"  if hc_pos  is not None else "# Hc+  (mT) : n/a\n")
        f.write(f"# Hc-  (mT) : {hc_neg:.4f}\n"  if hc_neg  is not None else "# Hc-  (mT) : n/a\n")
        f.write(f"# Hc   (mT) : {hc_avg:.4f}\n"  if hc_avg  is not None else "# Hc   (mT) : n/a\n")
        f.write(f"# Hr_asc    : {hr_asc:.6f}\n"   if hr_asc  is not None else "# Hr_asc    : n/a\n")
        f.write(f"# Hr_desc   : {hr_desc:.6f}\n"  if hr_desc is not None else "# Hr_desc   : n/a\n")
        f.write(f"# |Hr|      : {hr_abs:.6f}\n"   if hr_abs  is not None else "# |Hr|      : n/a\n")
        f.write(f"#\n")
        f.write("Field_mT\tRaw_Intensity\tCorrected_Intensity\n")
        for fd, ri, ci in zip(field, raw_intens, ycorr):
            f.write(f"{fd:.6f}\t{ri:.6f}\t{ci:.6f}\n")


def save_summary_plots(results: list, analysis_dir: str):
    """
    Given a list of result dicts (each must contain 'step', 'hc_avg', 'hr_abs',
    'dir_name'), create Hc-vs-step and Hr-vs-step plots and a summary CSV.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    # Filter to only those with a detected step number
    stepped = [r for r in results if r.get("step") is not None]
    if not stepped:
        return

    stepped.sort(key=lambda r: r["step"])
    steps = [r["step"] for r in stepped]

    hc_vals  = [r.get("hc_avg") for r in stepped]
    hr_vals  = [r.get("hr_abs") for r in stepped]   # unsigned squareness ratio

    # --- Hc vs. step ---
    hc_steps_valid = [(s, v) for s, v in zip(steps, hc_vals) if v is not None]
    if hc_steps_valid:
        xs, ys = zip(*hc_steps_valid)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, ys, "o-", color="#1F4E79", lw=1.8, markersize=7)
        ax.set_xlabel("Step Number", fontsize=12)
        ax.set_ylabel("Coercive Field Hc (mT)", fontsize=12)
        ax.set_title("Coercive Field vs. Step", fontsize=12)
        ax.grid(True, color="#e0e0e0", linestyle="--", linewidth=0.8)
        ax.set_xticks(xs)
        fig.tight_layout()
        fig.savefig(os.path.join(analysis_dir, "Hc_vs_step.png"), dpi=200)
        plt.close(fig)
        print(f"  [ok] Saved Hc_vs_step.png")

    # --- |Hr| (squareness) vs. step ---
    hr_steps_valid = [(s, v) for s, v in zip(steps, hr_vals) if v is not None]
    if hr_steps_valid:
        xs, ys = zip(*hr_steps_valid)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, ys, "s-", color="#E65100", lw=1.8, markersize=7)
        ax.set_xlabel("Step Number", fontsize=12)
        ax.set_ylabel("|Hr| – Squareness ratio", fontsize=12)
        ax.set_title("Remanence |Hr| vs. Step", fontsize=12)
        ax.set_ylim(bottom=0)
        ax.grid(True, color="#e0e0e0", linestyle="--", linewidth=0.8)
        ax.set_xticks(xs)
        fig.tight_layout()
        fig.savefig(os.path.join(analysis_dir, "Hr_vs_step.png"), dpi=200)
        plt.close(fig)
        print(f"  [ok] Saved Hr_vs_step.png")

    # --- Per-branch Hr↑ / Hr↓ vs. step ---
    hr_a = [r.get("hr_asc")  for r in stepped]
    hr_d = [r.get("hr_desc") for r in stepped]
    if any(v is not None for v in hr_a) or any(v is not None for v in hr_d):
        fig, ax = plt.subplots(figsize=(6, 4))
        if any(v is not None for v in hr_a):
            ax.plot(steps, hr_a, "s-", color="#E65100", lw=1.8, markersize=7, label="Hr(asc)")
        if any(v is not None for v in hr_d):
            ax.plot(steps, hr_d, "D-", color="#BF360C", lw=1.8, markersize=7, label="Hr(desc)")
        ax.set_xlabel("Step Number", fontsize=12)
        ax.set_ylabel("Remanence Hr (normalized)", fontsize=12)
        ax.set_title("Per-branch Remanence vs. Step", fontsize=12)
        ax.grid(True, color="#e0e0e0", linestyle="--", linewidth=0.8)
        ax.set_xticks(steps)
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(analysis_dir, "Hr_branches_vs_step.png"), dpi=200)
        plt.close(fig)
        print(f"  [ok] Saved Hr_branches_vs_step.png")

    # --- Summary CSV ---
    rows = []
    for r in stepped:
        rows.append({
            "Step":      r["step"],
            "Directory": r["dir_name"],
            "Hc+ (mT)":  f"{r['hc_pos']:.4f}"  if r.get("hc_pos")  is not None else "n/a",
            "Hc- (mT)":  f"{r['hc_neg']:.4f}"  if r.get("hc_neg")  is not None else "n/a",
            "Hc (mT)":   f"{r['hc_avg']:.4f}"  if r.get("hc_avg")  is not None else "n/a",
            "Hr_asc":     f"{r['hr_asc']:.6f}"  if r.get("hr_asc")  is not None else "n/a",
            "Hr_desc":    f"{r['hr_desc']:.6f}" if r.get("hr_desc") is not None else "n/a",
            "|Hr|": f"{r['hr_abs']:.6f}" if r.get("hr_abs") is not None else "n/a",
        })
    df_summary = pd.DataFrame(rows)
    txt_path = os.path.join(analysis_dir, "summary.txt")
    df_summary.to_csv(txt_path, sep="\t", index=False)
    print(f"  [ok] Saved summary.txt")



# ---------------------------------------------------------------------------
# Full results table (all directories, step or not)
# ---------------------------------------------------------------------------

def save_full_summary(results: list, analysis_dir: str):
    """
    Save a single tab-separated text file for ALL processed directories
    (including those without step numbers).
    """
    import pandas as pd
    rows = []
    for r in results:
        rows.append({
            "Directory": r["dir_name"],
            "Step":      r.get("step", ""),
            "Hc+ (mT)":  f"{r['hc_pos']:.4f}"  if r.get("hc_pos")  is not None else "n/a",
            "Hc- (mT)":  f"{r['hc_neg']:.4f}"  if r.get("hc_neg")  is not None else "n/a",
            "Hc (mT)":   f"{r['hc_avg']:.4f}"  if r.get("hc_avg")  is not None else "n/a",
            "Hr_asc":     f"{r['hr_asc']:.6f}"  if r.get("hr_asc")  is not None else "n/a",
            "Hr_desc":    f"{r['hr_desc']:.6f}" if r.get("hr_desc") is not None else "n/a",
            "|Hr|":       f"{r['hr_abs']:.6f}"  if r.get("hr_abs")  is not None else "n/a",
        })
    df = pd.DataFrame(rows)
    txt_path = os.path.join(analysis_dir, "all_results.txt")
    df.to_csv(txt_path, sep="\t", index=False)
    print(f"  [ok] Saved all_results.txt")


# ---------------------------------------------------------------------------
# Main batch routine
# ---------------------------------------------------------------------------

def process_directory(data_dir: str, analysis_dir: str, dir_name: str,
                      drift_corr: bool = True, linear_corr: bool = True,
                      quad_corr: bool = True,
                      print_func=print,
                      cancel_event=None) -> dict | None:
    """
    Process one data sub-directory.  Returns a result dict on success,
    or None on failure (missing data / images).
    """
    print_func(f"\nProcessing: {dir_name}")

    # ---- Load .txt data file ----
    txt_df = load_txt_data(data_dir, print_func=print_func)
    if txt_df is None:
        print_func(f"  [skip] No valid .txt data file found.")
        return None

    field = txt_df["Field"].to_numpy(dtype=np.float32)

    # ---- Collect image files ----
    image_files = sorted([
        f for f in os.listdir(data_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))
    ])

    # ---- Select background image ----
    # Check if mask.png exists in data_dir first (GUI behavior)
    all_files = os.listdir(data_dir)
    bg_name = next((f for f in all_files if f.lower() == "mask.png"), None)

    # Filter image_files to only those listed in the .txt file
    valid_files = set(txt_df["File"].str.strip().tolist())
    image_files = [f for f in image_files if f in valid_files]

    if bg_name is None:
        bg_name = find_background_image(data_dir, image_files)

    if bg_name is None:
        print_func(f"  [skip] Could not find a background image.")
        return None
    bg_path = os.path.join(data_dir, bg_name)
    try:
        background_array = pil_to_numpy_fast(Image.open(bg_path))
        print_func(f"  Background: {bg_name}")
    except Exception as exc:
        print_func(f"  [skip] Failed to load background image {bg_name}: {exc}")
        return None

    # ---- Image subtraction loop ----
    raw_intens = run_subtraction_loop(data_dir, txt_df, background_array, print_func=print_func, cancel_event=cancel_event)
    if np.all(np.isnan(raw_intens)):
        print_func(f"  [skip] All image subtractions failed (all NaN).")
        return None

    # Replace NaNs with linear interpolation so corrections don't crash
    nan_mask = np.isnan(raw_intens)
    if np.any(nan_mask):
        xp = np.where(~nan_mask)[0]
        fp = raw_intens[~nan_mask]
        raw_intens[nan_mask] = np.interp(np.where(nan_mask)[0], xp, fp)

    # ---- Auto-correction ----
    coeffs = auto_correct_coeffs(field, raw_intens, drift_corr=drift_corr,
                                 linear_corr=linear_corr, quad_corr=quad_corr,
                                 print_func=print_func)
    ycorr  = apply_correction(field, raw_intens, coeffs, normalize=True)
    print_func(f"  Auto-correct: drift={coeffs['drift']:.4f}, "
          f"linear={coeffs['linear']:.4f}, quad={coeffs['quad']:.6f}, "
          f"quad_offset={coeffs['quad_offset']:.2f} mT")

    # ---- Hc / Hr calculation ----
    hc_hr  = compute_hc_hr(field, ycorr)
    hc_pos = hc_hr.get("hc_pos")
    hc_neg = hc_hr.get("hc_neg")
    hc_avg = hc_hr.get("hc_avg")
    hr_asc  = hc_hr.get("hr_asc")
    hr_desc = hc_hr.get("hr_desc")
    hr_abs  = hc_hr.get("hr_abs")

    def _fmt(v):
        return f"{v:.4f}" if v is not None else "n/a"

    print_func(f"  Hc+ = {_fmt(hc_pos)} mT, Hc- = {_fmt(hc_neg)} mT, Hc = {_fmt(hc_avg)} mT")
    print_func(f"  Hr(asc) = {_fmt(hr_asc)}, Hr(desc) = {_fmt(hr_desc)}, |Hr| = {_fmt(hr_abs)}")

    # ---- Step detection ----
    step = extract_step_number(dir_name)
    if step is not None:
        print_func(f"  Step number detected: {step}")

    return dict(
        dir_name=dir_name,
        step=step,
        hc_pos=hc_pos,
        hc_neg=hc_neg,
        hc_avg=hc_avg,
        hr_asc=hr_asc,
        hr_desc=hr_desc,
        hr_abs=hr_abs,
        field=field,
        raw_intens=raw_intens,
        ycorr=ycorr,
        hc_hr=hc_hr,
        coeffs=coeffs,
    )


def _pool_initializer():
    """
    One-time initializer for each worker process in the pool.
    Pre-imports heavy modules so the first task doesn't pay import costs.
    Called automatically by Pool(initializer=...).
    """
    import matplotlib
    matplotlib.use("Agg")
    # Force numpy, PIL to be fully loaded in this process
    import numpy as np  # noqa: F401
    from PIL import Image  # noqa: F401


def process_directory_worker(args):
    """
    Worker function for parallel pools.
    args: tuple of (data_dir, analysis_dir, dir_name, drift, linear, quad, cancel_event)
    """
    import time as _time
    import os as _os

    data_dir, analysis_dir, dir_name, drift, linear, quad, cancel_event = args
    log_lines = []
    worker_pid = _os.getpid()
    t_start = _time.time()
    
    def buffered_print(*msg_args, **kwargs):
        sep = kwargs.get('sep', ' ')
        msg_str = sep.join(map(str, msg_args))
        log_lines.append(msg_str)

    buffered_print(f"  [worker PID {worker_pid}] Started at {_time.strftime('%H:%M:%S')}")
        
    try:
        result = process_directory(
            data_dir, analysis_dir, dir_name,
            drift_corr=drift, linear_corr=linear, quad_corr=quad,
            print_func=buffered_print, cancel_event=cancel_event
        )
        elapsed = _time.time() - t_start
        buffered_print(f"  [worker PID {worker_pid}] Finished in {elapsed:.1f}s")
        return result, log_lines, None
    except Exception as exc:
        import traceback
        elapsed = _time.time() - t_start
        err_str = f"Error processing {dir_name} (PID {worker_pid}, {elapsed:.1f}s): {exc}\n{traceback.format_exc()}"
        return None, log_lines, err_str



def _run_batch_processes(sub_dirs, analysis_dir, drift_corr, linear_corr, quad_corr, max_workers, cancel_event, thread_ref, results, total_dirs, pool=None):
    import multiprocessing
    import time

    # Prepare arguments (cancel_event is None for processes, we use pool.terminate())
    parent_dir = os.path.dirname(analysis_dir)
    tasks = []
    for d in sub_dirs:
        rel_path = os.path.relpath(d, parent_dir)
        dir_name = rel_path.replace(os.sep, "_")
        tasks.append((d, analysis_dir, dir_name, drift_corr, linear_corr, quad_corr, None))

    # Use pre-warmed pool if provided, otherwise create a new one
    own_pool = pool is None
    if own_pool:
        num_procs = max_workers if max_workers else max(1, multiprocessing.cpu_count() - 1)
        print(f"\n[info] Spinning up background processes...")
        print(f"Starting Process Pool with {num_procs} workers...")
        pool = multiprocessing.Pool(processes=num_procs, initializer=_pool_initializer)
    else:
        print(f"[info] Using pre-warmed process pool (workers already initialized).")

    if thread_ref is not None:
        thread_ref.active_pool = pool

    try:
        pending = []
        for i, task in enumerate(tasks):
            dir_name = task[2]
            print(f"[{i+1}/{total_dirs}] Queued: {dir_name}")
            res = pool.apply_async(process_directory_worker, (task,))
            pending.append((dir_name, res))

        completed = 0
        while pending:
            if cancel_event is not None and cancel_event.is_set():
                print("\n[info] Cancellation detected in run_batch. Terminating process pool...")
                pool.terminate()
                pool.join()
                return

            still_pending = []
            for dir_name, res in pending:
                if res.ready():
                    completed += 1
                    try:
                        result, log_lines, err_str = res.get()
                        print(f"\n--- Log for {dir_name} [{completed}/{total_dirs}] ---")
                        for line in log_lines:
                            print(line)
                        if err_str:
                            print(f"[ERROR] {err_str}")
                        if result is not None:
                            results.append(result)
                    except Exception as exc:
                        print(f"\n[ERROR] Task for {dir_name} raised exception: {exc}")
                else:
                    still_pending.append((dir_name, res))
            pending = still_pending
            time.sleep(0.05)

        pool.close()
        pool.join()
    except Exception as e:
        pool.terminate()
        pool.join()
        raise e



def _run_batch_threads(sub_dirs, analysis_dir, drift_corr, linear_corr, quad_corr, max_workers, cancel_event, thread_ref, results, total_dirs):
    import concurrent.futures
    import time

    parent_dir = os.path.dirname(analysis_dir)
    tasks = []
    for d in sub_dirs:
        rel_path = os.path.relpath(d, parent_dir)
        dir_name = rel_path.replace(os.sep, "_")
        tasks.append((d, analysis_dir, dir_name, drift_corr, linear_corr, quad_corr, cancel_event))

    num_threads = max_workers if max_workers else max(1, os.cpu_count() - 1)
    print(f"Starting Thread Pool with {num_threads} workers...")

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)
    if thread_ref is not None:
        thread_ref.active_executor = executor

    try:
        futures = {}
        for i, task in enumerate(tasks):
            dir_name = task[2]
            print(f"[{i+1}/{total_dirs}] Queued: {dir_name}")
            f = executor.submit(process_directory_worker, task)
            futures[f] = dir_name

        completed = 0
        pending = set(futures.keys())
        running_reported = set()

        while pending:
            if cancel_event is not None and cancel_event.is_set():
                print("\n[info] Cancellation detected. Shutting down thread pool...")
                executor.shutdown(wait=False, cancel_futures=True)
                return

            # Check for tasks that just started running to provide immediate visual feedback
            for f in pending:
                if f.running() and futures[f] not in running_reported:
                    print(f"  -> Started processing: {futures[f]}")
                    running_reported.add(futures[f])

            # Wait for any task to finish, up to 0.2 seconds
            done, pending = concurrent.futures.wait(
                pending, timeout=0.2, return_when=concurrent.futures.FIRST_COMPLETED
            )

            for f in done:
                dir_name = futures[f]
                completed += 1
                try:
                    result, log_lines, err_str = f.result()
                    print(f"\n--- Log for {dir_name} [{completed}/{total_dirs}] ---")
                    for line in log_lines:
                        print(line)
                    if err_str:
                        print(f"[ERROR] {err_str}")
                    if result is not None:
                        results.append(result)
                except Exception as exc:
                    print(f"\n[ERROR] Task for {dir_name} raised exception: {exc}")

        executor.shutdown(wait=True)
    except Exception as e:
        executor.shutdown(wait=False, cancel_futures=True)
        raise e



def _run_batch_sequential(sub_dirs, analysis_dir, drift_corr, linear_corr, quad_corr, cancel_event, results, total_dirs):
    print("Starting Sequential execution...")
    parent_dir = os.path.dirname(analysis_dir)
    for i, data_dir in enumerate(sub_dirs):
        if cancel_event is not None and cancel_event.is_set():
            print("\n[info] Cancellation detected. Halting sequential batch...")
            return
        rel_path = os.path.relpath(data_dir, parent_dir)
        dir_name = rel_path.replace(os.sep, "_")
        print(f"[{i+1}/{total_dirs}] Processing: {dir_name}")

        task = (data_dir, analysis_dir, dir_name, drift_corr, linear_corr, quad_corr, cancel_event)
        result, log_lines, err_str = process_directory_worker(task)

        print(f"\n--- Log for {dir_name} [{i+1}/{total_dirs}] ---")
        for line in log_lines:
            print(line)
        if err_str:
            print(f"[ERROR] {err_str}")
        if result is not None:
            results.append(result)


def run_batch(parent_dir: str, drift_corr: bool = True, linear_corr: bool = True,
              quad_corr: bool = True, mode="processes", max_workers=None,
              cancel_event=None, thread_ref=None):
    """
    Top-level batch routine. Scans *parent_dir* for data sub-directories,
    processes them (sequentially, via thread pool, or via process pool),
    and saves results + summary to an 'Analysis' sub-folder.
    """
    parent_dir = os.path.abspath(parent_dir)
    print(f"\n{'='*60}")
    print(f"Batch processing ({mode} mode): {parent_dir}")
    if max_workers:
        print(f"Workers: {max_workers}")
    print(f"{'='*60}")

    # ---- Pre-warm process pool while discovering directories ----
    # On Windows (spawn), pool creation imports numpy/PIL/matplotlib in each
    # worker — this takes 5-10s. By starting the pool BEFORE the network
    # directory scan, the two costs overlap instead of stacking.
    pool = None
    if mode == "processes":
        import multiprocessing as _mp
        num_procs = max_workers if max_workers else max(1, _mp.cpu_count() - 1)
        print(f"\n[info] Pre-warming process pool ({num_procs} workers)...")
        t_pool = time.time()
        pool = _mp.Pool(processes=num_procs, initializer=_pool_initializer)
        if thread_ref is not None:
            thread_ref.active_pool = pool
        print(f"[info] Pool created in {time.time() - t_pool:.1f}s (workers importing modules in background)")

    # ---- Discover sub-directories ----
    print(f"\n[info] Scanning for data directories...")
    t_disc = time.time()
    sub_dirs = discover_data_dirs(parent_dir)
    print(f"[info] Discovery complete: found {len(sub_dirs)} directories in {time.time() - t_disc:.1f}s")

    if not sub_dirs:
        print("[error] No sub-directories found in the selected folder.")
        if pool is not None:
            pool.terminate()
            pool.join()
        return

    # ---- Create Analysis output folder ----
    analysis_dir = os.path.join(parent_dir, "Analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    print(f"Output folder: {analysis_dir}\n")

    results = []
    total_dirs = len(sub_dirs)
    start_time = time.time()

    if mode == "processes":
        _run_batch_processes(sub_dirs, analysis_dir, drift_corr, linear_corr, quad_corr, max_workers, cancel_event, thread_ref, results, total_dirs, pool=pool)
    elif mode == "threads":
        _run_batch_threads(sub_dirs, analysis_dir, drift_corr, linear_corr, quad_corr, max_workers, cancel_event, thread_ref, results, total_dirs)
    else:  # sequential mode
        _run_batch_sequential(sub_dirs, analysis_dir, drift_corr, linear_corr, quad_corr, cancel_event, results, total_dirs)

    # ---- Sequential Post-Processing (Saving plots & data files) ----
    if results:
        print(f"\nSaving individual loop plots and text files to {analysis_dir}...")
        for r in results:
            dir_name = r["dir_name"]
            field = r["field"]
            raw_intens = r["raw_intens"]
            ycorr = r["ycorr"]
            hc_hr = r["hc_hr"]
            coeffs = r["coeffs"]
            
            safe_name = re.sub(r"[^\w\-]", "_", dir_name)
            loop_img_path  = os.path.join(analysis_dir, f"{safe_name}_loop.png")
            loop_data_path = os.path.join(analysis_dir, f"{safe_name}_loop.txt")
            
            try:
                save_loop_plot(field, ycorr, hc_hr, title=dir_name, save_path=loop_img_path)
                save_loop_data(field, raw_intens, ycorr, hc_hr, coeffs,
                               dir_name=dir_name, save_path=loop_data_path)
                print(f"  [ok] Saved {os.path.basename(loop_img_path)} and {os.path.basename(loop_data_path)}")
            except Exception as exc:
                print(f"  [ERROR] Failed to save output files for {dir_name}: {exc}")

        # ---- Compile Summary Files ----
        print("\nCompiling summary charts and tables...")
        try:
            save_summary_plots(results, analysis_dir)
            save_full_summary(results, analysis_dir)
        except Exception as exc:
            print(f"  [ERROR] Failed to save summary outputs: {exc}")

    elapsed_total = time.time() - start_time
    print(f"\nAll processing tasks completed in {elapsed_total:.2f} seconds.")


# ---------------------------------------------------------------------------
# GUI Components (only active if PyQt5 is installed)
# ---------------------------------------------------------------------------

class StdoutRedirector(object):
    def __init__(self, signal, original_stream, only_thread=False):
        self.signal = signal
        self.original_stream = original_stream
        self.only_thread = only_thread

    def write(self, text):
        if self.only_thread:
            try:
                from PyQt5.QtCore import QThread
                current = QThread.currentThread()
                if current and current.objectName() == "BatchProcessorThread":
                    self.signal.emit(text)
                    return
            except Exception:
                pass
            if self.original_stream:
                self.original_stream.write(text)
        else:
            self.signal.emit(text)

    def flush(self):
        if self.original_stream:
            self.original_stream.flush()


if HAS_PYQT5:
    class ProcessingThread(QThread):
        finished_signal = pyqtSignal(bool)

        def __init__(self, parent_dir, drift, linear, quad, mode="processes", max_workers=4):
            super().__init__()
            self.setObjectName("BatchProcessorThread")
            self.parent_dir = parent_dir
            self.drift = drift
            self.linear = linear
            self.quad = quad
            self.mode = mode
            self.max_workers = max_workers
            self.cancel_event = threading.Event()
            self.active_pool = None
            self.active_executor = None

        def run(self):
            try:
                run_batch(
                    self.parent_dir,
                    drift_corr=self.drift,
                    linear_corr=self.linear,
                    quad_corr=self.quad,
                    mode=self.mode,
                    max_workers=self.max_workers,
                    cancel_event=self.cancel_event,
                    thread_ref=self
                )
                if self.cancel_event.is_set():
                    self.finished_signal.emit(False)
                else:
                    self.finished_signal.emit(True)
            except Exception as e:
                print(f"\n[ERROR] Batch processing failed: {e}")
                self.finished_signal.emit(False)


    class BatchProcessorGUI(QWidget):
        append_text = pyqtSignal(str)

        def __init__(self, parent_dir=None, theme="dark", parent=None):
            super().__init__(None)
            self.parent_launcher = parent
            self.parent_dir = parent_dir or ""
            self.theme = theme
            self.thread = None
            
            # Save original stdout/stderr
            self.old_stdout = sys.stdout
            self.old_stderr = sys.stderr
            
            self.init_ui()
            
            # Connect the redirection signal
            self.append_text.connect(self.on_append_text)
            
            # If parent is provided (we are embedded), redirect only from BatchProcessorThread.
            only_thread = (parent is not None)
            sys.stdout = StdoutRedirector(self.append_text, sys.stdout, only_thread=only_thread)
            sys.stderr = StdoutRedirector(self.append_text, sys.stderr, only_thread=only_thread)

        def init_ui(self):
            self.setObjectName("MainBg")
            self.setWindowTitle("Batch Hysteresis Loop Processor")
            self.resize(750, 680)

            # Main layout
            layout = QVBoxLayout(self)
            layout.setContentsMargins(20, 20, 20, 20)
            layout.setSpacing(15)

            # Title Header
            title_layout = QVBoxLayout()
            title_layout.setSpacing(4)
            lbl_title = QLabel("Batch Loop Processor")
            lbl_title.setObjectName("SuiteTitle")
            title_layout.addWidget(lbl_title)
            
            lbl_subtitle = QLabel("Automate drift and Faraday corrections for multi-sweep datasets")
            lbl_subtitle.setObjectName("SuiteSubtitle")
            title_layout.addWidget(lbl_subtitle)
            layout.addLayout(title_layout)

            # 1. Target Directory GroupBox
            group_dir = QGroupBox("Target Directory")
            layout_dir = QHBoxLayout(group_dir)
            layout_dir.setContentsMargins(15, 15, 15, 15)
            layout_dir.setSpacing(10)

            self.txt_dir = QLineEdit(self.parent_dir)
            self.txt_dir.setPlaceholderText("Select parent directory containing sweep folders...")
            layout_dir.addWidget(self.txt_dir)

            btn_browse = QPushButton("Browse...")
            btn_browse.setCursor(Qt.PointingHandCursor)
            btn_browse.clicked.connect(self.on_browse)
            layout_dir.addWidget(btn_browse)
            layout.addWidget(group_dir)

            # 2. Correction Settings GroupBox
            group_corr = QGroupBox("Correction Settings")
            layout_corr = QHBoxLayout(group_corr)
            layout_corr.setContentsMargins(15, 15, 15, 15)
            layout_corr.setSpacing(20)

            self.chk_drift = QCheckBox("Drift Correction")
            self.chk_drift.setChecked(True)
            layout_corr.addWidget(self.chk_drift)

            self.chk_linear = QCheckBox("Linear Faraday Correction")
            self.chk_linear.setChecked(True)
            layout_corr.addWidget(self.chk_linear)

            self.chk_quad = QCheckBox("Quadratic Faraday Correction")
            self.chk_quad.setChecked(True)
            layout_corr.addWidget(self.chk_quad)
            layout.addWidget(group_corr)

            # 2.5. Parallel Settings GroupBox
            group_parallel = QGroupBox("Parallel Processing Settings")
            layout_parallel = QHBoxLayout(group_parallel)
            layout_parallel.setContentsMargins(15, 15, 15, 15)
            layout_parallel.setSpacing(20)

            lbl_mode = QLabel("Execution Mode:")
            layout_parallel.addWidget(lbl_mode)

            self.cmb_mode = QComboBox()
            self.cmb_mode.addItems([
                "Sequential",
                "Multi-threaded (GIL-bound)",
                "Multi-processed (Fastest)"
            ])
            self.cmb_mode.setCurrentIndex(2)  # Default to Multi-processed
            layout_parallel.addWidget(self.cmb_mode)

            lbl_workers = QLabel("Max CPU Workers:")
            layout_parallel.addWidget(lbl_workers)

            self.spin_workers = QSpinBox()
            self.spin_workers.setRange(1, os.cpu_count() or 4)
            default_workers = max(1, (os.cpu_count() or 4) - 1)
            self.spin_workers.setValue(default_workers)
            layout_parallel.addWidget(self.spin_workers)
            
            self.cmb_mode.currentIndexChanged.connect(self.on_mode_changed)
            layout.addWidget(group_parallel)

            # 3. Actions Row
            layout_actions = QHBoxLayout()
            
            self.btn_run = QPushButton("Run Batch")
            self.btn_run.setObjectName("LaunchButton")
            self.btn_run.setCursor(Qt.PointingHandCursor)
            self.btn_run.clicked.connect(self.on_run_batch)
            layout_actions.addWidget(self.btn_run)

            self.btn_stop = QPushButton("Stop Process")
            self.btn_stop.setObjectName("StopButton")
            self.btn_stop.setCursor(Qt.PointingHandCursor)
            self.btn_stop.setEnabled(False)
            self.btn_stop.clicked.connect(self.on_stop_batch)
            layout_actions.addWidget(self.btn_stop)
            
            layout_actions.addStretch()
            
            btn_clear = QPushButton("Clear Log")
            btn_clear.setCursor(Qt.PointingHandCursor)
            btn_clear.clicked.connect(self.on_clear_log)
            layout_actions.addWidget(btn_clear)
            layout.addLayout(layout_actions)

            # 4. Monospace Console Output
            self.txt_console = QTextEdit()
            self.txt_console.setObjectName("ConsoleOutput")
            self.txt_console.setReadOnly(True)
            layout.addWidget(self.txt_console)

            # Apply QSS Styling
            apply_theme(self, self.theme)
            self.on_mode_changed(self.cmb_mode.currentIndex())

        def on_mode_changed(self, index):
            if index == 0:  # Sequential
                self.spin_workers.setEnabled(False)
            else:
                self.spin_workers.setEnabled(True)

        def on_browse(self):
            start_dir = self.txt_dir.text() or os.path.dirname(os.path.abspath(__file__))
            selected_dir = QFileDialog.getExistingDirectory(self, "Select Parent Directory", start_dir)
            if selected_dir:
                self.txt_dir.setText(selected_dir)

        def on_run_batch(self):
            parent_dir = self.txt_dir.text().strip()
            if not parent_dir:
                self.txt_console.append("[error] Please select a target directory.\n")
                return
            if not os.path.isdir(parent_dir):
                self.txt_console.append(f"[error] Not a directory: {parent_dir}\n")
                return

            self.btn_run.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.txt_dir.setEnabled(False)
            self.chk_drift.setEnabled(False)
            self.chk_linear.setEnabled(False)
            self.chk_quad.setEnabled(False)
            self.cmb_mode.setEnabled(False)
            self.spin_workers.setEnabled(False)

            mode_idx = self.cmb_mode.currentIndex()
            if mode_idx == 0:
                mode = "sequential"
            elif mode_idx == 1:
                mode = "threads"
            else:
                mode = "processes"

            max_workers = self.spin_workers.value()

            self.thread = ProcessingThread(
                parent_dir,
                drift=self.chk_drift.isChecked(),
                linear=self.chk_linear.isChecked(),
                quad=self.chk_quad.isChecked(),
                mode=mode,
                max_workers=max_workers
            )
            self.thread.finished_signal.connect(self.on_processing_finished)
            self.thread.start()

        def on_stop_batch(self):
            if self.thread and self.thread.isRunning():
                self.txt_console.append("\n[info] Stopping batch process...")
                self.thread.cancel_event.set()
                if self.thread.active_pool:
                    try:
                        self.thread.active_pool.terminate()
                    except Exception:
                        pass
                if self.thread.active_executor:
                    try:
                        self.thread.active_executor.shutdown(wait=False, cancel_futures=True)
                    except Exception:
                        pass
                self.btn_stop.setEnabled(False)
                # Thread will gracefully exit and trigger on_processing_finished on its own.

        def on_clear_log(self):
            self.txt_console.clear()

        def on_append_text(self, text):
            self.txt_console.moveCursor(QTextCursor.End)
            self.txt_console.insertPlainText(text)
            self.txt_console.moveCursor(QTextCursor.End)

        def on_processing_finished(self, success):
            self.btn_run.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.txt_dir.setEnabled(True)
            self.chk_drift.setEnabled(True)
            self.chk_linear.setEnabled(True)
            self.chk_quad.setEnabled(True)
            self.cmb_mode.setEnabled(True)
            self.on_mode_changed(self.cmb_mode.currentIndex())
            if self.thread is not None:
                self.thread.deleteLater()
                self.thread = None

        def closeEvent(self, event):
            if self.thread and self.thread.isRunning():
                try:
                    self.thread.finished_signal.disconnect()
                except Exception:
                    pass
                self.thread.cancel_event.set()
                if self.thread.active_pool:
                    try:
                        self.thread.active_pool.terminate()
                    except Exception:
                        pass
                if self.thread.active_executor:
                    try:
                        self.thread.active_executor.shutdown(wait=False, cancel_futures=True)
                    except Exception:
                        pass
                self.thread.wait()
            sys.stdout = self.old_stdout
            sys.stderr = self.old_stderr
            event.accept()

        def change_theme(self, theme):
            self.theme = theme
            from gui_styles import apply_theme
            apply_theme(self, theme)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch MOKE hysteresis loop processor.")
    parser.add_argument(
        "parent_dir", nargs="?", default=None,
        help="Parent directory containing data sub-directories.  "
             "If omitted, a folder-picker dialog appears.")
    parser.add_argument(
        "--headless", action="store_true",
        help="Run in headless CLI mode without starting PyQt5 GUI.")
    parser.add_argument(
        "--theme", type=str, default="dark", choices=["dark", "charcoal", "light"],
        help="Theme to apply (dark, charcoal, or light). Only effective in GUI mode.")
    parser.add_argument(
        "--mode", type=str, default="processes", choices=["sequential", "threads", "processes"],
        help="Parallel processing mode (default: processes).")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Maximum number of parallel workers. Defaults to CPU count - 1.")
    args = parser.parse_args()

    parent_dir = args.parent_dir
    headless = args.headless or not HAS_PYQT5

    if not HAS_PYQT5 and not args.headless:
        print("[info] PyQt5 is not installed. Falling back to headless CLI mode.")

    if headless:
        # Run CLI mode
        import matplotlib
        matplotlib.use("Agg")
        if parent_dir is None:
            # Fall back to tkinter folder picker (no PyQt5 needed here)
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                root.lift()
                parent_dir = filedialog.askdirectory(
                    title="Select parent directory containing data sub-folders")
                root.destroy()
            except Exception:
                print("[error] No directory supplied and tkinter is unavailable.")
                sys.exit(1)

        if not parent_dir:
            print("[error] No directory selected.")
            sys.exit(1)

        if not os.path.isdir(parent_dir):
            print(f"[error] Not a directory: {parent_dir}")
            sys.exit(1)

        run_batch(parent_dir, mode=args.mode, max_workers=args.workers)
    else:
        # Run GUI mode
        # Enable high DPI scaling if supported (must be set before QApplication creation)
        if hasattr(Qt, 'AA_EnableHighDpiScaling'):
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
            QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

        app = QApplication(sys.argv)
        gui = BatchProcessorGUI(parent_dir=parent_dir, theme=args.theme)
        gui.show()
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()
