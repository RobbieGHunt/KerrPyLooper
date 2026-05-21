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

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt

# Add script directory to sys.path to ensure gui_styles import works
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
        QLabel, QLineEdit, QCheckBox, QFileDialog, QTextEdit, QGroupBox
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

def crop600(arr: np.ndarray) -> np.ndarray:
    """Crop array to at most 600 rows (matches GUI behaviour)."""
    if arr.shape[0] <= 600:
        return arr
    if arr.ndim == 3:
        return arr[:600, :, :]
    return arr[:600, :]


def load_txt_data(data_dir: str):
    """
    Find the first .txt file in *data_dir* and return a DataFrame with columns
    Field, Intensity, File.  Returns None if no valid file is found.
    """
    txt_file = None
    for fn in os.listdir(data_dir):
        if fn.lower().endswith(".txt"):
            txt_file = os.path.join(data_dir, fn)
            break
    if txt_file is None:
        return None

    try:
        df = pd.read_csv(txt_file, sep=None, engine="python",
                         comment="#", skip_blank_lines=True)
        df.columns = [c.strip() for c in df.columns]
        if len(df.columns) < 3:
            return None
        # Keep only rows whose third column ends with ".png"
        df = df[df[df.columns[2]].str.strip().str.lower().str.endswith(".png",
                                                                        na=False)]
        df = df.rename(columns={
            df.columns[0]: "Field",
            df.columns[1]: "Intensity",
            df.columns[2]: "File",
        }).reset_index(drop=True)
        df["Field"] = pd.to_numeric(df["Field"], errors="coerce")
        df["Intensity"] = pd.to_numeric(df["Intensity"], errors="coerce")
        df = df.dropna(subset=["Field", "Intensity"])
        return df if len(df) >= 3 else None
    except Exception as exc:
        print(f"  [warn] Could not read txt file {txt_file}: {exc}")
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


def run_subtraction_loop(data_dir: str, txt_df: pd.DataFrame,
                         background_array: np.ndarray) -> np.ndarray:
    """
    Subtract *background_array* from each image listed in *txt_df* and return
    the mean intensity of the difference as a 1-D float32 array.
    """
    means = []
    for _, row in txt_df.iterrows():
        img_file = row["File"].strip()
        img_path = os.path.join(data_dir, img_file)
        if not os.path.exists(img_path):
            means.append(np.nan)
            continue
        try:
            img_arr = np.array(Image.open(img_path))
            bg_arr  = background_array.copy()
            img_arr = crop600(img_arr)
            bg_arr  = crop600(bg_arr)
            min_shape = tuple(min(sa, sb)
                              for sa, sb in zip(img_arr.shape, bg_arr.shape))
            if img_arr.ndim == 3:
                img_c = img_arr[:min_shape[0], :min_shape[1], :min_shape[2]]
                bg_c  =  bg_arr[:min_shape[0], :min_shape[1], :min_shape[2]]
            else:
                img_c = img_arr[:min_shape[0], :min_shape[1]]
                bg_c  =  bg_arr[:min_shape[0], :min_shape[1]]
            diff = img_c.astype(np.float32) - bg_c.astype(np.float32)
            means.append(float(np.mean(diff)))
        except Exception as exc:
            print(f"  [warn] Error processing {img_file}: {exc}")
            means.append(np.nan)
    return np.array(means, dtype=np.float32)


def auto_correct_coeffs(field: np.ndarray, intensity: np.ndarray,
                        drift_corr: bool = True, linear_corr: bool = True,
                        quad_corr: bool = True) -> dict:
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
    # Linear Faraday: fit positive and negative saturation shelves
    # independently on each branch, then average. This prevents the
    # hysteresis step from biasing the Faraday slope.
    # ------------------------------------------------------------------
    sat_threshold = 0.80 * field_abs_max
    slopes = []
    for branch_idx in (asc_idx, desc_idx):
        f_b = field_off[branch_idx]
        y_b = intensity1[branch_idx]
        sat_pos = f_b > sat_threshold
        sat_neg = f_b < -sat_threshold
        branch_slopes = []
        if np.sum(sat_pos) >= 2:
            p_pos = np.polyfit(f_b[sat_pos], y_b[sat_pos], 1)
            branch_slopes.append(p_pos[0])
        if np.sum(sat_neg) >= 2:
            p_neg = np.polyfit(f_b[sat_neg], y_b[sat_neg], 1)
            branch_slopes.append(p_neg[0])
        if branch_slopes:
            slopes.append(np.mean(branch_slopes))

    linear_val = -float(np.mean(slopes)) if slopes else 0.0
    intensity2 = intensity1 + linear_val * field_off

    # ------------------------------------------------------------------
    # Residual quadratic (Cotton–Mouton): fit on the step-subtracted
    # background to prevent the MOKE step height from biasing the parameters.
    # ------------------------------------------------------------------
    sat_pos = field_off > sat_threshold
    sat_neg = field_off < -sat_threshold
    sat_all = sat_pos | sat_neg

    quad1           = 0.0
    quad_offset_val = 0.0

    if np.sum(sat_pos) >= 2 and np.sum(sat_neg) >= 2:
        # Subtract shelf means to get background-only signal at saturation
        M_pos = np.mean(intensity2[sat_pos])
        M_neg = np.mean(intensity2[sat_neg])
        y_bg = intensity2.copy()
        y_bg[sat_pos] -= M_pos
        y_bg[sat_neg] -= M_neg

        p2 = np.polyfit(field_off[sat_all], y_bg[sat_all], 2)
        a2, b2 = float(p2[0]), float(p2[1])
        if abs(a2) > 0:
            candidate_offset = -b2 / (2.0 * a2)
            if abs(candidate_offset) <= field_abs_max:
                # Physically plausible vertex position – use full quadratic
                quad1           = -a2
                quad_offset_val = candidate_offset
                linear_val      -= b2
            else:
                # Vertex far outside field range – absorb only linear part
                linear_val -= b2

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
        ptp = arr_corr.ptp()
        if ptp > 0:
            arr_corr = (arr_corr - arr_corr.min()) / ptp * 2 - 1
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
        hc_avg = 0.5 * (abs(hc_pos) + abs(hc_neg))

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


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def save_loop_plot(field: np.ndarray, ycorr: np.ndarray,
                   hc_hr: dict, title: str, save_path: str):
    """Save a hysteresis loop figure with Hc/Hr annotations."""
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

def process_directory(data_dir: str, analysis_dir: str,
                      drift_corr: bool = True, linear_corr: bool = True,
                      quad_corr: bool = True) -> dict | None:
    """
    Process one data sub-directory.  Returns a result dict on success,
    or None on failure (missing data / images).
    """
    dir_name = os.path.basename(data_dir)
    print(f"\nProcessing: {dir_name}")

    # ---- Load .txt data file ----
    txt_df = load_txt_data(data_dir)
    if txt_df is None:
        print(f"  [skip] No valid .txt data file found.")
        return None

    field = txt_df["Field"].to_numpy(dtype=np.float32)

    # ---- Collect image files ----
    image_files = sorted([
        f for f in os.listdir(data_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))
    ])
    if not image_files:
        print(f"  [skip] No image files found.")
        return None

    # ---- Select background image ----
    bg_name = find_background_image(data_dir, image_files)
    if bg_name is None:
        print(f"  [skip] Could not find a background image.")
        return None
    bg_path = os.path.join(data_dir, bg_name)
    try:
        background_array = np.array(Image.open(bg_path))
        print(f"  Background: {bg_name}")
    except Exception as exc:
        print(f"  [skip] Failed to load background image {bg_name}: {exc}")
        return None

    # ---- Image subtraction loop ----
    raw_intens = run_subtraction_loop(data_dir, txt_df, background_array)
    if np.all(np.isnan(raw_intens)):
        print(f"  [skip] All image subtractions failed (all NaN).")
        return None

    # Replace NaNs with linear interpolation so corrections don't crash
    nan_mask = np.isnan(raw_intens)
    if np.any(nan_mask):
        xp = np.where(~nan_mask)[0]
        fp = raw_intens[~nan_mask]
        raw_intens[nan_mask] = np.interp(np.where(nan_mask)[0], xp, fp)

    # ---- Auto-correction ----
    coeffs = auto_correct_coeffs(field, raw_intens, drift_corr=drift_corr,
                                 linear_corr=linear_corr, quad_corr=quad_corr)
    ycorr  = apply_correction(field, raw_intens, coeffs, normalize=True)
    print(f"  Auto-correct: drift={coeffs['drift']:.4f}, "
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

    print(f"  Hc+ = {_fmt(hc_pos)} mT, Hc- = {_fmt(hc_neg)} mT, Hc = {_fmt(hc_avg)} mT")
    print(f"  Hr(asc) = {_fmt(hr_asc)}, Hr(desc) = {_fmt(hr_desc)}, |Hr| = {_fmt(hr_abs)}")

    # ---- Save outputs ----
    safe_name = re.sub(r"[^\w\-]", "_", dir_name)  # filesystem-safe
    loop_img_path  = os.path.join(analysis_dir, f"{safe_name}_loop.png")
    loop_data_path = os.path.join(analysis_dir, f"{safe_name}_loop.txt")

    save_loop_plot(field, ycorr, hc_hr, title=dir_name, save_path=loop_img_path)
    save_loop_data(field, raw_intens, ycorr, hc_hr, coeffs,
                   dir_name=dir_name, save_path=loop_data_path)
    print(f"  [ok] Saved {os.path.basename(loop_img_path)} and "
          f"{os.path.basename(loop_data_path)}")

    # ---- Step detection ----
    step = extract_step_number(dir_name)
    if step is not None:
        print(f"  Step number detected: {step}")

    return dict(
        dir_name=dir_name,
        step=step,
        hc_pos=hc_pos,
        hc_neg=hc_neg,
        hc_avg=hc_avg,
        hr_asc=hr_asc,
        hr_desc=hr_desc,
        hr_abs=hr_abs,
    )


def run_batch(parent_dir: str, drift_corr: bool = True, linear_corr: bool = True,
              quad_corr: bool = True):
    """
    Top-level batch routine.  Scans *parent_dir* for data sub-directories,
    processes each one, and saves results + summary to an 'Analysis' sub-folder.
    """
    parent_dir = os.path.abspath(parent_dir)
    print(f"\n{'='*60}")
    print(f"Batch processing: {parent_dir}")
    print(f"{'='*60}")

    # ---- Discover sub-directories ----
    sub_dirs = sorted([
        os.path.join(parent_dir, d)
        for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
        and d.lower() != "analysis"   # don't process our own output
    ])

    if not sub_dirs:
        print("[error] No sub-directories found in the selected folder.")
        return

    # ---- Create Analysis output folder ----
    analysis_dir = os.path.join(parent_dir, "Analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    print(f"Output folder: {analysis_dir}\n")

    # ---- Process each sub-directory ----
    results = []
    for data_dir in sub_dirs:
        result = process_directory(data_dir, analysis_dir,
                                   drift_corr=drift_corr,
                                   linear_corr=linear_corr,
                                   quad_corr=quad_corr)
        if result is not None:
            results.append(result)

    if not results:
        print("\n[warn] No directories were processed successfully.")
        return

    # ---- Per-step summary plots (only if step numbers detected) ----
    has_steps = any(r.get("step") is not None for r in results)
    if has_steps:
        print(f"\nGenerating step-number summary plots…")
        save_summary_plots(results, analysis_dir)
    else:
        print("\n[info] No sequential step numbers detected – "
              "skipping Hc/Hr-vs-step plots.")

    # ---- Full results table (all directories) ----
    save_full_summary(results, analysis_dir)

    print(f"\n{'='*60}")
    print(f"Batch complete.  {len(results)} / {len(sub_dirs)} directories processed.")
    print(f"Results saved to: {analysis_dir}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# GUI Components (only active if PyQt5 is installed)
# ---------------------------------------------------------------------------

class StdoutRedirector(object):
    def __init__(self, signal):
        self.signal = signal

    def write(self, text):
        self.signal.emit(text)

    def flush(self):
        pass


if HAS_PYQT5:
    class ProcessingThread(QThread):
        finished_signal = pyqtSignal(bool)

        def __init__(self, parent_dir, drift, linear, quad):
            super().__init__()
            self.parent_dir = parent_dir
            self.drift = drift
            self.linear = linear
            self.quad = quad

        def run(self):
            try:
                run_batch(self.parent_dir, drift_corr=self.drift, linear_corr=self.linear, quad_corr=self.quad)
                self.finished_signal.emit(True)
            except Exception as e:
                print(f"\n[ERROR] Batch processing failed: {e}")
                self.finished_signal.emit(False)


    class BatchProcessorGUI(QWidget):
        append_text = pyqtSignal(str)

        def __init__(self, parent_dir=None, theme="dark"):
            super().__init__()
            self.parent_dir = parent_dir or ""
            self.theme = theme
            self.thread = None
            
            # Save original stdout/stderr
            self.old_stdout = sys.stdout
            self.old_stderr = sys.stderr
            
            self.init_ui()
            
            # Connect the redirection signal
            self.append_text.connect(self.on_append_text)
            sys.stdout = StdoutRedirector(self.append_text)
            sys.stderr = StdoutRedirector(self.append_text)

        def init_ui(self):
            self.setObjectName("MainBg")
            self.setWindowTitle("Batch Hysteresis Loop Processor")
            self.resize(750, 600)

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

            self.thread = ProcessingThread(
                parent_dir,
                drift=self.chk_drift.isChecked(),
                linear=self.chk_linear.isChecked(),
                quad=self.chk_quad.isChecked()
            )
            self.thread.finished_signal.connect(self.on_processing_finished)
            self.thread.start()

        def on_stop_batch(self):
            if self.thread and self.thread.isRunning():
                self.txt_console.append("\n[info] Stopping batch process...")
                self.thread.terminate()
                self.thread.wait()
                self.on_processing_finished(False)

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
            self.thread = None

        def closeEvent(self, event):
            if self.thread and self.thread.isRunning():
                self.thread.terminate()
                self.thread.wait()
            sys.stdout = self.old_stdout
            sys.stderr = self.old_stderr
            event.accept()


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
    args = parser.parse_args()

    parent_dir = args.parent_dir
    headless = args.headless or not HAS_PYQT5

    if not HAS_PYQT5 and not args.headless:
        print("[info] PyQt5 is not installed. Falling back to headless CLI mode.")

    if headless:
        # Run CLI mode
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

        run_batch(parent_dir)
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
