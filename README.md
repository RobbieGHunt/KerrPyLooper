# KerrPyLooper

KerrPyLooper is a tool for processing, correcting, and analyzing Kerr microscopy image series to measure magnetization hysteresis loops (MOKE loops).

It features robust background subtraction, dynamic Faraday rotation corrections, auto-coercivity estimation, and a field-dependent out-of-plane focus drift correction algorithm.

This program was made using old, often very manual scripts as the basis and transferred into a gui environment using Agentic AI. While the corrections and processing uses many standard methods, always be aware of this when using the technique.

## Features

- **Interactive GUI**:
  - Resizable panel layouts using a vertical splitter.
  - Real-time previews of background subtracted images with custom contrast stretch.
  - Synchronized image contrast adjustment controls.
  - "Zero All" shortcut button to instantly reset all correction factors.
  - Dynamic slider scaling to make manual adjustments physically intuitive.
  - Everything you can do in the "offline looper" available with an Evico microscope, and hopefully some more.

- **Ringing-Free Image Processing**: Pre-crops target and reference images to the relevant content (if you are using an Evico microscope) to eliminate bottom metadata labels and text, avoiding boundary ringing artifacts during FFT.

- **Robust Coercivity ($H_c$) and Remanence ($H_r$) Extraction**: Uses linear interpolation over saturation-focused midpoints to identify magnetic switching events, remaining immune to transient noise and switching shifts.

- **Batch Processing of Hysteresis Loops**: Able to explore many sub-directories to process hysteresis loop corrections and output the coercivity and remanence, will automatically plot this as a number of "steps" if folder are labelled in this way.

- **Vector analysis of magnetic domains**: With two datasets corresponding to longitudinal and transverse signals, reconstruct the local magnetization within the field of view allowing for more detailed analysis of domain structures.

- **Field-Dependent Out-of-Plane Focus Correction (Z-Drift)** (your mileage may vary):
  - **Defect ROI Selection**: Automatically selects a $128 \times 128$ pixel static defect region in the zero-field image to perform defocus estimation, ensuring defect boundaries stay in sharp focus.
  - **Wiener Deconvolution**: Restores image focus dynamically using frequency-domain deconvolution.
  - **Dynamic Range Scaling**: Fully supports 16-bit images by scaling deconvolution intensity clipping dynamically based on the input image's dtype.


## Codebase Structure

- `kerr_looper_AG.py`: The main interactive PyQt5 GUI application.
- `focus_corrector.py`: Standalone command-line Python script to batch-process focus drift correction on raw image series.

---

## Installation

### Prerequisites
Make sure you have Python 3 installed along with the required libraries:
```bash
pip install numpy scipy pandas pillow matplotlib PyQt5
```

---

## Usage

### 1. Main GUI Tool
Launch the interactive MOKE Loop Subtractor tool:
```bash
python suite_launcher.py
```
Launches a GUI from which you can access the sub-programs:

**Kerr Looper**
- Load your image directory (e.g. `example1`).
- Load your mapping file (e.g. `20260202A_A_dry.txt`).
- Select your zero-field background image and set it.
- Toggle Z-Drift correction and click "Auto Estimate Focus Drift".
- Fine-tune corrections (drift, linear Faraday, quadratic Faraday) using sliders.
- Click "Auto Correct" to fit correction parameters.
- View and export the hysteresis loop or the subtracted contrast images.

**Batch Processor**
 - Select the directory that contains all the loops you want to process.
 - Select which corrections to automatically apply.
 - Hit go!

**Vector Analyzer**
 - Select the directory that contains two directories: "x" for the longitudinal data (along the field direction) and "y" for the transverse data.
 - View the entire hysteresis loop in vector image format, and export all the images as a gif if desired.
 - Denoising options to correct for contamination that can cause artifacts in images.
 - Freely select the region of interest and only consider the vector analysis in this region if desired.

### 2. Standalone Focus Corrector Script
To batch-correct out-of-plane focus drift on a raw series from the command line:
```bash
python focus_corrector.py --img_dir "path/to/raw_images" --txt_path "path/to/mapping.txt" --output_dir "path/to/output_corrected"
```
This generates a folder of focus-corrected, cropped $600 \times 600$ images, the copied mapping text file, and a diagnostic defocus analysis plot (`focus_drift_analysis.png`).

