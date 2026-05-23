import os
import argparse
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
import pandas as pd
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from shared_utils.image_processing import crop_focus, wiener_deconvolve

def normalize_image(img):
    """
    Normalize image contrast/gain to have mean 0 and standard deviation 1.
    This eliminates overall brightness or intensity drift differences.
    """
    m = np.mean(img)
    s = np.std(img)
    if s == 0:
        return img - m
    return (img - m) / s

def get_gradient_magnitude(img):
    """
    Calculate Sobel gradient magnitude of an image.
    High-pass filters to isolate static defects from low-frequency domain features.
    """
    dx = ndimage.sobel(img, axis=0)
    dy = ndimage.sobel(img, axis=1)
    return np.sqrt(dx**2 + dy**2)

def find_defect_roi(img, patch_size=128):
    """
    Find coordinates of the patch_size x patch_size region in the image with
    the highest Sobel gradient energy (indicative of static defects/scratches).
    """
    grad = get_gradient_magnitude(img)
    h, w = img.shape[0], img.shape[1]
    step = 10
    margin = 20

    r_range = np.arange(margin, h - patch_size - margin, step)
    c_range = np.arange(margin, w - patch_size - margin, step)

    if len(r_range) == 0 or len(c_range) == 0:
        return (150, 150)

    window_sums = ndimage.uniform_filter(grad, size=patch_size, mode='constant') * (patch_size**2)

    r_idx = r_range + patch_size // 2
    c_idx = c_range + patch_size // 2

    scores_subgrid = window_sums[np.ix_(r_idx, c_idx)]
    max_idx = np.unravel_index(np.argmax(scores_subgrid), scores_subgrid.shape)

    return (int(r_range[max_idx[0]]), int(c_range[max_idx[1]]))

def estimate_defocus(ref_img_norm, target_img_norm):
    """
    Find the Gaussian blur standard deviation (sigma) that, when applied
    to ref_img_norm, best matches the gradient magnitude of target_img_norm.
    """
    target_grad = get_gradient_magnitude(target_img_norm)
    
    def loss(sigma):
        if sigma <= 0.01:
            blurred_ref = ref_img_norm
        else:
            blurred_ref = ndimage.gaussian_filter(ref_img_norm, sigma=sigma)
        blurred_grad = get_gradient_magnitude(blurred_ref)
        return np.mean((target_grad - blurred_grad) ** 2)
    
    res = minimize_scalar(loss, bounds=(0.0, 3.0), method='bounded')
    return res.x

def focus_correct_series(img_dir, txt_path, output_dir, balance=0.02):
    print(f"Loading mapping file: {txt_path}")
    df = pd.read_csv(txt_path, sep=None, engine='python', comment="#", skip_blank_lines=True)
    df.columns = [c.strip() for c in df.columns]
    
    # Rename columns to ensure standard names
    field_col = df.columns[0]
    intens_col = df.columns[1]
    file_col = df.columns[2]
    df = df.rename(columns={field_col: "Field", intens_col: "Intensity", file_col: "File"})
    df = df[df["File"].str.lower().str.endswith(".png", na=False)].reset_index(drop=True)
    
    # Identify reference image (closest to zero field)
    zero_idx = np.argmin(np.abs(df["Field"]))
    ref_filename = df.iloc[zero_idx]["File"].strip()
    ref_field = df.iloc[zero_idx]["Field"]
    print(f"Reference in-focus image chosen: {ref_filename} at Field: {ref_field:.4f} mT")
    
    ref_path = os.path.join(img_dir, ref_filename)
    ref_img_raw = np.array(Image.open(ref_path))
    if ref_img_raw.ndim == 3:
        ref_img = (0.2989 * ref_img_raw[:, :, 0] + 0.5870 * ref_img_raw[:, :, 1] + 0.1140 * ref_img_raw[:, :, 2])
    else:
        ref_img = ref_img_raw.astype(np.float64)
    ref_cropped = crop_focus(ref_img)
    
    # Find defect ROI coordinates in 600x600 image
    patch_size = 128
    r, c = find_defect_roi(ref_cropped, patch_size=patch_size)
    print(f"Detected defect ROI for focus estimation at row: {r}, col: {c}")
    ref_patch = ref_cropped[r:r+patch_size, c:c+patch_size]
    ref_patch_norm = normalize_image(ref_patch)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Track defocus for each image
    sigmas = []
    fields = []
    filenames = []
    
    print("\nEstimating defocus parameters across the series...")
    for idx, row in enumerate(df.itertuples(index=False)):
        fname = row.File.strip()
        fpath = os.path.join(img_dir, fname)
        target_img_raw = np.array(Image.open(fpath))
        if target_img_raw.ndim == 3:
            target_img = (0.2989 * target_img_raw[:, :, 0] + 0.5870 * target_img_raw[:, :, 1] + 0.1140 * target_img_raw[:, :, 2])
        else:
            target_img = target_img_raw.astype(np.float64)
        target_cropped = crop_focus(target_img)
        target_patch = target_cropped[r:r+patch_size, c:c+patch_size]
        target_patch_norm = normalize_image(target_patch)
        
        sigma_est = estimate_defocus(ref_patch_norm, target_patch_norm)
        sigmas.append(sigma_est)
        fields.append(row.Field)
        filenames.append(fname)
        
        if idx % 10 == 0 or idx == len(df) - 1:
            print(f"  Processed {idx+1}/{len(df)}: {fname} (Field: {row.Field:8.2f} mT) -> Defocus Sigma: {sigma_est:.4f}")
            
    # Fit quadratic model to defocus curve
    fields = np.array(fields)
    sigmas = np.array(sigmas)
    p = np.polyfit(fields, sigmas, 2)
    fitted_sigmas = np.polyval(p, fields)
    # Clip fitted sigmas to be non-negative
    fitted_sigmas = np.maximum(0, fitted_sigmas)
    
    print(f"\nDefocus curve fit: sigma = {p[0]:.2e} * H^2 + {p[1]:.2e} * H + {p[2]:.4f}")
    
    # Generate and save diagnostic plot
    plt.figure(figsize=(8, 5))
    plt.scatter(fields, sigmas, color='b', alpha=0.6, label='Estimated Defocus')
    plt.plot(np.sort(fields), np.polyval(p, np.sort(fields)), color='r', lw=2, label='Quadratic Fit')
    plt.xlabel('Magnetic Field (mT)')
    plt.ylabel('Defocus Sigma (pixels)')
    plt.title('Field-Dependent Out-of-Plane Defocus')
    plt.grid(True)
    plt.legend()
    plot_path = os.path.join(output_dir, "focus_drift_analysis.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved diagnostic plot: {plot_path}")
    
    # Correct and save the images
    print("\nApplying focus correction (deblurring target images)...")
    for idx, row in enumerate(df.itertuples(index=False)):
        fname = filenames[idx]
        fpath = os.path.join(img_dir, fname)
        target_img_pil = Image.open(fpath)
        target_arr_raw = np.array(target_img_pil)
        
        # Crop the target to 600x600 first
        target_arr = crop_focus(target_arr_raw)
        
        # Determine defocus value from fitted curve (smooths out noise)
        sigma_fit = fitted_sigmas[idx]
        
        # Deblur
        if len(target_arr.shape) == 3:
            # Color image: deblur each channel independently
            deblurred_channels = []
            for c in range(target_arr.shape[2]):
                deblurred_channels.append(wiener_deconvolve(target_arr[:, :, c].astype(np.float64), sigma_fit, balance))
            deblurred = np.stack(deblurred_channels, axis=2)
        else:
            deblurred = wiener_deconvolve(target_arr.astype(np.float64), sigma_fit, balance)
            
        # Dynamically determine dynamic range limits based on input dtype
        max_val = np.iinfo(target_arr.dtype).max if np.issubdtype(target_arr.dtype, np.integer) else 255
        orig_dtype = target_arr.dtype
        deblurred_clipped = np.clip(deblurred, 0, max_val).astype(orig_dtype)
        
        # Save output image
        out_path = os.path.join(output_dir, fname)
        Image.fromarray(deblurred_clipped).save(out_path)
        
    # Copy the mapping text file to output directory as well
    out_txt_path = os.path.join(output_dir, os.path.basename(txt_path))
    df_out = df.copy()
    # Write back the original column names
    df_out.columns = df.columns
    df_out.to_csv(out_txt_path, sep='\t', index=False)
    print(f"Focus correction completed. Saved all corrected files to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correct field-dependent focus drift in Kerr microscopy image series.")
    parser.add_argument("--img_dir", required=True, help="Directory containing original image PNGs")
    parser.add_argument("--txt_path", required=True, help="Path to mapping text file")
    parser.add_argument("--output_dir", required=True, help="Directory to save focus-corrected images")
    parser.add_argument("--balance", type=float, default=0.02, help="Wiener filter noise-to-signal balance (default: 0.02)")
    
    args = parser.parse_args()
    focus_correct_series(args.img_dir, args.txt_path, args.output_dir, args.balance)
