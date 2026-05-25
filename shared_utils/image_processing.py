import numpy as np
import math

def crop(arr, max_h=600, max_w=900, center_w=True):
    """Crop image to specified dimensions.

    Args:
        arr: The input array
        max_h: Maximum height
        max_w: Maximum width. If None, width is not cropped.
        center_w: Whether to center the crop horizontally
    """
    h, w = arr.shape[0], arr.shape[1]
    h_crop = min(h, max_h)

    if max_w is None:
        if arr.ndim == 3:
            return arr[:h_crop, :, :]
        return arr[:h_crop, :]

    w_crop = min(w, max_w)

    if center_w:
        w_start = (w - w_crop) // 2
    else:
        w_start = 0

    if arr.ndim == 3:
        return arr[:h_crop, w_start:w_start + w_crop, :]
    return arr[:h_crop, w_start:w_start + w_crop]

def crop600(arr):
    """Crop image to at most 600 rows × 900 cols (centred). Default for GUI tools."""
    return crop(arr, max_h=600, max_w=900, center_w=True)

def crop_batch(arr):
    """Crop array to at most 600 rows. Does not crop width."""
    return crop(arr, max_h=600, max_w=None)

def crop_focus(arr):
    """Crop arrays to 600x600 pixels (center horizontally, top vertically)."""
    return crop(arr, max_h=600, max_w=600, center_w=True)

def wiener_deconvolve(image, sigma, balance=0.02):
    if sigma <= 0.05:
        return image
    h, w = image.shape
    u = np.fft.fftfreq(h)
    v = np.fft.fftfreq(w)
    uu, vv = np.meshgrid(u, v, indexing='ij')
    otf = np.exp(-2 * np.pi**2 * sigma**2 * (uu**2 + vv**2))
    img_fft = np.fft.fft2(image)
    otf_conj = np.conj(otf)
    wiener_filter = otf_conj / (np.abs(otf)**2 + balance)
    deblurred_fft = img_fft * wiener_filter
    deblurred = np.real(np.fft.ifft2(deblurred_fft))
    return deblurred

def get_roi_mean(arr, shape_type, roi_data):
    if shape_type == "None" or roi_data is None:
        return np.mean(arr)

    h, w = arr.shape[:2]

    if shape_type in ["Rectangle", "Square"]:
        cx, cy, rw, rh, angle_deg = roi_data
        if rw <= 0 or rh <= 0:
            return np.mean(arr)

        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        y, x = np.ogrid[:h, :w]

        dx = x - cx
        dy = y - cy

        rot_x = dx * cos_a + dy * sin_a
        rot_y = -dx * sin_a + dy * cos_a

        mask = (np.abs(rot_x) <= rw / 2) & (np.abs(rot_y) <= rh / 2)

        if not np.any(mask):
            return np.mean(arr)
        return np.mean(arr[mask])

    elif shape_type == "Circle":
        cx, cy, r = roi_data
        if r <= 0:
            return np.mean(arr)

        y, x = np.ogrid[:h, :w]
        mask = (x - cx)**2 + (y - cy)**2 <= r**2

        if not np.any(mask):
            return np.mean(arr)
        return np.mean(arr[mask])

    return np.mean(arr)

def compute_subtracted_mean(img_arr, bg_base, enable_z=False, coeff=0.0, method_idx=0, field=0.0, enable_roi=False, roi_shape="None", roi_data=None, crop_func=crop600):
    # Crop target first to prevent metadata ringing & optimize speed
    img_arr = crop_func(img_arr)

    # If z-drift correction is enabled and we are blurring the reference,
    # we must copy the background for this specific iteration.
    # Otherwise, we can just use the pre-cropped array.
    if enable_z and method_idx == 0:
        bg_arr = bg_base.copy()
    else:
        bg_arr = bg_base

    if enable_z:
        import scipy.ndimage as ndimage
        sigma = coeff * (field ** 2)
        if sigma > 0.05:
            if method_idx == 0:
                # Blur Reference
                if bg_arr.ndim == 3:
                    for c in range(bg_arr.shape[2]):
                        bg_arr[:, :, c] = ndimage.gaussian_filter(bg_arr[:, :, c].astype(np.float64), sigma=sigma)
                else:
                    bg_arr = ndimage.gaussian_filter(bg_arr.astype(np.float64), sigma=sigma)
            else:
                # Deblur Target
                max_val = np.iinfo(img_arr.dtype).max if np.issubdtype(img_arr.dtype, np.integer) else 255
                orig_dtype = img_arr.dtype
                if img_arr.ndim == 3:
                    img_deblurred = np.zeros_like(img_arr, dtype=np.float64)
                    for c in range(img_arr.shape[2]):
                        img_deblurred[:, :, c] = wiener_deconvolve(img_arr[:, :, c].astype(np.float64), sigma=sigma)
                    img_arr = np.clip(img_deblurred, 0, max_val).astype(orig_dtype)
                else:
                    img_deblurred = wiener_deconvolve(img_arr.astype(np.float64), sigma=sigma)
                    img_arr = np.clip(img_deblurred, 0, max_val).astype(orig_dtype)

    # Check if shapes are identical to bypass tuple zip/min/slicing overhead
    if img_arr.shape == bg_arr.shape:
        # Subtract directly, forcing float32 output by ensuring background is float32 (prevents uint8 underflow)
        if bg_arr.dtype != np.float32:
            bg_arr = bg_arr.astype(np.float32)
        arr_cropped = img_arr - bg_arr
    else:
        min_shape = tuple(min(sa, sb) for sa, sb in zip(img_arr.shape, bg_arr.shape))
        if img_arr.ndim == 3:
            img_c = img_arr[:min_shape[0], :min_shape[1], :min_shape[2]]
            bg_c = bg_arr[:min_shape[0], :min_shape[1], :min_shape[2]]
        else:
            img_c = img_arr[:min_shape[0], :min_shape[1]]
            bg_c = bg_arr[:min_shape[0], :min_shape[1]]
        if bg_c.dtype != np.float32:
            bg_c = bg_c.astype(np.float32)
        arr_cropped = img_c - bg_c

    # Store raw subtraction (do not bake the correction in)
    if enable_roi:
        mean_val = get_roi_mean(arr_cropped, roi_shape, roi_data)
    else:
        mean_val = np.mean(arr_cropped)
    return float(mean_val)
