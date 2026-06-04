import sys
import operator
import types
# NumPy 2.x namespace compatibility shim for older dependencies (e.g. SciPy, scikit-image)
try:
    import numpy as _np
    _np.core = _np._core
    sys.modules["numpy.core"] = _np._core
    for _name, _obj in list(_np._core.__dict__.items()):
        if isinstance(_obj, type(sys)):
            sys.modules[f"numpy.core.{_name}"] = _obj
    del _name, _obj, _np
except Exception:
    pass

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

_wiener_filter_rcache = {}
_ruu2_rvv2_cache = {}

try:
    from scipy.fft import rfft2 as _scipy_rfft2, irfft2 as _scipy_irfft2
    def _rfft2(x): return _scipy_rfft2(x, workers=1)
    def _irfft2(x, s=None): return _scipy_irfft2(x, s=s, workers=1)
except ImportError:
    from numpy.fft import rfft2 as _rfft2, irfft2 as _irfft2

def wiener_deconvolve(image, sigma, balance=0.02):
    if sigma <= 0.05:
        return image
    h, w = image.shape
    key_filter = (h, w, float(sigma), float(balance))
    if key_filter in _wiener_filter_rcache:
        wiener_filter = _wiener_filter_rcache[key_filter]
    else:
        key_grid = (h, w)
        if key_grid in _ruu2_rvv2_cache:
            uu2_vv2 = _ruu2_rvv2_cache[key_grid]
        else:
            u = np.fft.fftfreq(h).astype(np.float32)
            v = np.fft.rfftfreq(w).astype(np.float32)
            uu2_vv2 = u[:, np.newaxis]**2 + v[np.newaxis, :]**2
            if len(_ruu2_rvv2_cache) >= 32:
                _ruu2_rvv2_cache.pop(next(iter(_ruu2_rvv2_cache)))
            _ruu2_rvv2_cache[key_grid] = uu2_vv2
        otf = np.exp((-2.0 * np.pi**2 * sigma**2) * uu2_vv2)
        wiener_filter = (otf / (otf**2 + balance)).astype(np.float32)
        if len(_wiener_filter_rcache) >= 128:
            _wiener_filter_rcache.pop(next(iter(_wiener_filter_rcache)))
        _wiener_filter_rcache[key_filter] = wiener_filter

    image_f32 = image.astype(np.float32, copy=False)
    img_fft = _rfft2(image_f32)
    deblurred_fft = img_fft * wiener_filter
    deblurred = _irfft2(deblurred_fft, s=(h, w))
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

import concurrent.futures

def extract_hc_from_loop(field, intensity):
    """Robustly extract global Hc from a single loop."""
    if len(field) < 2 or len(intensity) < 2:
        return np.nan

    i_min = np.argmin(field)
    i_max = np.argmax(field)

    if i_min < i_max:
        asc_idx = list(range(i_min, i_max + 1))
        desc_idx = list(range(i_max, len(field))) + list(range(0, i_min + 1))
    else:
        desc_idx = list(range(i_max, i_min + 1))
        asc_idx = list(range(i_min, len(field))) + list(range(0, i_max + 1))

    f_asc = field[asc_idx]
    y_asc = intensity[asc_idx]
    sort_asc = np.argsort(f_asc)
    f_asc = f_asc[sort_asc]
    y_asc = y_asc[sort_asc]

    f_desc = field[desc_idx]
    y_desc = intensity[desc_idx]
    sort_desc = np.argsort(f_desc)
    f_desc = f_desc[sort_desc]
    y_desc = y_desc[sort_desc]

    field_off = field - np.mean(field)
    field_abs_max = np.max(np.abs(field_off))

    if field_abs_max == 0:
        return np.nan

    sat_mask_pos = field_off > (0.8 * field_abs_max)
    sat_mask_neg = field_off < (-0.8 * field_abs_max)

    sat_pos = np.mean(intensity[sat_mask_pos]) if np.sum(sat_mask_pos) > 0 else intensity[i_max]
    sat_neg = np.mean(intensity[sat_mask_neg]) if np.sum(sat_mask_neg) > 0 else intensity[i_min]
    mid = 0.5 * (sat_pos + sat_neg)

    def find_crossings(f_branch, y_branch, mid_level):
        crossings = []
        for i in range(len(f_branch) - 1):
            y0, y1 = y_branch[i], y_branch[i+1]
            f0, f1 = f_branch[i], f_branch[i+1]
            if (y0 - mid_level) * (y1 - mid_level) <= 0 and y0 != y1:
                frac = (mid_level - y0) / (y1 - y0)
                f_cross = f0 + frac * (f1 - f0)
                crossings.append((f_cross, abs(y1 - y0)))
        return crossings

    crossings_asc = find_crossings(f_asc, y_asc, mid)
    crossings_desc = find_crossings(f_desc, y_desc, mid)

    hc_pos = None
    if crossings_asc:
        crossings_asc.sort(key=lambda x: x[1], reverse=True)
        hc_pos = crossings_asc[0][0]
    elif len(f_asc) > 0:
        hc_pos = f_asc[np.argmin(np.abs(y_asc - mid))]

    hc_neg = None
    if crossings_desc:
        crossings_desc.sort(key=lambda x: x[1], reverse=True)
        hc_neg = crossings_desc[0][0]
    elif len(f_desc) > 0:
        hc_neg = f_desc[np.argmin(np.abs(y_desc - mid))]

    if hc_pos is not None and hc_neg is not None:
        return (abs(hc_pos) + abs(hc_neg)) / 2.0
    elif hc_pos is not None:
        return abs(hc_pos)
    elif hc_neg is not None:
        return abs(hc_neg)
    return np.nan


def calculate_local_hc(img_stack, field, bin_size=4, return_hr=False, apply_corrections=True):
    """
    Calculate spatially resolved local coercivity mapping.
    img_stack: (N_images, H, W) float32 array
    field: (N_images,) array
    bin_size: integer for downsampling/binning
    """
    N, H, W = img_stack.shape

    # Subsample or bin the image stack
    binned_stack = img_stack[:, ::bin_size, ::bin_size]
    b_H, b_W = binned_stack.shape[1], binned_stack.shape[2]
    P = b_H * b_W

    # Flatten spatial dimensions to (N, P), making a copy to ensure safe in-place operations
    flat_stack = binned_stack.reshape(N, P).copy()

    # Background subtraction
    zero_idx = np.argmin(np.abs(field))
    flat_stack -= flat_stack[zero_idx, :]

    if N < 2:
        hc_map = np.full((b_H, b_W), np.nan, dtype=np.float32)
        hc_map_full = np.repeat(np.repeat(hc_map, bin_size, axis=0), bin_size, axis=1)
        if return_hr:
            return hc_map_full[:H, :W], hc_map_full[:H, :W]
        return hc_map_full[:H, :W]

    i_min = np.argmin(field)
    i_max = np.argmax(field)

    if i_min < i_max:
        asc_idx = np.arange(i_min, i_max + 1)
        desc_idx = np.concatenate([np.arange(i_max, N), np.arange(0, i_min + 1)])
    else:
        desc_idx = np.arange(i_max, i_min + 1)
        asc_idx = np.concatenate([np.arange(i_min, N), np.arange(0, i_max + 1)])

    f_asc = field[asc_idx]
    sort_asc = np.argsort(f_asc)
    f_asc = f_asc[sort_asc]

    f_desc = field[desc_idx]
    sort_desc = np.argsort(f_desc)
    f_desc = f_desc[sort_desc]

    field_off = field - np.mean(field)
    field_abs_max = np.max(np.abs(field_off))

    if field_abs_max == 0:
        hc_map = np.full((b_H, b_W), np.nan, dtype=np.float32)
        hc_map_full = np.repeat(np.repeat(hc_map, bin_size, axis=0), bin_size, axis=1)
        if return_hr:
            return hc_map_full[:H, :W], hc_map_full[:H, :W]
        return hc_map_full[:H, :W]

    sat_mask_pos = field_off > (0.8 * field_abs_max)
    sat_mask_neg = field_off < (-0.8 * field_abs_max)

    if np.sum(sat_mask_pos) > 0:
        sat_pos = np.mean(flat_stack[sat_mask_pos, :], axis=0)
    else:
        sat_pos = flat_stack[i_max, :]

    if np.sum(sat_mask_neg) > 0:
        sat_neg = np.mean(flat_stack[sat_mask_neg, :], axis=0)
    else:
        sat_neg = flat_stack[i_min, :]

    mid = 0.5 * (sat_pos + sat_neg)

    # ------------------------------------------------------------------
    # Apply Vectorized Loop Corrections (drift, linear Faraday, quadratic Cotton-Mouton)
    # ------------------------------------------------------------------
    if apply_corrections:
        idx = np.arange(N, dtype=np.float32)
        idx_off = idx - idx.mean()
        field_off = field - np.mean(field)
        field_abs_max = np.max(np.abs(field_off))

        # Pass 1 - endpoint drift alignment (vectorized per pixel, in-place)
        drift1 = (flat_stack[0, :] - flat_stack[-1, :]) / N
        flat_stack += drift1 * idx_off[:, np.newaxis]

        # Joint 4-Parameter Fit: y = c1*h + c2*h^2 + c3*sign(h) + c4
        sat_threshold = 0.80 * field_abs_max
        fit_mask = np.abs(field_off) > sat_threshold
        if np.sum(fit_mask) < 4:
            sat_threshold = 0.50 * field_abs_max
            fit_mask = np.abs(field_off) > sat_threshold

        if np.sum(fit_mask) >= 4:
            h_fit = field_off[fit_mask]
            y_fit = flat_stack[fit_mask, :]
            
            # Design matrix for least squares fit: shape (M_fit, 4)
            A = np.column_stack([h_fit, h_fit**2, np.sign(h_fit), np.ones_like(h_fit)])
            try:
                # Solve for all pixels simultaneously: A @ coeffs = y_fit
                # Using np.linalg.pinv(A) @ y_fit is ~27x faster than np.linalg.lstsq
                coeffs_fit = np.linalg.pinv(A) @ y_fit
                c1, c2 = coeffs_fit[0], coeffs_fit[1]
                linear_val = -c1
                quad1 = -c2
                
                # Check for extreme quadratic corrections and clip them
                # A typical Cotton-Mouton contribution shouldn't exceed reasonable bounds of signal range.
                # Clip c2/quad1 to +/- 10.0 / (field_abs_max**2) to avoid runaway fits at noisy pixels.
                quad_limit = 10.0 / (field_abs_max ** 2) if field_abs_max > 0 else 1.0
                quad1 = np.clip(quad1, -quad_limit, quad_limit)
            except Exception:
                linear_val = np.zeros(P, dtype=np.float32)
                quad1 = np.zeros(P, dtype=np.float32)
        else:
            linear_val = np.zeros(P, dtype=np.float32)
            quad1 = np.zeros(P, dtype=np.float32)

        # Pass 2 - apply shape correction & second drift correction (in-place additions)
        field_off_col = field_off[:, np.newaxis]
        field_off_sq_col = (field_off ** 2)[:, np.newaxis]
        flat_stack += field_off_col * linear_val + field_off_sq_col * quad1

        drift2 = (flat_stack[0, :] - flat_stack[-1, :]) / N
        flat_stack += drift2 * idx_off[:, np.newaxis]

    # Re-evaluate saturation shelves after correction to calculate a clean midpoint level
    if np.sum(sat_mask_pos) > 0:
        sat_pos = np.mean(flat_stack[sat_mask_pos, :], axis=0)
    else:
        sat_pos = flat_stack[i_max, :]

    if np.sum(sat_mask_neg) > 0:
        sat_neg = np.mean(flat_stack[sat_mask_neg, :], axis=0)
    else:
        sat_neg = flat_stack[i_min, :]

    mid = 0.5 * (sat_pos + sat_neg)

    # Use single-pass indexing to avoid duplicate array allocations and copies
    y_asc = flat_stack[asc_idx[sort_asc], :]
    y_desc = flat_stack[desc_idx[sort_desc], :]

    def find_crossings_vectorized(f_branch, y_branch, mid):
        M, P_dim = y_branch.shape
        if M < 2:
            return np.full(P_dim, np.nan, dtype=np.float32)

        diff = y_branch - mid
        diff_0 = diff[:-1, :]
        diff_1 = diff[1:, :]

        y0 = y_branch[:-1, :]
        y1 = y_branch[1:, :]

        cross_mask = (diff_0 * diff_1 <= 0)
        cross_mask &= (y0 != y1)
        dy = np.abs(y1 - y0)
        dy_masked = np.where(cross_mask, dy, -1.0)

        i_max_sel = np.argmax(dy_masked, axis=0)
        col_indices = np.arange(P_dim)
        max_dy = dy_masked[i_max_sel, col_indices]

        f0 = f_branch[i_max_sel]
        f1 = f_branch[np.minimum(i_max_sel + 1, M - 1)]

        y0_sel = y0[i_max_sel, col_indices]
        y1_sel = y1[i_max_sel, col_indices]

        denom = y1_sel - y0_sel
        
        # Prevent division-by-zero warnings by calculating selectively
        frac = np.zeros_like(denom)
        valid_denom = denom != 0
        frac[valid_denom] = (mid[valid_denom] - y0_sel[valid_denom]) / denom[valid_denom]
        
        f_cross = f0 + frac * (f1 - f0)

        # Only calculate fallback when actually needed
        fallback_mask = max_dy < 0
        if np.any(fallback_mask):
            fallback_idx = np.argmin(np.abs(diff[:, fallback_mask]), axis=0)
            hc_fallback = f_branch[fallback_idx]
            f_cross[fallback_mask] = hc_fallback

        return f_cross

    hc_pos = find_crossings_vectorized(f_asc, y_asc, mid)
    hc_neg = find_crossings_vectorized(f_desc, y_desc, mid)

    nan_pos = np.isnan(hc_pos)
    nan_neg = np.isnan(hc_neg)

    hc_combined = np.where(
        ~nan_pos & ~nan_neg,
        (np.abs(hc_pos) + np.abs(hc_neg)) / 2.0,
        np.where(
            ~nan_pos,
            np.abs(hc_pos),
            np.where(
                ~nan_neg,
                np.abs(hc_neg),
                np.nan
            )
        )
    )

    hc_map = hc_combined.reshape(b_H, b_W)
    hc_map_full = np.repeat(np.repeat(hc_map, bin_size, axis=0), bin_size, axis=1)

    if return_hr:
        iz_asc = np.argmin(np.abs(f_asc))
        iz_desc = np.argmin(np.abs(f_desc))
        hr_asc = y_asc[iz_asc, :]
        hr_desc = y_desc[iz_desc, :]
        hr_abs = 0.5 * (np.abs(hr_asc) + np.abs(hr_desc))
        
        hr_map = hr_abs.reshape(b_H, b_W)
        hr_map_full = np.repeat(np.repeat(hr_map, bin_size, axis=0), bin_size, axis=1)
        return hc_map_full[:H, :W], hr_map_full[:H, :W]

    return hc_map_full[:H, :W]


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
