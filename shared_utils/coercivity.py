import numpy as np


def compute_coercivity_map(data, fields):
    """
    Computes spatial coercivity map for a 3D array of images.
    data: (frames, height, width) float array
    fields: (frames,) float array
    Returns: hc_pos_map, hc_neg_map (each shape (height, width))
    """
    f_off = fields - np.mean(fields)
    f_abs_max = np.max(np.abs(f_off))
    sat_pos_mask = f_off > 0.8 * f_abs_max
    sat_neg_mask = f_off < -0.8 * f_abs_max

    # If no points in >80%, just use the max/min index
    if not np.any(sat_pos_mask):
        sat_pos_mask[np.argmax(f_off)] = True
    if not np.any(sat_neg_mask):
        sat_neg_mask[np.argmin(f_off)] = True

    sat_pos = np.mean(data[sat_pos_mask], axis=0)
    sat_neg = np.mean(data[sat_neg_mask], axis=0)
    mid = 0.5 * (sat_pos + sat_neg)

    i_min = np.argmin(fields)
    i_max = np.argmax(fields)

    if i_min < i_max:
        asc_idx = np.arange(i_min, i_max + 1)
        desc_idx = np.concatenate(
            [np.arange(i_max, len(fields)), np.arange(0, i_min + 1)])
    else:
        desc_idx = np.arange(i_max, i_min + 1)
        asc_idx = np.concatenate(
            [np.arange(i_min, len(fields)), np.arange(0, i_max + 1)])

    f_asc = fields[asc_idx]
    y_asc = data[asc_idx]
    sort_asc = np.argsort(f_asc)
    f_asc = f_asc[sort_asc]
    y_asc = y_asc[sort_asc]

    f_desc = fields[desc_idx]
    y_desc = data[desc_idx]
    sort_desc = np.argsort(f_desc)
    f_desc = f_desc[sort_desc]
    y_desc = y_desc[sort_desc]

    def find_crossing(f_branch, y_branch, mid_level):
        yc = y_branch - mid_level
        sign_y = np.sign(yc)
        diff_sign = np.diff(sign_y, axis=0)
        cross_idx = np.argmax(np.abs(diff_sign) > 0, axis=0)

        r, c = np.indices(mid_level.shape)
        y0 = yc[cross_idx, r, c]
        y1 = yc[cross_idx + 1, r, c]
        f0 = f_branch[cross_idx]
        f1 = f_branch[cross_idx + 1]

        dy = y1 - y0
        dy[dy == 0] = 1e-6
        cross_f = f0 - y0 * (f1 - f0) / dy

        valid = np.any(np.abs(diff_sign) > 0, axis=0)
        cross_f[~valid] = np.nan
        return cross_f

    hc_pos_map = find_crossing(f_asc, y_asc, mid)
    hc_neg_map = find_crossing(f_desc, y_desc, mid)

    return hc_pos_map, hc_neg_map
