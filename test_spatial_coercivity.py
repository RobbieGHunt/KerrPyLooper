import numpy as np
import pytest
from shared_utils.coercivity import compute_coercivity_map

def test_coercivity_map():
    frames = 50
    height = 5
    width = 5
    fields = np.linspace(-10, 10, frames)
    data = np.zeros((frames, height, width))
    hc_true = np.linspace(2, 8, width)
    for i in range(height):
        for j in range(width):
            data[:, i, j] = np.tanh(fields - hc_true[j])

    hc_pos, hc_neg = compute_coercivity_map(data, fields)

    for i in range(height):
        for j in range(width):
            assert np.abs(hc_pos[i, j] - hc_true[j]) < 0.2
