import numpy as np
from scipy.signal import convolve2d


class SimpleMLEngine:
    """
    A lightweight inference engine using NumPy and SciPy to simulate
    a small CNN for super-resolution and advanced denoising of Kerr images,
    avoiding heavy dependencies like PyTorch or TensorFlow.
    """
    def __init__(self):
        # Define some basic kernels simulating convolutional layers

        # 1. Smoothing / Denoising (Gaussian-like)
        self.kernel_denoise = np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=np.float32) / 16.0

        # 2. Sharpening / Super-resolution (Laplacian-like)
        self.kernel_sharpen = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)

    def enhance(self, image_array):
        """
        Enhance the image array.
        Expects a 2D NumPy array.
        """
        if image_array is None:
            return None

        original_dtype = image_array.dtype
        image_float = image_array.astype(np.float32)

        # Step 1: Denoise
        denoised = convolve2d(
            image_float, self.kernel_denoise,
            mode='same', boundary='symm'
        )

        # Step 2: Sharpen
        sharpened = convolve2d(
            denoised, self.kernel_sharpen,
            mode='same', boundary='symm'
        )

        # Clip to valid range and cast back
        # Assume it's a difference image if there are negative values
        if np.any(image_float < 0):
            # Difference array, don't clip to 0
            enhanced = sharpened
        else:
            # Standard image
            if np.issubdtype(original_dtype, np.integer):
                max_val = np.iinfo(original_dtype).max
                enhanced = np.clip(sharpened, 0, max_val)
            else:
                enhanced = np.clip(sharpened, 0, 255)  # Fallback

        return enhanced.astype(original_dtype)
