## 2024-05-21 - Image Pre-processing Optimization
**Learning:** In Kerr MOKE image processing scripts (`batch_processor.py`, `kerr_looper_AG.py`), the background reference image is used as the subtrahend against every image in the sweep. The codebase originally cropped and cast the background image inside the inner subtraction loop, causing redundant O(N) memory allocations and array operations.
**Action:** Always hoist invariant image cropping (`crop600`) and casting operations outside of iterative loops. Be careful when background images are mutated per-iteration (e.g., dynamic Z-drift blurring), in which case `.copy()` on the pre-cropped reference should be conditionally applied.

## 2024-05-22 - Fast Matplotlib Frame Extraction
**Learning:** Inside a loop (like `make_movie`), using `savefig` with a `BytesIO` buffer adds massive overhead from PNG encoding/decoding.
**Action:** Use `fig.canvas.draw()` followed by `fig.canvas.buffer_rgba()` and `Image.frombuffer()` for a 2-3x speedup when extracting raw frames from Matplotlib.

## Iterating Pandas DataFrames
When iterating over Pandas DataFrames in performance-sensitive sections, prefer `enumerate(df.itertuples(index=False))` over `df.iterrows()`. Using `itertuples` returns namedtuples instead of constructing Pandas Series for every row, making it significantly faster (e.g. from 0.5s to 0.02s for 10000 rows). Also be sure to change `row['Column']` syntax to `row.Column`.
- Iterating pandas DataFrames: Use `df.itertuples(index=False)` instead of `df.iterrows()`. Measured a ~15x iteration speedup doing this in focus_corrector.py.
- Synchronous I/O in Pandas dataframe loops within a PyQt5 GUI can be safely optimized using `concurrent.futures.ThreadPoolExecutor` and `.map`. Operations like PIL image opening and basic numpy conversions release the GIL, providing significant speedups. When parallelizing dataframe iterations, replacing `iterrows()` with `itertuples(index=False)` provides additional execution speed benefits by reducing series creation overhead.
## 2025-02-09 - [Drift Corrector SQDIFF FFT]
**Learning:** O(N^2) nested loops for template matching in `estimate_shift_sqdiff` are extremely slow. They can be perfectly replaced by fast frequency domain cross-correlation. Since SQDIFF computes `sum((ref - target)^2) = sum(ref^2) + sum(target^2) - 2 * sum(ref * target)`, the third term is exactly `scipy.signal.fftconvolve(target, ref[::-1, ::-1])`, while the second term is `fftconvolve(target**2, ones)`. The first term is a constant. Using `mode='valid'` and extracting only the search region (with proper NaN padding for edge cases) correctly evaluates the distance map over the exact candidate shifts.
**Action:** Always replace O(N^2) pixel-shifting nested loops for image matching with `scipy.signal.fftconvolve` using the expanded quadratic formula for exact matching without the massive performance penalty.
