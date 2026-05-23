## 2024-05-21 - Image Pre-processing Optimization
**Learning:** In Kerr MOKE image processing scripts (`batch_processor.py`, `kerr_looper_AG.py`), the background reference image is used as the subtrahend against every image in the sweep. The codebase originally cropped and cast the background image inside the inner subtraction loop, causing redundant O(N) memory allocations and array operations.
**Action:** Always hoist invariant image cropping (`crop600`) and casting operations outside of iterative loops. Be careful when background images are mutated per-iteration (e.g., dynamic Z-drift blurring), in which case `.copy()` on the pre-cropped reference should be conditionally applied.

## 2024-05-22 - Fast Matplotlib Frame Extraction
**Learning:** Inside a loop (like `make_movie`), using `savefig` with a `BytesIO` buffer adds massive overhead from PNG encoding/decoding.
**Action:** Use `fig.canvas.draw()` followed by `fig.canvas.buffer_rgba()` and `Image.frombuffer()` for a 2-3x speedup when extracting raw frames from Matplotlib.

## Iterating Pandas DataFrames
When iterating over Pandas DataFrames in performance-sensitive sections, prefer `enumerate(df.itertuples(index=False))` over `df.iterrows()`. Using `itertuples` returns namedtuples instead of constructing Pandas Series for every row, making it significantly faster (e.g. from 0.5s to 0.02s for 10000 rows). Also be sure to change `row['Column']` syntax to `row.Column`.
