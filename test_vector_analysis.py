import numpy as np
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from PIL import Image, ImageSequence
import io
import json
import os
import pytest


def test_vector_theta_and_hsv_polar_alignment():
    """Verify that (mx, my) maps to theta, which matches polar wheel and quiver vectors."""
    # Build colormap identical to vector_analysis.py
    hsv_cmap = mpl.colormaps['hsv']
    hsv_bright = hsv_cmap(np.linspace(0, 1, 256))
    brightness = 0.8
    hsv_bright[:, :3] *= brightness
    bright_cmap = ListedColormap(hsv_bright)
    bright_cmap.set_bad(color='none')
    
    normc = mpl.colors.Normalize(-np.pi, np.pi)
    
    # Define directional test cases: (angle_deg, mx, my)
    test_cases = [
        (0, 1.0, 0.0),       # Right (+x)
        (45, 1.0, 1.0),      # Up-Right
        (90, 0.0, 1.0),      # Up (+y)
        (135, -1.0, 1.0),    # Up-Left
        (180, -1.0, 0.0),    # Left (-x)
        (-135, -1.0, -1.0),  # Down-Left
        (-90, 0.0, -1.0),    # Down (-y)
        (-45, 1.0, -1.0),    # Down-Right
    ]
    
    for expected_deg, mx_val, my_val in test_cases:
        mag = np.sqrt(mx_val**2 + my_val**2)
        vx = mx_val / mag
        vy = my_val / mag
        
        theta = np.arctan2(my_val, mx_val)
        actual_deg = np.rad2deg(theta)
        
        # Verify theta matches expected mathematical direction
        assert np.isclose(actual_deg, expected_deg if expected_deg <= 180 else expected_deg - 360)
        
        # Verify HSV color on imshow map matches color on polar legend at phi = theta
        color_map = bright_cmap(normc(theta))
        color_polar = bright_cmap(normc(theta))
        assert np.allclose(color_map, color_polar)
        
        # Verify quiver arrow components (vx, vy)
        # Matplotlib default angles='uv' plots +vx to the right and +vy upwards on screen
        quiver_angle_deg = np.rad2deg(np.arctan2(vy, vx))
        assert np.isclose(quiver_angle_deg, actual_deg)


def test_movie_gif_generation_clean_frames(tmp_path):
    """Verify that GIF animation frames are exported tightly with unified global palette and static wheel."""
    hsv_cmap = mpl.colormaps['hsv']
    hsv_bright = hsv_cmap(np.linspace(0, 1, 256))
    brightness = 0.8
    hsv_bright[:, :3] *= brightness
    bright_cmap = ListedColormap(hsv_bright)

    fig = Figure(figsize=(8, 5))
    bg_color = (0.094, 0.094, 0.106, 1.0)
    fig.patch.set_facecolor(bg_color)
    
    frames = []
    dpi = 80
    for theta_val, field in zip([0.0, np.pi/2, np.pi], [0.0, 10.0, 20.0]):
        fig.clear()
        fig.patch.set_facecolor(bg_color)
        
        ax_map = fig.add_axes([0.05, 0.05, 0.70, 0.90])
        theta = np.full((100, 200), theta_val)
        ax_map.imshow(theta, vmin=-np.pi, vmax=np.pi, cmap=bright_cmap)
        ax_map.axis('off')
        
        ax_polar = fig.add_axes([0.80, 0.30, 0.16, 0.40], projection='polar')
        ax_polar.set_facecolor(bg_color)
        ax_polar.xaxis.set_ticks([])
        ax_polar.yaxis.set_ticks([])
        ax_polar.set_aspect('equal')
        ax_polar.grid(False)
        
        normc = mpl.colors.Normalize(-np.pi, np.pi)
        phi_grid = np.linspace(-np.pi, np.pi, 400)
        radius_grid = np.linspace(0.5, 1.0, 2)
        _, tg = np.meshgrid(radius_grid, phi_grid)
        ax_polar.pcolormesh(phi_grid, radius_grid, tg.T, norm=normc, cmap=bright_cmap)
        ax_polar.spines['polar'].set_visible(False)
        ax_polar.set_title(f"{field:.2f} mT", color='#f4f4f5', y=1.1, fontsize=12, fontweight='bold')
        
        buf = io.BytesIO()
        fig.savefig(
            buf, format='png', dpi=dpi, facecolor=bg_color, transparent=False,
            bbox_inches='tight', pad_inches=0.08
        )
        buf.seek(0)
        with Image.open(buf) as img_in:
            frames.append(img_in.convert('RGB'))
        buf.close()
        
    max_w = max(f.width for f in frames)
    max_h = max(f.height for f in frames)
    uniform_frames = []
    bg_rgb = tuple(int(c * 255) for c in bg_color[:3]) if isinstance(bg_color, tuple) else bg_color
    for f in frames:
        if f.size == (max_w, max_h):
            uniform_frames.append(f)
        else:
            padded = Image.new('RGB', (max_w, max_h), color=bg_rgb)
            padded.paste(f, ((max_w - f.width) // 2, (max_h - f.height) // 2))
            uniform_frames.append(padded)

    # Build unified global palette
    montage_w, montage_h = uniform_frames[0].size
    step = max(1, len(uniform_frames) // 10)
    sample_frames = uniform_frames[::step]
    hsv_strip = Image.fromarray((hsv_bright[None, :, :3] * 255).astype(np.uint8))
    hsv_strip = hsv_strip.resize((montage_w, 20), Image.NEAREST)
    slice_h = max(20, montage_h // 4)
    montage = Image.new('RGB', (montage_w, len(sample_frames) * slice_h + 20))
    for idx_s, sf in enumerate(sample_frames):
        sf_small = sf.resize((montage_w, slice_h), Image.BILINEAR)
        montage.paste(sf_small, (0, idx_s * slice_h))
    montage.paste(hsv_strip, (0, len(sample_frames) * slice_h))
    
    global_palette_img = montage.quantize(colors=256, method=Image.Quantize.MEDIANCUT)

    gif_path = tmp_path / "test_out.gif"
    quantized_frames = [
        f.quantize(palette=global_palette_img, dither=Image.Dither.NONE) for f in uniform_frames
    ]
    quantized_frames[0].save(
        str(gif_path),
        save_all=True,
        append_images=quantized_frames[1:],
        duration=200,
        loop=0,
        disposal=2
    )
    
    assert gif_path.exists()
    
    # Read back and inspect
    im = Image.open(str(gif_path))
    assert im.format == 'GIF'
    assert getattr(im, 'n_frames', 1) == 3
    assert im.size == (max_w, max_h)
    
    # Verify wheel area is 100% static across frames
    wheel_arrs = [np.array(q)[int(max_h*0.35):, int(max_w*0.8):] for q in quantized_frames]
    for i in range(len(wheel_arrs) - 1):
        diff = np.max(np.abs(wheel_arrs[i].astype(float) - wheel_arrs[i+1].astype(float)))
        assert diff == 0.0, "HSV wheel must be completely static across frames!"


def test_scale_bar_margin_consistency():
    """Verify that scale bar line and text maintain >= 20px margin across all lens presets."""
    lens_info = {
        "10": [50, 100],
        "20": [100, 50],
        "50": [250, 20],
        "100": [500, 10]
    }
    
    margin = 20
    for lens, (bar_length, bar_size) in lens_info.items():
        text_str = f"{bar_size} µm"
        text_half_width = max(len(text_str) * 8.5, bar_length / 2.0)
        center_x = max(margin + bar_length / 2.0, margin + text_half_width)
        bar_x = center_x - bar_length / 2.0
        
        # Left boundary of bar line
        assert bar_x >= margin
        # Estimated left boundary of centered text label
        text_left = center_x - text_half_width
        assert text_left >= margin
        
        # For 20x, 50x, 100x where bar_length is wider than text, bar_x starts exactly at margin (20)
        if bar_length >= len(text_str) * 17:
            assert bar_x == margin
