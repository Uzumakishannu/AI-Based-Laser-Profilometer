import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


# Change this path when you want to run a different single video.
# Example: INPUT_VIDEO_PATH = r"your_video_path_here.mp4"
INPUT_VIDEO_PATH = r"C:\Users\shann\Downloads\laservideo1.mp4"

# Responsive layout constraints
MAX_DASHBOARD_W = 1600
MAX_DASHBOARD_H = 900
MIN_PANEL_W = 300

# Persistent Settings
CONFIG_PATH = Path(__file__).resolve().parent / "profiler_config.json"
GLOBAL_SENSITIVITY = 1.0


def load_settings():
    global GLOBAL_SENSITIVITY
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            GLOBAL_SENSITIVITY = max(0.1, min(5.0, float(data.get("sensitivity", 1.0))))
            print(f"Loaded sensitivity: {GLOBAL_SENSITIVITY:.2f}")
        except Exception as e:
            print(f"Error loading config: {e}")


def save_settings():
    try:
        data = {"sensitivity": round(GLOBAL_SENSITIVITY, 2)}
        CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"Error saving config: {e}")


def get_threshold(base_val):
    """Adjusts a threshold based on global sensitivity. 
    Higher sensitivity = lower threshold (more detections)."""
    return max(0.01, min(0.99, base_val / max(0.01, GLOBAL_SENSITIVITY)))


load_settings()


def find_laser(frame):
    r = frame[:, :, 2].astype(np.float32)
    g = frame[:, :, 1].astype(np.float32)
    b = frame[:, :, 0].astype(np.float32)

    red_laser = np.clip(r - (g + b) * 0.5, 0, 255)
    col_sums = red_laser.sum(axis=0)
    laser_x = int(np.argmax(col_sums))

    x1 = max(0, laser_x - 2)
    x2 = min(frame.shape[1], laser_x + 3)
    profile = red_laser[:, x1:x2].max(axis=1)
    quality = np.clip(profile / 255.0, 0, 1)
    return profile, quality


def zscore_map(surface):
    mean = surface.mean(axis=1, keepdims=True)
    std = surface.std(axis=1, keepdims=True) + 1e-6
    return (surface - mean) / std


def normalize01(data):
    data = data.astype(np.float32)
    if data.size == 0:
        return data
    min_v = float(np.nanmin(data))
    max_v = float(np.nanmax(data))
    if max_v <= min_v + 1e-6:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - min_v) / (max_v - min_v), 0, 1)


def threshold_response(data, percentile=88, floor=0.10, scale=0.60):
    response = normalize01(data)
    values = response[response > 0]
    if values.size == 0:
        return response
    
    # Adjust thresholds based on sensitivity
    adj_percentile = max(10, min(99, percentile - (GLOBAL_SENSITIVITY - 1.0) * 15))
    adj_floor = max(0.01, min(0.9, floor / max(0.1, GLOBAL_SENSITIVITY)))
    
    cutoff = max(adj_floor, float(np.percentile(values, adj_percentile)) * scale)
    response[response < cutoff] = 0
    return normalize01(response)


def component_filter(
    score,
    min_area=6,
    max_area_ratio=0.06,
    min_width=2,
    min_height=2,
    max_aspect=None,
    max_width_ratio=None,
    max_height_ratio=None,
    reject_border_y_ratio=0.0,
    threshold=0.20,
    close_kernel=(5, 5),
    open_kernel=None,
):
    score = normalize01(score)
    if score.size == 0:
        return score

    mask = (score > get_threshold(threshold)).astype(np.uint8)
    if close_kernel:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if open_kernel:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    filled = np.zeros_like(score, dtype=np.float32)
    max_area = max(min_area, int(score.size * max_area_ratio))

    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < min_area or area > max_area:
            continue
        if w < min_width or h < min_height:
            continue
        if max_aspect is not None and max(w / max(h, 1), h / max(w, 1)) > max_aspect:
            continue
        if max_width_ratio is not None and w > score.shape[1] * max_width_ratio:
            continue
        if max_height_ratio is not None and h > score.shape[0] * max_height_ratio:
            continue
        if reject_border_y_ratio > 0:
            margin = int(score.shape[0] * reject_border_y_ratio)
            if y <= margin or y + h >= score.shape[0] - margin:
                continue

        region = labels == label
        strength = float(score[region].max())
        # The 0.35 base threshold for 'filled' is adjusted by sensitivity
        filled[region] = np.maximum(filled[region], max(strength * 0.72, get_threshold(0.35)))

    return normalize01(filled)


def fill_contour(mask_shape, contour):
    mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    return mask > 0


def filter_elongated_regions(
    score,
    threshold=0.30,
    min_area=5,
    max_area_ratio=0.012,
    min_aspect=5.0,
    max_short_axis_ratio=0.035,
):
    score = normalize01(score)
    mask = (score > get_threshold(threshold)).astype(np.uint8) * 255
    if mask.size == 0:
        return score

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered = np.zeros_like(score, dtype=np.float32)
    max_area = max(min_area, int(score.size * max_area_ratio))
    max_short_axis = max(3.0, min(score.shape) * max_short_axis_ratio)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        (_cx, _cy), (w, h), _angle = cv2.minAreaRect(contour)
        long_axis = max(w, h)
        short_axis = max(1.0, min(w, h))
        aspect = long_axis / short_axis
        if aspect < min_aspect or short_axis > max_short_axis:
            continue

        region = fill_contour(score.shape, contour)
        if not np.any(region):
            continue
        filtered[region] = np.maximum(filtered[region], score[region])

    return normalize01(filtered)


def filter_rounded_peak_regions(
    score,
    peak_source,
    threshold=0.36,
    min_area=6,
    max_area_ratio=0.018,
    min_circularity=0.45,
    max_aspect=1.9,
    min_peak=0.34,
):
    score = normalize01(score)
    peak_source = normalize01(peak_source)
    mask = (score > get_threshold(threshold)).astype(np.uint8) * 255
    if mask.size == 0:
        return score

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered = np.zeros_like(score, dtype=np.float32)
    max_area = max(min_area, int(score.size * max_area_ratio))

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 1e-6:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)

        x, y, w, h = cv2.boundingRect(contour)
        aspect = max(w / max(h, 1), h / max(w, 1))
        if circularity < min_circularity or aspect > max_aspect:
            continue
        if y <= 2 or y + h >= score.shape[0] - 2:
            continue

        region = fill_contour(score.shape, contour)
        peak = float(peak_source[region].max()) if np.any(region) else 0.0
        if peak < min_peak:
            continue
        filtered[region] = np.maximum(filtered[region], np.maximum(score[region], peak * 0.75))

    return normalize01(filtered)


def sheet_boundary_band(surface, quality_map, band_kernel=(17, 17)):
    sheet_score = normalize01(surface.astype(np.float32)) * 0.55
    sheet_score += normalize01(quality_map.astype(np.float32)) * 0.45
    sheet_score = cv2.GaussianBlur(sheet_score, (9, 9), 0)

    values = sheet_score[sheet_score > 0]
    if values.size == 0:
        return np.zeros_like(sheet_score, dtype=np.float32)

    threshold = max(0.08, float(np.percentile(values, 35)) * 0.80)
    sheet_mask = (sheet_score > threshold).astype(np.uint8)
    sheet_mask = cv2.morphologyEx(sheet_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)))

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(sheet_mask, 8)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        sheet_mask = (labels == largest).astype(np.uint8)

    gradient = cv2.morphologyEx(sheet_mask, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, band_kernel))
    return gradient.astype(np.float32)


def detect_missing_coating(_z_map, quality_map):
    low_q = 1.0 - quality_map.astype(np.float32)
    low_q_sm = cv2.GaussianBlur(low_q, (5, 5), 0)
    # Sensitivity directly boosts the visibility of low-quality regions
    return np.clip(low_q_sm * 2.0 * GLOBAL_SENSITIVITY, 0, 1)


def detect_scratches(surface, quality_map=None):
    surface = surface.astype(np.float32)
    blur = cv2.GaussianBlur(surface, (3, 3), 0)
    grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    local_background = gaussian_filter(surface, sigma=(7, 11))
    surface_cut = np.abs(surface - local_background)
    profile_lines = threshold_response(
        np.abs(grad_y) * 0.62 + np.abs(grad_x) * 0.26 + surface_cut * 0.12,
        percentile=91,
        floor=0.08,
        scale=0.58,
    )

    if quality_map is not None:
        low_q = normalize01(1.0 - quality_map.astype(np.float32))
        broad_low_q = cv2.GaussianBlur(low_q, (27, 27), 0)
        narrow_low_q = np.clip(low_q - broad_low_q, 0, None)

        low_q_u8 = to_uint8(low_q)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 23))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 1))
        vertical_lines = cv2.morphologyEx(low_q_u8, cv2.MORPH_TOPHAT, vertical_kernel).astype(np.float32) / 255.0
        horizontal_lines = cv2.morphologyEx(low_q_u8, cv2.MORPH_TOPHAT, horizontal_kernel).astype(np.float32) / 255.0
        quality_lines = threshold_response(
            np.maximum(np.maximum(vertical_lines, horizontal_lines), narrow_low_q),
            percentile=90,
            floor=0.07,
            scale=0.55,
        )
        scratches = np.maximum(profile_lines, quality_lines)
    else:
        scratches = profile_lines

    scratches = filter_elongated_regions(
        scratches,
        threshold=0.28,
        min_area=5,
        max_area_ratio=0.010,
        min_aspect=5.5,
        max_short_axis_ratio=0.030,
    )
    return normalize01(cv2.GaussianBlur(scratches, (3, 3), 0))


def detect_irregular_edges(surface, quality_map, missing_map):
    surface = surface.astype(np.float32)
    boundary_band = sheet_boundary_band(surface, quality_map)
    if boundary_band.max() <= 0:
        return boundary_band

    quality = normalize01(quality_map.astype(np.float32))
    low_q = normalize01(1.0 - quality)
    qx = cv2.Sobel(quality, cv2.CV_32F, 1, 0, ksize=3)
    qy = cv2.Sobel(quality, cv2.CV_32F, 0, 1, ksize=3)
    quality_edge = np.sqrt(qx * qx + qy * qy)

    smooth = cv2.GaussianBlur(surface, (5, 5), 0)
    sx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    profile_edge = np.sqrt(sx * sx + sy * sy)

    missing_mask = (missing_map > 0.32).astype(np.uint8)
    boundary_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    missing_boundary = cv2.morphologyEx(missing_mask, cv2.MORPH_GRADIENT, boundary_kernel).astype(np.float32)

    jagged = cv2.Laplacian(cv2.GaussianBlur(boundary_band, (5, 5), 0), cv2.CV_32F)
    boundary_only = cv2.dilate(boundary_band.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    edge_score = (
        normalize01(quality_edge) * 0.42
        + normalize01(profile_edge) * 0.30
        + missing_boundary * 0.35
        + normalize01(np.abs(jagged)) * 0.35
    )
    edge_score *= boundary_only.astype(np.float32)
    edge_score = threshold_response(edge_score, percentile=72, floor=0.05, scale=0.50)

    edge_mask = (edge_score > 0.16).astype(np.uint8)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 5))
    edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, edge_kernel)
    edge_mask *= boundary_only.astype(np.uint8)
    return normalize01(np.maximum(edge_score, edge_mask.astype(np.float32) * 0.70))


def detect_air_bubbles(surface, z_map, quality_map=None):
    surface = surface.astype(np.float32)
    z = z_map.astype(np.float32)
    background = gaussian_filter(surface, sigma=(13, 17))
    raised_profile = np.clip(surface - background, 0, None)
    raised_z = np.clip(z - 0.85, 0, None)

    local_peak = cv2.GaussianBlur(normalize01(raised_profile), (5, 5), 0)
    peak_core = cv2.dilate(local_peak, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    peak_core = (local_peak >= peak_core - 1e-4).astype(np.float32) * local_peak
    bubble_score = normalize01(raised_profile) * 0.50 + normalize01(raised_z) * 0.38 + peak_core * 0.28

    if quality_map is not None:
        high_quality_peak = cv2.GaussianBlur(normalize01(quality_map.astype(np.float32)), (7, 7), 0)
        bubble_score *= np.clip(high_quality_peak * 1.15, 0.40, 1.10)
        boundary_band = sheet_boundary_band(surface, quality_map, band_kernel=(21, 21))
        bubble_score *= 1.0 - np.clip(cv2.dilate(boundary_band.astype(np.uint8),
                                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))), 0, 1)

    bubble_score = cv2.GaussianBlur(bubble_score, (5, 5), 0)
    bubble_score = threshold_response(bubble_score, percentile=90, floor=0.12, scale=0.62)
    bubbles = filter_rounded_peak_regions(
        bubble_score,
        peak_source=raised_profile,
        threshold=0.32,
        min_area=6,
        max_area_ratio=0.014,
        min_circularity=0.42,
        max_aspect=1.85,
        min_peak=0.32,
    )
    return normalize01(cv2.GaussianBlur(bubbles, (5, 5), 0))


def detect_texture(surface, z_map=None, quality_map=None):
    surface = surface.astype(np.float32)
    mean = gaussian_filter(surface, sigma=(3, 3))
    mean_sq = gaussian_filter(surface * surface, sigma=(3, 3))
    variance = np.maximum(mean_sq - mean * mean, 0)
    roughness = np.sqrt(variance)
    roughness = cv2.GaussianBlur(roughness, (5, 5), 0)
    texture = normalize01(roughness)

    if z_map is not None:
        pits = np.clip(-z_map.astype(np.float32) - 0.68, 0, None)
        if quality_map is not None:
            low_q = normalize01(1.0 - quality_map.astype(np.float32))
            small_dark_dips = np.clip(low_q - cv2.GaussianBlur(low_q, (17, 17), 0), 0, None)
            pits = np.maximum(normalize01(pits), normalize01(small_dark_dips))

        pits = component_filter(
            pits,
            min_area=3,
            max_area_ratio=0.025,
            min_width=1,
            min_height=1,
            max_aspect=4.0,
            threshold=0.14,
            close_kernel=(5, 5),
            open_kernel=None,
        )
        texture = np.maximum(texture, pits * 0.95)

    return normalize01(texture)


def apply_green(gray):
    color = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    color[:, :, 1] = gray
    return color


def to_uint8(data):
    return cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def apply_missing_color(gray):
    return cv2.applyColorMap(gray, cv2.COLORMAP_OCEAN)


def apply_scratch_color(gray):
    return cv2.applyColorMap(gray, cv2.COLORMAP_WINTER)


def apply_irregular_color(gray):
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


def apply_bubble_color(gray):
    return cv2.applyColorMap(gray, cv2.COLORMAP_SUMMER)


def apply_texture_color(gray):
    return cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA)


def apply_depth_color(gray):
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


def resize_with_padding(img, target_w, target_h):
    """Resizes an image maintaining aspect ratio and adds black padding."""
    if img is None or img.size == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    if len(img_resized.shape) == 2:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    dx = (target_w - nw) // 2
    dy = (target_h - nh) // 2
    canvas[dy : dy + nh, dx : dx + nw] = img_resized
    return canvas


def draw_header(panel, title, active=False):
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 40), (18, 23, 31), -1)
    color = (80, 220, 255) if active else (230, 235, 240)
    cv2.putText(panel, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
    cv2.putText(
        panel,
        "ACTIVE" if active else "LIVE",
        (panel.shape[1] - 100, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        1,
        cv2.LINE_AA,
    )


def to_panel(data, title, width, height, mode="green", active=False):
    norm = to_uint8(data)
    if mode == "green":
        color = apply_green(norm)
    elif mode == "missing":
        color = apply_missing_color(norm)
    elif mode == "scratch":
        color = apply_scratch_color(norm)
    elif mode == "irregular":
        color = apply_irregular_color(norm)
    elif mode == "bubble":
        color = apply_bubble_color(norm)
    elif mode == "texture":
        color = apply_texture_color(norm)
    else:
        color = apply_depth_color(norm)

    # Physical accuracy scaling: Maintain 1:1 aspect ratio of the data.
    h, w = color.shape[:2]
    scale = height / max(1, h)
    nw = max(1, int(w * scale))

    panel = np.zeros((height, width, 3), dtype=np.uint8)
    if nw <= width:
        # Fits or needs padding on right
        img_resized = cv2.resize(color, (nw, height), interpolation=cv2.INTER_AREA)
        panel[:, :nw] = img_resized
    else:
        # Too wide, show latest (scroll)
        start_x = w - int(width / scale)
        color_crop = color[:, max(0, start_x) :]
        img_resized = cv2.resize(color_crop, (width, height), interpolation=cv2.INTER_AREA)
        panel[:] = img_resized

    panel = cv2.addWeighted(panel, 0.88, np.full_like(panel, (8, 10, 14)), 0.12, 0)
    for x in range(0, width, max(1, width // 6)):
        cv2.line(panel, (x, 40), (x, height - 1), (24, 30, 38), 1)
    for y in range(40, height, max(1, (height - 40) // 4)):
        cv2.line(panel, (0, y), (width - 1, y), (24, 30, 38), 1)
    draw_header(panel, title, active)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (55, 65, 78), 1)
    return panel


def pseudo_3d_surface_panel(surface, width, height):
    if surface.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    smooth = gaussian_filter(surface.astype(np.float32), sigma=(2, 2))
    norm = to_uint8(smooth)
    depth = apply_depth_color(norm)

    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    shade = np.clip(0.75 - gx * 0.012 - gy * 0.006, 0.35, 1.25)
    shaded = np.clip(depth.astype(np.float32) * shade[..., None], 0, 255).astype(np.uint8)

    # Physical accuracy resize for the background
    h, w = shaded.shape[:2]
    scale = height / max(1, h)
    nw = max(1, int(w * scale))

    panel = np.zeros((height, width, 3), dtype=np.uint8)
    if nw <= width:
        img_resized = cv2.resize(shaded, (nw, height), interpolation=cv2.INTER_AREA)
        panel[:, :nw] = img_resized
    else:
        start_x = w - int(width / scale)
        img_resized = cv2.resize(shaded[:, max(0, start_x) :], (width, height), interpolation=cv2.INTER_AREA)
        panel[:] = img_resized

    # Mesh-like profiler lines (adjusted for scaled height)
    mesh_scale = height / 700.0
    for y in range(55, height, max(14, height // 12)):
        row_idx = int((y / max(1, height - 1)) * (surface.shape[0] - 1))
        pts = []
        for x in range(0, width, 5):
            col_idx = int((x / max(1, width - 1)) * (surface.shape[1] - 1))
            z = smooth[row_idx, col_idx] if smooth.size else 0
            yy = int(y - (z - np.mean(smooth)) * 0.025 * mesh_scale)
            if 40 <= yy < height:
                pts.append((x, yy))
        if len(pts) > 1:
            cv2.polylines(panel, [np.array(pts, dtype=np.int32)], False, (235, 245, 255), 1, cv2.LINE_AA)

    draw_header(panel, "3D SURFACE RECONSTRUCTION", active=True)
    cv2.putText(panel, "relative depth", (12, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4 * mesh_scale, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (55, 65, 78), 1)
    return panel


def save_interactive_surface_html(surface, output_html):
    if surface.size == 0:
        return
    step_y = max(1, surface.shape[0] // 120)
    step_x = max(1, surface.shape[1] // 160)
    z = surface[::step_y, ::step_x].astype(np.float32)
    z = gaussian_filter(z, sigma=(1.2, 1.2))
    z = cv2.normalize(z, None, -12, 12, cv2.NORM_MINMAX)
    z_list = [[round(float(v), 3) for v in row] for row in z]

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Interactive 3D Laser Surface</title>
<style>
body {{ margin:0; background:#080b10; color:#eef3f8; font-family:Arial,sans-serif; }}
header {{ height:64px; padding:12px 18px; box-sizing:border-box; background:#141a22; border-bottom:1px solid #313a48; }}
h1 {{ margin:0; font-size:19px; }}
p {{ margin:5px 0 0; color:#aeb8c6; font-size:13px; }}
canvas {{ display:block; width:100vw; height:calc(100vh - 64px); cursor:grab; }}
canvas:active {{ cursor:grabbing; }}
</style>
</head>
<body>
<header>
<h1>Interactive 3D Surface Reconstruction</h1>
<p>Drag to rotate, mouse wheel to zoom. Relative depth from red laser profile.</p>
</header>
<canvas id="c"></canvas>
<script>
const Z = {json.dumps(z_list)};
const rows = Z.length, cols = Z[0].length;
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
let rx = -0.95, rz = -0.72, zoom = 1.0, drag = false, lx = 0, ly = 0;

function resize() {{
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(canvas.clientWidth * dpr);
  canvas.height = Math.floor(canvas.clientHeight * dpr);
  draw();
}}
function color(z, light) {{
  const t = Math.max(0, Math.min(1, (z + 12) / 24));
  const r = Math.floor((255 * Math.min(1, Math.max(0, 1.7*t - .15))) * light);
  const g = Math.floor((255 * Math.sin(Math.PI*t)) * light);
  const b = Math.floor((255 * Math.min(1, Math.max(0, 1.7*(1-t) - .15))) * light);
  return `rgb(${{r}},${{g}},${{b}})`;
}}
function project(x, y, z) {{
  x = (x / (cols - 1) - .5) * 3.2;
  y = (y / (rows - 1) - .5) * 2.0;
  z = z / 8.5;
  const cz = Math.cos(rz), sz = Math.sin(rz), cx = Math.cos(rx), sx = Math.sin(rx);
  const x1 = x*cz - y*sz;
  const y1 = x*sz + y*cz;
  const y2 = y1*cx - z*sx;
  const scale = Math.min(canvas.width, canvas.height) * .33 * zoom;
  return [canvas.width/2 + x1*scale, canvas.height/2 + y2*scale];
}}
function draw() {{
  ctx.fillStyle = '#080b10';
  ctx.fillRect(0,0,canvas.width,canvas.height);
  for (let y = rows - 2; y >= 0; y--) {{
    for (let x = 0; x < cols - 1; x++) {{
      const z00 = Z[y][x], z10 = Z[y][x+1], z11 = Z[y+1][x+1], z01 = Z[y+1][x];
      const avg = (z00 + z10 + z11 + z01) / 4;
      const light = Math.max(.45, Math.min(1.25, .82 + (z10-z00)*.018 + (z01-z00)*.012));
      const p0 = project(x,y,z00), p1 = project(x+1,y,z10), p2 = project(x+1,y+1,z11), p3 = project(x,y+1,z01);
      ctx.beginPath(); ctx.moveTo(p0[0],p0[1]); ctx.lineTo(p1[0],p1[1]); ctx.lineTo(p2[0],p2[1]); ctx.lineTo(p3[0],p3[1]); ctx.closePath();
      ctx.fillStyle = color(avg, light); ctx.fill();
    }}
  }}
  ctx.strokeStyle = 'rgba(245,250,255,.16)';
  ctx.lineWidth = Math.max(1, window.devicePixelRatio || 1);
  for (let y=0; y<rows; y+=4) {{
    ctx.beginPath();
    for (let x=0; x<cols; x++) {{
      const p = project(x,y,Z[y][x]);
      if (x===0) ctx.moveTo(p[0],p[1]); else ctx.lineTo(p[0],p[1]);
    }}
    ctx.stroke();
  }}
}}
canvas.addEventListener('mousedown', e => {{ drag = true; lx = e.clientX; ly = e.clientY; }});
window.addEventListener('mouseup', () => drag = false);
window.addEventListener('mousemove', e => {{
  if (!drag) return;
  rz += (e.clientX - lx) * .008;
  rx += (e.clientY - ly) * .008;
  lx = e.clientX; ly = e.clientY; draw();
}});
canvas.addEventListener('wheel', e => {{
  e.preventDefault();
  zoom *= e.deltaY < 0 ? 1.08 : .92;
  zoom = Math.max(.35, Math.min(3.0, zoom));
  draw();
}}, {{passive:false}});
window.addEventListener('resize', resize);
resize();
</script>
</body>
</html>"""
    Path(output_html).write_text(html, encoding="utf-8")


def default_input_path():
    candidates = [
        INPUT_VIDEO_PATH,
        r"C:\Users\shann\Downloads\profilometer\WhatsApp Video 2026-05-08 at 5.13.31 PM.mp4",
        r"C:\Users\shann\Downloads\WhatsApp Video 2026-05-08 at 5.13.31 PM.mp4",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def build_canvas(frame, raw_map, missing_map, scratch_map, edge_map, bubble_map, texture_map, layout):
    win_w, win_h = layout["win_w"], layout["win_h"]
    live_w = layout["live_w"]
    panel_w, panel_h = layout["panel_w"], layout["panel_h"]

    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    canvas[:] = (8, 10, 14)

    live = resize_with_padding(frame, live_w, win_h)
    canvas[:, :live_w] = live
    cv2.rectangle(canvas, (0, 0), (live_w - 1, win_h - 1), (70, 82, 96), 2)
    cv2.rectangle(canvas, (0, 0), (live_w, 42), (18, 23, 31), -1)
    cv2.putText(canvas, "ORIGINAL LIVE VIDEO", (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (245, 248, 250), 2, cv2.LINE_AA)
    
    sens_text = f"SENSITIVITY: {GLOBAL_SENSITIVITY:.2f}  [+/- to adjust]"
    cv2.putText(canvas, sens_text, (live_w - 360, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (120, 200, 255), 1, cv2.LINE_AA)

    # Determine active status based on sensitivity-adjusted thresholds
    t_missing = get_threshold(0.35)
    t_scratch = get_threshold(0.35)
    t_edge = get_threshold(0.35)
    t_bubble = get_threshold(0.35)
    t_texture = get_threshold(0.40)

    panels = [
        to_panel(missing_map, "MISSING COATING", panel_w, panel_h, "missing", active=missing_map.max() > t_missing),
        to_panel(scratch_map, "SCRATCHES", panel_w, panel_h, "scratch", active=scratch_map.max() > t_scratch),
        to_panel(edge_map, "IRREGULAR EDGES", panel_w, panel_h, "irregular", active=edge_map.max() > t_edge),
        to_panel(bubble_map, "AIR BUBBLES", panel_w, panel_h, "bubble", active=bubble_map.max() > t_bubble),
        to_panel(texture_map, "TEXTURE ANALYSIS", panel_w, panel_h, "texture", active=texture_map.max() > t_texture),
        pseudo_3d_surface_panel(raw_map, panel_w, panel_h),
    ]

    for i, panel in enumerate(panels):
        col = i % 3
        row = i // 3
        x = live_w + col * panel_w
        y = row * panel_h
        canvas[y : y + panel_h, x : x + panel_w] = panel

    return canvas


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def collect_input_videos(input_path, input_dir):
    if input_dir:
        root = Path(input_dir)
        videos = [path for path in root.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS]
        return sorted(videos)
    return [Path(input_path or default_input_path())]


def resolve_output_path(input_path, output_arg, output_dir, batch_mode):
    if output_arg and not batch_mode:
        return Path(output_arg)
    return Path(output_dir) / f"{Path(input_path).stem}_industrial_grid_ui.mp4"


def resolve_html_path(input_path, output_path, html_output, save_html, batch_mode):
    if html_output and not batch_mode:
        return Path(html_output)
    if save_html:
        return output_path.with_name(f"{Path(input_path).stem}_interactive_3d_surface.html")
    return None


def summarize_defect_map(data, threshold=0.25):
    if data is None or data.size == 0:
        return {"max": 0.0, "mean": 0.0, "coverage_percent": 0.0}
    return {
        "max": round(float(np.max(data)), 4),
        "mean": round(float(np.mean(data)), 4),
        "coverage_percent": round(float(np.mean(data > threshold) * 100.0), 3),
    }


def process_video(input_path, output_path, html_output=None, display=False, panel_update_interval=8, max_frames=0):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Video not found: {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if frame_h <= 0 or frame_w <= 0:
        raise RuntimeError(f"Could not read video: {input_path}")

    # Initial target sizes
    target_h = 800
    panel_h = target_h // 2
    panel_w = int(panel_h * 1.33)  # Standard aspect for panels
    live_w = int(target_h * (frame_w / frame_h))

    # Responsive down-scaling if it exceeds screen limits
    total_w = live_w + (3 * panel_w)
    if total_w > MAX_DASHBOARD_W:
        scale = MAX_DASHBOARD_W / total_w
        target_h = int(target_h * scale)
        live_w = int(live_w * scale)
        panel_w = int(panel_w * scale)
        panel_h = target_h // 2

    if target_h > MAX_DASHBOARD_H:
        scale = MAX_DASHBOARD_H / target_h
        target_h = MAX_DASHBOARD_H
        live_w = int(live_w * scale)
        panel_w = int(panel_w * scale)
        panel_h = target_h // 2

    win_w = live_w + (3 * panel_w)
    win_h = target_h
    layout = {
        "win_w": win_w, "win_h": win_h,
        "live_w": live_w, "panel_w": panel_w, "panel_h": panel_h
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (win_w, win_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")

    alloc_frames = max(total_frames, max_frames, 1)
    surface_map = np.zeros((frame_h, alloc_frames), dtype=np.float32)
    quality_history = np.zeros((frame_h, alloc_frames), dtype=np.float32)
    profile_count = 0
    frame_idx = 0
    cached_panels = None
    final_maps = {}
    progress_total = max_frames if max_frames > 0 else total_frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if profile_count >= surface_map.shape[1]:
            extra_cols = max(300, surface_map.shape[1])
            surface_map = np.pad(surface_map, ((0, 0), (0, extra_cols)), mode="constant")
            quality_history = np.pad(quality_history, ((0, 0), (0, extra_cols)), mode="constant")

        profile, quality = find_laser(frame)
        surface_map[:len(profile), profile_count] = profile
        quality_history[:len(quality), profile_count] = quality
        profile_count += 1
        frame_idx += 1

        raw_map = surface_map[:, :profile_count]
        quality_map = quality_history[:, :profile_count]

        if (
            cached_panels is None
            or frame_idx % max(1, panel_update_interval) == 0
            or frame_idx == total_frames
            or (max_frames > 0 and frame_idx == max_frames)
        ):
            z_map = zscore_map(raw_map)
            missing_map = detect_missing_coating(z_map, quality_map)
            scratch_map = detect_scratches(raw_map, quality_map)
            edge_map = detect_irregular_edges(raw_map, quality_map, missing_map)
            bubble_map = detect_air_bubbles(raw_map, z_map, quality_map)
            texture_map = detect_texture(raw_map, z_map, quality_map)
            cached_panels = (raw_map, missing_map, scratch_map, edge_map, bubble_map, texture_map)
            final_maps = {
                "missing_coating": missing_map,
                "scratches": scratch_map,
                "irregular_edges_peel_off": edge_map,
                "air_bubbles": bubble_map,
                "texture_pits": texture_map,
            }

        canvas = build_canvas(frame, *cached_panels, layout=layout)
        writer.write(canvas)

        if display:
            cv2.imshow("Laser Surface Inspection", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("+") or key == ord("="):
                GLOBAL_SENSITIVITY = min(5.0, GLOBAL_SENSITIVITY + 0.05)
                save_settings()
            elif key == ord("-") or key == ord("_"):
                GLOBAL_SENSITIVITY = max(0.1, GLOBAL_SENSITIVITY - 0.05)
                save_settings()

        if frame_idx == 1 or frame_idx % 100 == 0:
            if progress_total > 0:
                print(f"{input_path.name}: processed {frame_idx}/{progress_total}", flush=True)
            else:
                print(f"{input_path.name}: processed {frame_idx}", flush=True)

        if max_frames > 0 and frame_idx >= max_frames:
            break

    cap.release()
    writer.release()

    if html_output:
        html_output = Path(html_output)
        html_output.parent.mkdir(parents=True, exist_ok=True)
        save_interactive_surface_html(surface_map[:, :profile_count], html_output)

    report = {
        "input_video": str(input_path),
        "output_video": str(output_path),
        "interactive_3d_html": str(html_output) if html_output else "",
        "fps": round(float(fps), 4),
        "frames_processed": int(frame_idx),
        "duration_seconds": round(float(frame_idx / fps), 3) if fps else 0,
        "defect_channels": {
            name: summarize_defect_map(data)
            for name, data in final_maps.items()
        },
        "classification_notes": {
            "missing_coating": "Large low-reflection coating loss or missing paint.",
            "irregular_edges_peel_off": "Torn coating boundary, peel-off skin, edge damage, and jagged delamination.",
            "air_bubbles": "Raised laser-profile peaks and oval swelling regions.",
            "scratches": "Narrow line defects and black line-like laser intensity dips.",
            "texture_pits": "Distributed roughness, small pitting, pin holes, and local depressions.",
        },
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if display:
        cv2.destroyAllWindows()

    print(f"Output video: {output_path}")
    print(f"Analysis report: {report_path}")
    if html_output:
        print(f"Interactive 3D HTML: {html_output}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Industrial grid UI for red-laser surface defect inspection.")
    parser.add_argument("--input", default="", help="Input video path. Use quotes around paths with spaces.")
    parser.add_argument("--input-dir", default="", help="Optional folder path to process all videos inside it.")
    parser.add_argument("--output", default="", help="Output video path for a single input video.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "outputs" / "simple_grid_ui"),
        help="Output folder for automatic naming and batch runs.",
    )
    parser.add_argument("--html-output", default="", help="Optional interactive 3D HTML path for a single input video.")
    parser.add_argument("--save-html", action="store_true", help="Save an interactive 3D HTML surface for each video.")
    parser.add_argument("--display", action="store_true", help="Show live OpenCV window while processing.")
    parser.add_argument(
        "--panel-update-interval",
        type=int,
        default=8,
        help="Refresh the right-side analysis panels every N frames while preserving output FPS.",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Optional quick-test limit. Use 0 for full video.")
    args = parser.parse_args()

    input_videos = collect_input_videos(args.input, args.input_dir)
    if not input_videos:
        raise FileNotFoundError("No video files found.")

    batch_mode = len(input_videos) > 1 or bool(args.input_dir)
    reports = []
    for input_path in input_videos:
        output_path = resolve_output_path(input_path, args.output, args.output_dir, batch_mode)
        html_path = resolve_html_path(input_path, output_path, args.html_output, args.save_html, batch_mode)
        reports.append(
            process_video(
                input_path=input_path,
                output_path=output_path,
                html_output=html_path,
                display=args.display,
                panel_update_interval=args.panel_update_interval,
                max_frames=args.max_frames,
            )
        )

    if len(reports) > 1:
        batch_report = Path(args.output_dir) / "batch_analysis_report.json"
        batch_report.parent.mkdir(parents=True, exist_ok=True)
        batch_report.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"Batch analysis report: {batch_report}")


if __name__ == "__main__":
    main()
