"""2D Aviation Flight HUD GPU draw handler and pure layout calculator."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import blf
import bpy
import gpu
from gpu_extras.batch import batch_for_shader

from .. import constants
from ..state import CACHE, Category, DiagnosticReport, FlightStatus, Severity

_DRAW_HANDLE: Any = None
_SHADER_2D: Any = None

_ColorRGBA = Tuple[float, float, float, float]
_Rect = Tuple[float, float, float, float]
_Point2D = Tuple[float, float]

_STATUS_COLORS: Dict[FlightStatus, _ColorRGBA] = {
    FlightStatus.GO: constants.HUD_COLOR_GO,
    FlightStatus.CAUTION: constants.HUD_COLOR_CAUTION,
    FlightStatus.NO_GO: constants.HUD_COLOR_NO_GO,
}

_SEVERITY_COLORS: Dict[Severity, _ColorRGBA] = {
    Severity.OK: constants.HUD_COLOR_GO,
    Severity.WARNING: constants.HUD_COLOR_CAUTION,
    Severity.CRITICAL: constants.HUD_COLOR_NO_GO,
}

_GAUGE_ORDER = ("ALB", "RGH", "TXL", "SCL")

_VALID_POSITIONS = frozenset({
    "TOP_LEFT",
    "TOP_RIGHT",
    "BOTTOM_LEFT",
    "BOTTOM_RIGHT",
})


def _get_shader_2d() -> Any:
    """Lazily compile and cache the built-in 2D uniform color shader."""
    global _SHADER_2D
    if _SHADER_2D is None:
        _SHADER_2D = gpu.shader.from_builtin("UNIFORM_COLOR")
    return _SHADER_2D


def _build_legend(
    mode_int: int,
    report: Optional[DiagnosticReport],
) -> Optional[Dict[str, Any]]:
    """Build legend title and color stops for the active diagnostic mode."""
    if mode_int == constants.MODE_ALBEDO:
        return {
            "title": "ALBEDO LUM (0.0 -> 1.0)",
            "stops": [
                (constants.COLOR_CRUSHED_BLACK, "0.00 - 0.02  Crushed Black"),
                (constants.COLOR_DARK_METAL, "Metal < 0.45  Dark Metal"),
                (constants.COLOR_SEMI_METAL, "0.05 - 0.95  Semi-Metal"),
                (constants.COLOR_BLOWN_WARN, "0.70 - 0.81  Blown Warn"),
                (constants.COLOR_BLOWN_CRIT, "0.81 - 1.00  Blown Crit"),
            ],
        }
    if mode_int == constants.MODE_ROUGHNESS:
        return {
            "title": "ROUGHNESS / SPECULAR / IOR",
            "stops": [
                (constants.COLOR_MIRROR_ROUGH, "Roughness < 0.02  Mirror"),
                (constants.COLOR_DEAD_ROUGH, "Roughness > 0.98  Dead"),
                (constants.COLOR_HATCH_YELLOW, "Spec IOR outside [0.25, 0.75]"),
                (constants.COLOR_HATCH_RED, "IOR outside [1.30, 1.80]"),
            ],
        }
    if mode_int == constants.MODE_TEXEL_DENSITY:
        target = constants.TEXEL_DEFAULT_TARGET
        if report is not None and report.texel_stats:
            first_stat = next(iter(report.texel_stats.values()))
            if first_stat.target_px_per_m > 0.0:
                target = first_stat.target_px_per_m
        low_px = int(round(target * constants.TEXEL_RATIO_MIN))
        opt_px = int(round(target))
        high_px = int(round(target * constants.TEXEL_RATIO_MAX))
        return {
            "title": f"TEXEL DENSITY ({low_px} -> {high_px} px/m)",
            "stops": [
                (constants.COLOR_TEXEL_LOW, f"< {low_px} px/m  (< 0.5x Low)"),
                (constants.COLOR_TEXEL_OK, f"{opt_px} px/m  (1.0x Target)"),
                (constants.COLOR_TEXEL_HIGH, f"> {high_px} px/m  (> 2.0x High)"),
            ],
        }
    if mode_int == constants.MODE_COMBINED:
        return {
            "title": "COMBINED DIAGNOSTIC KEY",
            "stops": [
                (constants.COLOR_CRUSHED_BLACK, "Crushed Albedo / Low Texel"),
                (constants.COLOR_BLOWN_CRIT, "Blown Albedo / High Texel"),
                (constants.COLOR_SEMI_METAL, "Semi-Metal / Dark Metal"),
                (constants.COLOR_MIRROR_ROUGH, "Roughness / Spec / IOR"),
            ],
        }
    return None


def compute_hud_layout(
    region_w: int,
    region_h: int,
    ui_scale: float,
    position: str,
    report: Optional[DiagnosticReport],
    mode_int: int,
) -> Dict[str, Any]:
    """Compute pure-Python 2D pixel geometry and telemetry data for the Flight HUD."""
    scale = max(float(ui_scale or 0.0), 1.0)
    rw = max(float(region_w), 1.0)
    rh = max(float(region_h), 1.0)
    corner = position if position in _VALID_POSITIONS else "TOP_LEFT"

    pad = 10.0 * scale
    gap = 6.0 * scale
    small_gap = 4.0 * scale
    margin_x = 18.0 * scale
    margin_top = 42.0 * scale
    margin_bottom = 18.0 * scale

    w = 296.0 * scale
    header_h = 24.0 * scale
    meta_h = 14.0 * scale
    gauge_h = 38.0 * scale
    banner_h = 22.0 * scale

    cs_issues = report.issues_by_category(Category.COLORSPACE) if report is not None else []
    cs_count = len(cs_issues)
    has_banner = cs_count > 0

    legend_spec = _build_legend(mode_int, report)
    legend_header_h = 15.0 * scale
    legend_row_h = 13.5 * scale
    legend_pad = 6.0 * scale
    legend_h = 0.0
    if legend_spec is not None:
        legend_h = legend_pad * 2.0 + legend_header_h + len(legend_spec["stops"]) * legend_row_h

    h = pad + header_h + small_gap + meta_h + gap + gauge_h + pad
    if has_banner:
        h += gap + banner_h
    if legend_spec is not None:
        h += gap + legend_h

    if "LEFT" in corner:
        x = margin_x
    else:
        x = max(margin_x, rw - w - margin_x)

    if "BOTTOM" in corner:
        y = margin_bottom
    else:
        y = max(margin_bottom, rh - h - margin_top)

    inner_w = w - 2.0 * pad
    cursor_top = y + h - pad

    # Header & Status Badge
    header_y = cursor_top - header_h
    header_rect: _Rect = (x + pad, header_y, inner_w, header_h)
    badge_w = 82.0 * scale
    badge_h = 20.0 * scale
    badge_x = x + w - pad - badge_w
    badge_y = header_y + (header_h - badge_h) * 0.5
    badge_rect: _Rect = (badge_x, badge_y, badge_w, badge_h)

    if report is None:
        status_label = "READY"
        status_color = constants.HUD_TEXT_MUTED
        counts_text = "NOT SCANNED"
        scan_ms_text = "-- ms"
    else:
        status_enum = report.status
        status_label = status_enum.value
        status_color = _STATUS_COLORS.get(status_enum, constants.HUD_TEXT_MUTED)
        counts_text = f"{report.critical_count} CRIT  {report.warning_count} WARN"
        scan_ms_text = f"{report.scan_duration_ms:.1f}ms"

    mode_name = constants.MODE_INT_TO_NAME.get(mode_int, "OFF")
    meta_y = header_y - small_gap - meta_h
    meta_rect: _Rect = (x + pad, meta_y, inner_w, meta_h)
    meta_text = f"MODE: {mode_name}  |  SCAN: {scan_ms_text}"

    # 4 Telemetry Gauges
    gauges_y = meta_y - gap - gauge_h
    col_gap = 6.0 * scale
    cell_w = (inner_w - 3.0 * col_gap) / 4.0
    bar_margin = 4.0 * scale
    bar_h = 5.0 * scale
    bar_w = max(cell_w - 2.0 * bar_margin, 2.0)

    raw_scores = report.gauge_scores() if report is not None else None
    gauges: List[Dict[str, Any]] = []
    for idx, key in enumerate(_GAUGE_ORDER):
        cell_x = x + pad + idx * (cell_w + col_gap)
        cell_rect: _Rect = (cell_x, gauges_y, cell_w, gauge_h)
        if raw_scores is None:
            sev = Severity.OK
            score = 0.0
            color = constants.HUD_TEXT_MUTED
            pct_text = "--"
        else:
            sev, score = raw_scores[key]
            score = max(0.0, min(100.0, float(score)))
            color = _SEVERITY_COLORS.get(sev, constants.HUD_TEXT_MUTED)
            pct_text = f"{int(round(score))}%"

        bar_x = cell_x + bar_margin
        bar_y = gauges_y + bar_margin
        bar_bg_rect: _Rect = (bar_x, bar_y, bar_w, bar_h)
        fill_w = bar_w * (score / 100.0)
        bar_rect: _Rect = (bar_x, bar_y, fill_w, bar_h)

        gauges.append({
            "key": key,
            "severity": sev.value,
            "color": color,
            "score": score,
            "pct_text": pct_text,
            "cell_rect": cell_rect,
            "bar_bg_rect": bar_bg_rect,
            "bar_rect": bar_rect,
        })

    cursor_y = gauges_y

    # Optional Data-Map Color Space Warning Banner
    banner: Optional[Dict[str, Any]] = None
    if has_banner:
        banner_y = cursor_y - gap - banner_h
        banner_rect: _Rect = (x + pad, banner_y, inner_w, banner_h)
        banner = {
            "rect": banner_rect,
            "text": f"! DATA-MAP COLORSPACE: {cs_count} sRGB ISSUE(S)",
            "color": constants.HUD_BANNER_BG,
            "count": cs_count,
        }
        cursor_y = banner_y

    # Optional Active Mode Legend
    legend: Optional[Dict[str, Any]] = None
    if legend_spec is not None:
        legend_y = cursor_y - gap - legend_h
        legend_rect: _Rect = (x + pad, legend_y, inner_w, legend_h)
        swatch_size = 9.0 * scale
        stop_rows: List[Dict[str, Any]] = []
        row_top = legend_y + legend_h - legend_pad - legend_header_h
        for s_idx, (stop_color, stop_label) in enumerate(legend_spec["stops"]):
            row_y = row_top - (s_idx + 1) * legend_row_h
            swatch_y = row_y + (legend_row_h - swatch_size) * 0.5
            swatch_rect: _Rect = (x + pad + legend_pad, swatch_y, swatch_size, swatch_size)
            stop_rows.append({
                "color": stop_color,
                "label": stop_label,
                "swatch_rect": swatch_rect,
                "text_pos": (x + pad + legend_pad + swatch_size + 6.0 * scale, row_y + 2.5 * scale),
            })

        legend = {
            "title": legend_spec["title"],
            "rect": legend_rect,
            "title_pos": (x + pad + legend_pad, legend_y + legend_h - legend_pad - 11.0 * scale),
            "stops": [(s["color"], s["label"]) for s in stop_rows],
            "rows": stop_rows,
        }

    return {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "scale": scale,
        "position": corner,
        "title": "FLIGHTCHECK",
        "header_rect": header_rect,
        "meta_rect": meta_rect,
        "meta_text": meta_text,
        "status": {
            "label": status_label,
            "color": status_color,
            "counts_text": counts_text,
            "badge_rect": badge_rect,
        },
        "gauges": gauges,
        "banner": banner,
        "legend": legend,
    }


def _append_rect_tris(verts: List[_Point2D], rect: _Rect) -> None:
    """Append two triangles (6 vertices) for a 2D axis-aligned rectangle."""
    rx, ry, rw, rh = rect
    if rw <= 0.0 or rh <= 0.0:
        return
    x1 = rx + rw
    y1 = ry + rh
    verts.extend((
        (rx, ry),
        (x1, ry),
        (x1, y1),
        (rx, ry),
        (x1, y1),
        (rx, y1),
    ))


def _append_rect_outline(verts: List[_Point2D], rect: _Rect) -> None:
    """Append 4 line segments (8 vertices) outlining a 2D rectangle."""
    rx, ry, rw, rh = rect
    if rw <= 0.0 or rh <= 0.0:
        return
    x1 = rx + rw
    y1 = ry + rh
    verts.extend((
        (rx, ry), (x1, ry),
        (x1, ry), (x1, y1),
        (x1, y1), (rx, y1),
        (rx, y1), (rx, ry),
    ))


def _draw_text(
    font_id: int,
    px: float,
    py: float,
    size_pt: float,
    color: _ColorRGBA,
    text: str,
) -> None:
    """Draw 2D screen-space text using Blender 5.0+ 2-argument blf.size()."""
    blf.size(font_id, max(int(round(size_pt)), 8))
    blf.color(font_id, color[0], color[1], color[2], color[3])
    blf.position(font_id, float(px), float(py), 0.0)
    blf.draw(font_id, text)


def draw_hud_2d() -> None:
    """POST_PIXEL draw callback for the FlightCheck 2D Aviation HUD."""
    if bpy.app.background:
        return

    context = bpy.context
    if context is None or context.region is None or context.scene is None:
        return

    settings = getattr(context.scene, "flightcheck", None)
    if settings is None or not settings.enabled or not settings.show_hud_2d:
        return

    t_start = time.perf_counter()
    try:
        shader = _get_shader_2d()
    except Exception:
        return

    wm_state = getattr(context.window_manager, "flightcheck_wm", None)
    position = wm_state.hud_position if wm_state is not None else "TOP_LEFT"
    mode_int = constants.MODE_NAME_TO_INT.get(settings.diagnostic_mode, constants.MODE_OFF)
    ui_scale = max(float(context.preferences.system.ui_scale), 1.0)
    report = CACHE.get_report()

    layout = compute_hud_layout(
        region_w=context.region.width,
        region_h=context.region.height,
        ui_scale=ui_scale,
        position=position,
        report=report,
        mode_int=mode_int,
    )

    scale = layout["scale"]
    color_tris: Dict[_ColorRGBA, List[_Point2D]] = {}

    def _add_rect(color: _ColorRGBA, rect: _Rect) -> None:
        bucket = color_tris.setdefault(color, [])
        _append_rect_tris(bucket, rect)

    # Background and inner panels
    _add_rect(constants.HUD_BG, (layout["x"], layout["y"], layout["w"], layout["h"]))

    status = layout["status"]
    badge_bg = (
        status["color"][0] * 0.25,
        status["color"][1] * 0.25,
        status["color"][2] * 0.25,
        0.92,
    )
    _add_rect(badge_bg, status["badge_rect"])

    bar_bg_color: _ColorRGBA = (0.14, 0.18, 0.24, 0.95)
    for gauge in layout["gauges"]:
        _add_rect(constants.HUD_PANEL_BG, gauge["cell_rect"])
        _add_rect(bar_bg_color, gauge["bar_bg_rect"])
        _add_rect(gauge["color"], gauge["bar_rect"])

    banner = layout["banner"]
    if banner is not None:
        _add_rect(banner["color"], banner["rect"])

    legend = layout["legend"]
    if legend is not None:
        _add_rect(constants.HUD_PANEL_BG, legend["rect"])
        for row in legend["rows"]:
            _add_rect(row["color"], row["swatch_rect"])

    line_verts: List[_Point2D] = []
    _append_rect_outline(line_verts, (layout["x"], layout["y"], layout["w"], layout["h"]))
    _append_rect_outline(line_verts, status["badge_rect"])

    gpu.state.blend_set("ALPHA")
    try:
        shader.bind()
        for rgba, verts in color_tris.items():
            if not verts:
                continue
            shader.uniform_float("color", rgba)
            batch = batch_for_shader(shader, "TRIS", {"pos": verts})
            batch.draw(shader)

        if line_verts:
            shader.uniform_float("color", constants.HUD_BORDER)
            border_batch = batch_for_shader(shader, "LINES", {"pos": line_verts})
            border_batch.draw(shader)

        font_id = 0
        hx, hy, _, _ = layout["header_rect"]
        _draw_text(font_id, hx, hy + 6.0 * scale, 11.5 * scale, constants.HUD_TEXT_PRIMARY, layout["title"])

        bx, by, bw, _ = status["badge_rect"]
        badge_font_size = max(int(round(10.5 * scale)), 8)
        blf.size(font_id, badge_font_size)
        text_w, _ = blf.dimensions(font_id, status["label"])
        badge_tx = bx + max((bw - text_w) * 0.5, 4.0 * scale)
        _draw_text(font_id, badge_tx, by + 5.0 * scale, 10.5 * scale, status["color"], status["label"])

        mx, my, _, _ = layout["meta_rect"]
        meta_line = f"{status['counts_text']}  |  {layout['meta_text']}"
        _draw_text(font_id, mx, my + 2.5 * scale, 8.8 * scale, constants.HUD_TEXT_MUTED, meta_line)

        for gauge in layout["gauges"]:
            cx, cy, cw, ch = gauge["cell_rect"]
            _draw_text(font_id, cx + 4.0 * scale, cy + ch - 13.0 * scale, 9.5 * scale, constants.HUD_TEXT_PRIMARY, gauge["key"])
            _draw_text(font_id, cx + cw - 28.0 * scale, cy + ch - 13.0 * scale, 9.0 * scale, gauge["color"], gauge["pct_text"])

        if banner is not None:
            wx, wy, _, _ = banner["rect"]
            _draw_text(font_id, wx + 6.0 * scale, wy + 6.0 * scale, 9.5 * scale, constants.HUD_TEXT_PRIMARY, banner["text"])

        if legend is not None:
            tx, ty = legend["title_pos"]
            _draw_text(font_id, tx, ty, 9.0 * scale, constants.HUD_TEXT_MUTED, legend["title"])
            for row in legend["rows"]:
                lx, ly = row["text_pos"]
                _draw_text(font_id, lx, ly, 8.8 * scale, constants.HUD_TEXT_PRIMARY, row["label"])
    finally:
        gpu.state.blend_set("NONE")
        CACHE.set_draw_time_ms((time.perf_counter() - t_start) * 1000.0)


def is_handler_registered() -> bool:
    """Return True if the 2D HUD POST_PIXEL draw handler is currently active."""
    return _DRAW_HANDLE is not None


def register_draw_handler() -> None:
    """Idempotently register the 2D HUD POST_PIXEL draw handler on SpaceView3D."""
    global _DRAW_HANDLE
    if _DRAW_HANDLE is None:
        _DRAW_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
            draw_hud_2d,
            (),
            "WINDOW",
            "POST_PIXEL",
        )


def unregister_draw_handler() -> None:
    """Idempotently remove the 2D HUD POST_PIXEL draw handler."""
    global _DRAW_HANDLE
    if _DRAW_HANDLE is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_DRAW_HANDLE, "WINDOW")
        except (ValueError, RuntimeError):
            pass
        _DRAW_HANDLE = None
