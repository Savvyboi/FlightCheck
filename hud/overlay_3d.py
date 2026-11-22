"""3D Viewport bounding cage overlay and pure geometry builder."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from .. import constants
from ..state import CACHE, Category, DiagnosticReport

_DRAW_HANDLE: Any = None
_SHADER_3D: Any = None
_IS_POLYLINE: bool = False

_Vec3 = Tuple[float, float, float]
_ColorRGBA = Tuple[float, float, float, float]

# Blender obj.bound_box corner ordering:
# 0:(-,-,-) 1:(-,-,+) 2:(-,+,+) 3:(-,+,-) 4:(+,-,-) 5:(+,-,+) 6:(+,+,+) 7:(+,+,-)
_BOX_EDGE_INDICES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

_FACE_DIAGONAL_INDICES: Tuple[Tuple[int, int], ...] = (
    # -X face / +X face
    (0, 2), (1, 3),
    (4, 6), (5, 7),
    # -Y face / +Y face
    (0, 5), (1, 4),
    (2, 7), (3, 6),
    # -Z face / +Z face
    (0, 7), (3, 4),
    (1, 6), (2, 5),
)


def _get_shader_3d() -> Tuple[Any, bool]:
    """Lazily compile and cache the built-in 3D line shader."""
    global _SHADER_3D, _IS_POLYLINE
    if _SHADER_3D is not None:
        return _SHADER_3D, _IS_POLYLINE

    try:
        _SHADER_3D = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
        _IS_POLYLINE = True
    except Exception:
        _SHADER_3D = gpu.shader.from_builtin("UNIFORM_COLOR")
        _IS_POLYLINE = False
    return _SHADER_3D, _IS_POLYLINE


def _append_box_edges(target: List[_Vec3], corners: List[_Vec3]) -> None:
    """Append the 12 world-space bounding box edges (24 endpoints) to target."""
    for i0, i1 in _BOX_EDGE_INDICES:
        target.append(corners[i0])
        target.append(corners[i1])


def _append_face_diagonals(target: List[_Vec3], corners: List[_Vec3]) -> None:
    """Append diagonal X-crosses across all 6 faces for inverted-scale cages."""
    for i0, i1 in _FACE_DIAGONAL_INDICES:
        target.append(corners[i0])
        target.append(corners[i1])


def build_cage_segments(
    scene: Any,
    report: Optional[DiagnosticReport],
) -> Dict[str, List[_Vec3]]:
    """Build world-space LINES vertex lists for scale and dimension warning cages."""
    segments: Dict[str, List[_Vec3]] = {
        "crit": [],
        "warn": [],
        "dim": [],
    }
    if scene is None or report is None:
        return segments

    scale_issue_codes: Dict[str, Set[str]] = {}
    for issue in report.issues:
        if issue.category == Category.SCALE and issue.object_name:
            scale_issue_codes.setdefault(issue.object_name, set()).add(issue.code)

    candidate_names: List[str] = list(report.scale_info.keys())
    seen_names: Set[str] = set(candidate_names)
    for obj_name in scale_issue_codes:
        if obj_name not in seen_names:
            candidate_names.append(obj_name)
            seen_names.add(obj_name)

    for obj_name in candidate_names:
        obj = scene.objects.get(obj_name)
        if obj is None or obj.type != "MESH":
            continue

        info = report.scale_info.get(obj_name)
        codes = scale_issue_codes.get(obj_name, set())

        is_inverted = (info.is_inverted if info is not None else False) or ("SCALE_INVERTED" in codes)
        is_non_uniform = (not info.is_uniform if info is not None else False) or ("SCALE_NON_UNIFORM" in codes)
        is_unapplied = (
            info is not None and not info.is_applied and info.is_uniform and not info.is_inverted
        ) or ("SCALE_UNAPPLIED" in codes)
        is_dim = (
            (info.has_dimension_outlier if info is not None else False)
            or ("DIMENSION_TOO_SMALL" in codes)
            or ("DIMENSION_TOO_LARGE" in codes)
        )

        if not (is_inverted or is_non_uniform or is_unapplied or is_dim):
            continue

        mat = obj.matrix_world
        corners: List[_Vec3] = []
        for corner in obj.bound_box:
            world_pt = mat @ Vector(corner)
            corners.append((float(world_pt.x), float(world_pt.y), float(world_pt.z)))

        if len(corners) < 8:
            continue

        if is_inverted or is_non_uniform:
            _append_box_edges(segments["crit"], corners)
            if is_inverted:
                _append_face_diagonals(segments["crit"], corners)
        elif is_unapplied:
            _append_box_edges(segments["warn"], corners)

        if is_dim:
            _append_box_edges(segments["dim"], corners)

    return segments


def draw_overlay_3d() -> None:
    """POST_VIEW draw callback rendering 3D bounding cages for scale/dimension issues."""
    if bpy.app.background:
        return

    context = bpy.context
    if context is None or context.region is None or context.scene is None:
        return

    settings = getattr(context.scene, "flightcheck", None)
    if settings is None or not settings.enabled or not settings.show_overlay_3d:
        return

    report = CACHE.get_report()
    if report is None:
        return

    segments = build_cage_segments(context.scene, report)
    if not (segments["crit"] or segments["warn"] or segments["dim"]):
        return

    try:
        shader, is_polyline = _get_shader_3d()
    except Exception:
        return

    layers: Tuple[Tuple[str, _ColorRGBA, float], ...] = (
        ("dim", constants.CAGE_COLOR_DIM, 2.0),
        ("warn", constants.CAGE_COLOR_WARN, 2.2),
        ("crit", constants.CAGE_COLOR_CRIT, 2.8),
    )

    viewport_size = (float(context.region.width), float(context.region.height))

    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("LESS_EQUAL")
    try:
        shader.bind()
        if is_polyline:
            shader.uniform_float("viewportSize", viewport_size)

        for key, color, line_width in layers:
            coords = segments[key]
            if not coords:
                continue
            if is_polyline:
                shader.uniform_float("lineWidth", line_width)
            else:
                gpu.state.line_width_set(line_width)
            shader.uniform_float("color", color)
            batch = batch_for_shader(shader, "LINES", {"pos": coords})
            batch.draw(shader)
    finally:
        gpu.state.depth_test_set("NONE")
        gpu.state.blend_set("NONE")


def is_handler_registered() -> bool:
    """Return True if the 3D cage POST_VIEW draw handler is currently active."""
    return _DRAW_HANDLE is not None


def register_draw_handler() -> None:
    """Idempotently register the 3D cage POST_VIEW draw handler on SpaceView3D."""
    global _DRAW_HANDLE
    if _DRAW_HANDLE is None:
        _DRAW_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
            draw_overlay_3d,
            (),
            "WINDOW",
            "POST_VIEW",
        )


def unregister_draw_handler() -> None:
    """Idempotently remove the 3D cage POST_VIEW draw handler."""
    global _DRAW_HANDLE
    if _DRAW_HANDLE is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_DRAW_HANDLE, "WINDOW")
        except (ValueError, RuntimeError):
            pass
        _DRAW_HANDLE = None
