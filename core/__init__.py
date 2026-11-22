"""Core analysis and shader diagnostic package for FlightCheck."""

from __future__ import annotations

import importlib
import time
from typing import Any, Optional

import bpy

from .. import constants
from ..state import CACHE, DiagnosticReport
from . import graph_crawler, scale_analyzer, texel_analyzer

try:
    from . import pbr_shader_nodes
except ImportError:
    pbr_shader_nodes = None  # type: ignore[assignment]

try:
    from . import shader_splicer
except ImportError:
    shader_splicer = None  # type: ignore[assignment]


def __getattr__(name: str) -> Any:
    if name in ("pbr_shader_nodes", "shader_splicer"):
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _count_scanned_images(
    scene: Optional[bpy.types.Scene],
    selected_only: bool = False,
) -> int:
    if scene is None:
        return 0
    mat_entries = graph_crawler._iter_scene_materials(scene, selected_only=selected_only)
    if not mat_entries:
        return 0
    materials = [mat for mat, _obj_name in mat_entries]
    return len(graph_crawler.collect_colorspace_map(materials))


def run_full_scan(
    scene: Optional[bpy.types.Scene],
    write_texel_attr: bool = True,
) -> DiagnosticReport:
    """Run shader/colorspace, scale/dimension, and texel density scans and cache the report."""
    t_start = time.perf_counter()

    fc = getattr(scene, "flightcheck", None) if scene is not None else None
    selected_only = bool(getattr(fc, "selected_only", False))
    target_td = float(getattr(fc, "target_texel_density", constants.TEXEL_DEFAULT_TARGET))
    default_tex = int(getattr(fc, "default_texture_size", constants.TEXEL_DEFAULT_RES))

    mat_issues, scanned_materials = graph_crawler.scan_scene_materials(
        scene,
        selected_only=selected_only,
    )
    scale_info, scale_issues = scale_analyzer.scan_scene_scales(
        scene,
        selected_only=selected_only,
    )
    texel_stats, texel_issues = texel_analyzer.scan_scene_texels(
        scene,
        target_px_per_m=target_td,
        default_tex_size=default_tex,
        selected_only=selected_only,
        write_attribute=write_texel_attr,
    )

    all_issues = mat_issues + scale_issues + texel_issues
    scanned_objects = max(len(scale_info), len(texel_stats))
    scanned_images = _count_scanned_images(scene, selected_only=selected_only)
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    report = DiagnosticReport(
        issues=all_issues,
        scale_info=scale_info,
        texel_stats=texel_stats,
        scanned_objects=scanned_objects,
        scanned_materials=scanned_materials,
        scanned_images=scanned_images,
        scan_duration_ms=elapsed_ms,
    )
    CACHE.set_report(report)
    return report


__all__ = [
    "graph_crawler",
    "scale_analyzer",
    "texel_analyzer",
    "pbr_shader_nodes",
    "shader_splicer",
    "run_full_scan",
]
