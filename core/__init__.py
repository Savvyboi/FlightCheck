"""Core analysis and shader diagnostic package for FlightCheck."""

from __future__ import annotations

import time
from typing import Optional

import bpy

from .. import constants
from ..state import CACHE, DiagnosticReport
from . import (
    graph_crawler,
    pbr_shader_nodes,
    scale_analyzer,
    shader_splicer,
    texel_analyzer,
)


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

    mat_issues, scanned_materials, scanned_images = graph_crawler.scan_scene_materials_with_stats(
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
    CACHE.set_report(report, scene=scene)
    return report


__all__ = [
    "graph_crawler",
    "scale_analyzer",
    "texel_analyzer",
    "pbr_shader_nodes",
    "shader_splicer",
    "run_full_scan",
]
