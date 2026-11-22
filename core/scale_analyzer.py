"""Object transform scale and world-space dimension analysis."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import bpy

from .. import constants
from ..state import Category, DiagnosticIssue, ObjectScaleInfo, Severity


def _iter_target_mesh_objects(
    scene: Optional[bpy.types.Scene],
    selected_only: bool = False,
) -> List[bpy.types.Object]:
    """Return mesh objects in ``scene``, optionally filtered to selected objects."""
    if scene is None:
        return []

    view_layer = None
    if selected_only:
        ctx = getattr(bpy, "context", None)
        if ctx is not None and getattr(ctx, "scene", None) == scene and getattr(ctx, "view_layer", None) is not None:
            view_layer = ctx.view_layer
        elif len(scene.view_layers) > 0:
            view_layer = scene.view_layers[0]

    objects: List[bpy.types.Object] = []
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        if selected_only:
            try:
                is_selected = obj.select_get(view_layer=view_layer) if view_layer is not None else obj.select_get()
            except RuntimeError:
                is_selected = False
            if not is_selected:
                continue
        objects.append(obj)
    return objects


def analyze_object_scale(obj: bpy.types.Object) -> Tuple[ObjectScaleInfo, List[DiagnosticIssue]]:
    """Analyze scale uniformity, inversion, applied state, and world dimensions for a mesh object."""
    sx, sy, sz = (float(obj.scale.x), float(obj.scale.y), float(obj.scale.z))
    dx, dy, dz = (float(obj.dimensions.x), float(obj.dimensions.y), float(obj.dimensions.z))
    det = float(obj.matrix_world.to_3x3().determinant())

    tol = constants.SCALE_TOLERANCE
    is_inverted = (det < 0.0) or (sx < 0.0) or (sy < 0.0) or (sz < 0.0)

    ax, ay, az = abs(sx), abs(sy), abs(sz)
    max_abs_diff = max(abs(ax - ay), abs(ay - az), abs(ax - az))
    is_uniform = max_abs_diff <= tol

    max_unapplied_delta = max(abs(sx - 1.0), abs(sy - 1.0), abs(sz - 1.0))
    is_applied = (not is_inverted) and (max_unapplied_delta <= tol)

    mesh = getattr(obj, "data", None)
    has_verts = mesh is not None and len(mesh.vertices) > 0
    max_dim = max(dx, dy, dz)
    too_small = has_verts and (0.0 < max_dim < constants.DIM_MIN_METERS)
    too_large = has_verts and (max_dim > constants.DIM_MAX_METERS)
    has_dimension_outlier = too_small or too_large

    issues: List[DiagnosticIssue] = []

    if is_inverted:
        measured_inv = det if det < 0.0 else min(sx, sy, sz)
        issues.append(
            DiagnosticIssue(
                code="SCALE_INVERTED",
                category=Category.SCALE,
                severity=Severity.CRITICAL,
                message=(
                    f"Object '{obj.name}' has inverted scale "
                    f"({sx:.3f}, {sy:.3f}, {sz:.3f}, det={det:.3f})"
                ),
                object_name=obj.name,
                measured_value=measured_inv,
                expected_range="det > 0, scale > 0",
                fixable=True,
            )
        )

    if not is_uniform:
        issues.append(
            DiagnosticIssue(
                code="SCALE_NON_UNIFORM",
                category=Category.SCALE,
                severity=Severity.CRITICAL,
                message=(
                    f"Object '{obj.name}' has non-uniform scale "
                    f"({sx:.3f}, {sy:.3f}, {sz:.3f})"
                ),
                object_name=obj.name,
                measured_value=max_abs_diff,
                expected_range=f"max |s_i - s_j| <= {tol:.3f}",
                fixable=True,
            )
        )
    elif not is_inverted and not is_applied:
        issues.append(
            DiagnosticIssue(
                code="SCALE_UNAPPLIED",
                category=Category.SCALE,
                severity=Severity.WARNING,
                message=(
                    f"Object '{obj.name}' has unapplied uniform scale "
                    f"({sx:.3f}, {sy:.3f}, {sz:.3f})"
                ),
                object_name=obj.name,
                measured_value=max_unapplied_delta,
                expected_range="(1.000, 1.000, 1.000)",
                fixable=True,
            )
        )

    if too_small:
        issues.append(
            DiagnosticIssue(
                code="DIMENSION_TOO_SMALL",
                category=Category.SCALE,
                severity=Severity.WARNING,
                message=(
                    f"Object '{obj.name}' max world dimension is {max_dim:.6f} m "
                    f"(< {constants.DIM_MIN_METERS} m)"
                ),
                object_name=obj.name,
                measured_value=max_dim,
                expected_range=f"[{constants.DIM_MIN_METERS}, {constants.DIM_MAX_METERS}] m",
                fixable=False,
            )
        )

    if too_large:
        issues.append(
            DiagnosticIssue(
                code="DIMENSION_TOO_LARGE",
                category=Category.SCALE,
                severity=Severity.WARNING,
                message=(
                    f"Object '{obj.name}' max world dimension is {max_dim:.2f} m "
                    f"(> {constants.DIM_MAX_METERS} m)"
                ),
                object_name=obj.name,
                measured_value=max_dim,
                expected_range=f"[{constants.DIM_MIN_METERS}, {constants.DIM_MAX_METERS}] m",
                fixable=False,
            )
        )

    if any(issue.severity == Severity.CRITICAL for issue in issues):
        severity = Severity.CRITICAL
    elif any(issue.severity == Severity.WARNING for issue in issues):
        severity = Severity.WARNING
    else:
        severity = Severity.OK

    info = ObjectScaleInfo(
        object_name=obj.name,
        scale=(sx, sy, sz),
        dimensions=(dx, dy, dz),
        determinant=det,
        is_uniform=is_uniform,
        is_applied=is_applied,
        is_inverted=is_inverted,
        has_dimension_outlier=has_dimension_outlier,
        severity=severity,
    )
    return info, issues


def scan_scene_scales(
    scene: Optional[bpy.types.Scene],
    selected_only: bool = False,
) -> Tuple[Dict[str, ObjectScaleInfo], List[DiagnosticIssue]]:
    """Scan mesh objects in ``scene`` for transform scale and dimension anomalies."""
    scale_info: Dict[str, ObjectScaleInfo] = {}
    issues: List[DiagnosticIssue] = []

    for obj in _iter_target_mesh_objects(scene, selected_only=selected_only):
        info, obj_issues = analyze_object_scale(obj)
        scale_info[obj.name] = info
        issues.extend(obj_issues)

    return scale_info, issues
