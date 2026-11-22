"""Object transform scale, delta scale, animation/driver guard, and dimension analysis."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

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


def _action_has_scale_fcurves(action: Any, slot_handle: int = 0) -> bool:
    """Return True if ``action`` keyframes ``scale`` or ``delta_scale`` (slotted or legacy)."""
    if action is None:
        return False

    for layer in getattr(action, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            for bag in getattr(strip, "channelbags", ()):
                if slot_handle != 0 and getattr(bag, "slot_handle", 0) != slot_handle:
                    continue
                for fc in getattr(bag, "fcurves", ()):
                    if getattr(fc, "data_path", "") in ("scale", "delta_scale"):
                        return True

    legacy_fcurves = getattr(action, "fcurves", None)
    if legacy_fcurves is not None:
        for fc in legacy_fcurves:
            if getattr(fc, "data_path", "") in ("scale", "delta_scale"):
                return True

    return False


def has_animated_or_driven_scale(obj: Optional[bpy.types.Object]) -> bool:
    """Return True if ``obj`` has drivers, keyframes, or NLA strips animating scale/delta_scale."""
    if obj is None:
        return False
    ad = getattr(obj, "animation_data", None)
    if ad is None:
        return False

    for drv in getattr(ad, "drivers", ()):
        dp = getattr(drv, "data_path", "")
        if dp.startswith("scale") or dp.startswith("delta_scale"):
            return True

    action = getattr(ad, "action", None)
    slot_handle = int(getattr(ad, "action_slot_handle", 0) or 0)
    if _action_has_scale_fcurves(action, slot_handle):
        return True

    for track in getattr(ad, "nla_tracks", ()):
        for strip in getattr(track, "strips", ()):
            strip_action = getattr(strip, "action", None)
            strip_slot = int(getattr(strip, "action_slot_handle", 0) or 0)
            if _action_has_scale_fcurves(strip_action, strip_slot):
                return True

    return False


def _non_fixable_reasons(
    editable: bool,
    animated: bool,
    parent_inverted: bool = False,
) -> List[str]:
    reasons: List[str] = []
    if parent_inverted:
        reasons.append("parent-inherited negative scale")
    if animated:
        reasons.append("animated/driven scale")
    if not editable:
        reasons.append("linked/non-editable datablock")
    return reasons


def analyze_object_scale(obj: bpy.types.Object) -> Tuple[ObjectScaleInfo, List[DiagnosticIssue]]:
    """Analyze scale uniformity, inversion, applied state, and world dimensions for a mesh object."""
    sx, sy, sz = (float(obj.scale.x), float(obj.scale.y), float(obj.scale.z))
    ds = getattr(obj, "delta_scale", (1.0, 1.0, 1.0))
    dsx, dsy, dsz = (float(ds[0]), float(ds[1]), float(ds[2]))
    eff_sx, eff_sy, eff_sz = (sx * dsx, sy * dsy, sz * dsz)
    eff_scale = (eff_sx, eff_sy, eff_sz)

    dx, dy, dz = (float(obj.dimensions.x), float(obj.dimensions.y), float(obj.dimensions.z))
    world_det = float(obj.matrix_world.to_3x3().determinant())
    local_det = eff_sx * eff_sy * eff_sz

    local_inverted = (local_det < 0.0) or any(v < 0.0 for v in eff_scale)
    parent_inverted = (world_det < 0.0) and not local_inverted
    is_inverted = local_inverted or parent_inverted

    tol = constants.SCALE_TOLERANCE
    ax, ay, az = abs(eff_sx), abs(eff_sy), abs(eff_sz)
    max_abs_diff = max(abs(ax - ay), abs(ay - az), abs(ax - az))
    rel_tol = tol * max(1.0, ax, ay, az)
    is_uniform = max_abs_diff <= rel_tol

    max_unapplied_delta = max(abs(eff_sx - 1.0), abs(eff_sy - 1.0), abs(eff_sz - 1.0))
    is_applied = (not is_inverted) and (max_unapplied_delta <= tol)

    mesh = getattr(obj, "data", None)
    has_verts = mesh is not None and len(mesh.vertices) > 0
    max_dim = max(dx, dy, dz)
    too_small = has_verts and (0.0 < max_dim < constants.DIM_MIN_METERS)
    too_large = has_verts and (max_dim > constants.DIM_MAX_METERS)
    has_dimension_outlier = too_small or too_large

    editable = constants.is_editable_id(obj) and constants.is_editable_id(mesh)
    animated = has_animated_or_driven_scale(obj)

    issues: List[DiagnosticIssue] = []

    if is_inverted:
        measured_inv = world_det if world_det < 0.0 else min(eff_scale)
        inv_reasons = _non_fixable_reasons(editable, animated, parent_inverted=parent_inverted)
        fixable_inv = len(inv_reasons) == 0
        suffix = f" [not auto-fixable: {'; '.join(inv_reasons)}]" if inv_reasons else ""
        issues.append(
            DiagnosticIssue(
                code="SCALE_INVERTED",
                category=Category.SCALE,
                severity=Severity.CRITICAL,
                message=(
                    f"Object '{obj.name}' has inverted scale "
                    f"({eff_sx:.3f}, {eff_sy:.3f}, {eff_sz:.3f}, det={world_det:.3f}){suffix}"
                ),
                object_name=obj.name,
                measured_value=measured_inv,
                expected_range="det > 0, scale > 0",
                fixable=fixable_inv,
            )
        )

    local_reasons = _non_fixable_reasons(editable, animated, parent_inverted=False)
    fixable_local = len(local_reasons) == 0
    local_suffix = f" [not auto-fixable: {'; '.join(local_reasons)}]" if local_reasons else ""

    if not is_uniform:
        issues.append(
            DiagnosticIssue(
                code="SCALE_NON_UNIFORM",
                category=Category.SCALE,
                severity=Severity.CRITICAL,
                message=(
                    f"Object '{obj.name}' has non-uniform scale "
                    f"({eff_sx:.3f}, {eff_sy:.3f}, {eff_sz:.3f}){local_suffix}"
                ),
                object_name=obj.name,
                measured_value=max_abs_diff,
                expected_range=f"max |s_i - s_j| <= {rel_tol:.3f}",
                fixable=fixable_local,
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
                    f"({eff_sx:.3f}, {eff_sy:.3f}, {eff_sz:.3f}){local_suffix}"
                ),
                object_name=obj.name,
                measured_value=max_unapplied_delta,
                expected_range="(1.000, 1.000, 1.000)",
                fixable=fixable_local,
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
        scale=eff_scale,
        dimensions=(dx, dy, dz),
        determinant=world_det,
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
