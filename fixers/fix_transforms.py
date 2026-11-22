"""Headless-safe, multi-user-aware object transform scale baker."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import bmesh
import bpy
from mathutils import Matrix

from .. import constants
from ..state import internal_mutation
from . import iter_scene_mesh_objects, sync_wm_issues, update_view_layer

_MATCH_EPS = 1e-4


def _effective_scale(obj: bpy.types.Object) -> Tuple[float, float, float]:
    """Return combined ``obj.scale * obj.delta_scale`` components."""
    sx, sy, sz = (float(obj.scale.x), float(obj.scale.y), float(obj.scale.z))
    ds = getattr(obj, "delta_scale", (1.0, 1.0, 1.0))
    dsx, dsy, dsz = (float(ds[0]), float(ds[1]), float(ds[2]))
    return (sx * dsx, sy * dsy, sz * dsz)


def _needs_scale_fix(obj: bpy.types.Object) -> bool:
    """Return True if ``obj`` has unapplied, delta, or inverted scale exceeding tolerance."""
    sx, sy, sz = (float(obj.scale.x), float(obj.scale.y), float(obj.scale.z))
    ds = getattr(obj, "delta_scale", (1.0, 1.0, 1.0))
    dsx, dsy, dsz = (float(ds[0]), float(ds[1]), float(ds[2]))
    ex, ey, ez = (sx * dsx, sy * dsy, sz * dsz)
    tol = constants.SCALE_TOLERANCE
    return any(abs(v - 1.0) > tol for v in (sx, sy, sz, dsx, dsy, dsz, ex, ey, ez))


def _is_animated_or_driven_scale(obj: bpy.types.Object) -> bool:
    """Delegate to ``scale_analyzer.has_animated_or_driven_scale`` with local driver fallback."""
    from ..core import scale_analyzer

    checker = getattr(scale_analyzer, "has_animated_or_driven_scale", None)
    if callable(checker):
        return bool(checker(obj))

    ad = getattr(obj, "animation_data", None)
    if ad is None:
        return False
    for drv in getattr(ad, "drivers", ()):
        dp = getattr(drv, "data_path", "")
        if dp == "scale" or dp.startswith("scale") or dp == "delta_scale" or dp.startswith("delta_scale"):
            return True
    return False


def _apply_scale_to_mesh(
    mesh: bpy.types.Mesh,
    sx: float,
    sy: float,
    sz: float,
) -> None:
    """Bake diagonal scale ``(sx, sy, sz)`` into ``mesh`` and reverse faces if negative det."""
    if abs(sx - 1.0) <= 1e-6 and abs(sy - 1.0) <= 1e-6 and abs(sz - 1.0) <= 1e-6:
        return

    scale_mat = Matrix.Diagonal((sx, sy, sz, 1.0))
    mesh.transform(scale_mat, shape_keys=True)

    if (sx * sy * sz) < 0.0:
        bm = bmesh.new()
        try:
            bm.from_mesh(mesh)
            if len(bm.faces) > 0:
                bmesh.ops.reverse_faces(bm, faces=list(bm.faces))
            bm.normal_update()
            bm.to_mesh(mesh)
        finally:
            bm.free()

    mesh.update()


def _reset_object_scale_and_preserve_children(
    obj: bpy.types.Object,
    scene: Optional[bpy.types.Scene],
) -> None:
    """Reset ``obj.scale`` and ``delta_scale`` to unity while preserving child world transforms."""
    old_pw = obj.matrix_world.copy()
    child_snapshots = [
        (child, child.matrix_parent_inverse.copy())
        for child in obj.children
        if child.parent_type == "OBJECT" and constants.is_editable_id(child)
    ]

    obj.scale = (1.0, 1.0, 1.0)
    if hasattr(obj, "delta_scale"):
        obj.delta_scale = (1.0, 1.0, 1.0)

    if child_snapshots:
        update_view_layer(scene)
        inv_new_pw = obj.matrix_world.inverted_safe()
        for child, old_mpi in child_snapshots:
            child.matrix_parent_inverse = inv_new_pw @ old_pw @ old_mpi
        update_view_layer(scene)


def _scales_match(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
    tol: float = _MATCH_EPS,
) -> bool:
    return (
        abs(a[0] - b[0]) <= tol
        and abs(a[1] - b[1]) <= tol
        and abs(a[2] - b[2]) <= tol
    )


def fix_object_transform(obj: Optional[bpy.types.Object]) -> bool:
    """Bake effective scale into ``obj.data`` safely, splitting if shared with other users."""
    if obj is None or obj.type != "MESH" or obj.data is None:
        return False
    if getattr(obj, "mode", "OBJECT") == "EDIT":
        return False
    if not constants.is_editable_id(obj) or not constants.is_editable_id(obj.data):
        return False
    if _is_animated_or_driven_scale(obj):
        return False
    if not _needs_scale_fix(obj):
        return False

    ex, ey, ez = _effective_scale(obj)
    users = [u for u in bpy.data.objects if u.data == obj.data]
    if len(users) > 1:
        obj.data = obj.data.copy()

    _apply_scale_to_mesh(obj.data, ex, ey, ez)

    user_scenes = list(getattr(obj, "users_scene", ()))
    primary_scene = user_scenes[0] if user_scenes else None
    _reset_object_scale_and_preserve_children(obj, primary_scene)
    for user_scene in user_scenes:
        update_view_layer(user_scene)
    if not user_scenes:
        update_view_layer(None)
    return True


def fix_scene_transforms(
    scene: Optional[bpy.types.Scene],
    selected_only: bool = False,
) -> Dict[str, int]:
    """Bake object scales into meshes across ``scene`` without fragile bpy.ops calls."""
    if scene is None:
        return {
            "objects_fixed": 0,
            "meshes_split": 0,
            "skipped_animated": 0,
            "skipped_edit": 0,
            "skipped_linked": 0,
        }

    update_view_layer(scene)

    targets: List[bpy.types.Object] = []
    skipped_animated = 0
    skipped_edit = 0
    skipped_linked = 0

    for obj in iter_scene_mesh_objects(scene, selected_only=selected_only):
        if obj.data is None:
            continue
        if not _needs_scale_fix(obj):
            continue
        if getattr(obj, "mode", "OBJECT") == "EDIT":
            skipped_edit += 1
            continue
        if not constants.is_editable_id(obj) or not constants.is_editable_id(obj.data):
            skipped_linked += 1
            continue
        if _is_animated_or_driven_scale(obj):
            skipped_animated += 1
            continue
        targets.append(obj)

    if not targets:
        return {
            "objects_fixed": 0,
            "meshes_split": 0,
            "skipped_animated": skipped_animated,
            "skipped_edit": skipped_edit,
            "skipped_linked": skipped_linked,
        }

    grouped: Dict[int, List[bpy.types.Object]] = defaultdict(list)
    mesh_by_ptr: Dict[int, bpy.types.Mesh] = {}
    for obj in targets:
        ptr = obj.data.as_pointer()
        grouped[ptr].append(obj)
        mesh_by_ptr[ptr] = obj.data

    all_users_by_ptr: Dict[int, List[bpy.types.Object]] = defaultdict(list)
    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.data is not None:
            ptr = obj.data.as_pointer()
            if ptr in mesh_by_ptr:
                all_users_by_ptr[ptr].append(obj)

    objects_fixed = 0
    meshes_split = 0

    for ptr, group_objs in grouped.items():
        mesh = mesh_by_ptr[ptr]
        all_users = all_users_by_ptr[ptr]
        fix_ptrs = {o.as_pointer() for o in group_objs}
        all_in_fix_set = len(all_users) == len(group_objs) and all(
            u.as_pointer() in fix_ptrs for u in all_users
        )

        ref_scale = _effective_scale(group_objs[0])
        uniform_group_scale = all(
            _scales_match(_effective_scale(o), ref_scale)
            for o in group_objs
        )

        if all_in_fix_set and uniform_group_scale:
            _apply_scale_to_mesh(mesh, ref_scale[0], ref_scale[1], ref_scale[2])
            for user in group_objs:
                _reset_object_scale_and_preserve_children(user, scene)
                objects_fixed += 1
        else:
            for user in group_objs:
                ex, ey, ez = _effective_scale(user)
                user.data = user.data.copy()
                meshes_split += 1
                _apply_scale_to_mesh(user.data, ex, ey, ez)
                _reset_object_scale_and_preserve_children(user, scene)
                objects_fixed += 1

    update_view_layer(scene)
    return {
        "objects_fixed": objects_fixed,
        "meshes_split": meshes_split,
        "skipped_animated": skipped_animated,
        "skipped_edit": skipped_edit,
        "skipped_linked": skipped_linked,
    }


class FLIGHTCHECK_OT_fix_transforms(bpy.types.Operator):
    """Bake unapplied and inverted object scales safely into mesh geometry."""

    bl_idname = "flightcheck.fix_transforms"
    bl_label = "Apply & Fix Transforms"
    bl_description = (
        "Bake object scale into mesh geometry and reverse inverted face normals cleanly"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return context.scene is not None and context.mode == "OBJECT"

    def execute(self, context: bpy.types.Context) -> set[str]:
        from ..core import run_full_scan
        from ..hud import tag_view3d_redraw

        scene = context.scene
        fc = getattr(scene, "flightcheck", None)
        selected_only = bool(fc.selected_only) if fc is not None else False

        with internal_mutation():
            result = fix_scene_transforms(scene, selected_only=selected_only)
            report = run_full_scan(scene, write_texel_attr=True)
            sync_wm_issues(context, report)
            tag_view3d_redraw()

        skipped_total = (
            result["skipped_animated"]
            + result["skipped_edit"]
            + result["skipped_linked"]
        )
        msg = (
            f"FlightCheck: Fixed {result['objects_fixed']} object transform(s); "
            f"split {result['meshes_split']} multi-user mesh(es)."
        )
        if skipped_total > 0:
            msg += (
                f" Skipped: {result['skipped_animated']} animated, "
                f"{result['skipped_edit']} edit-mode, "
                f"{result['skipped_linked']} linked."
            )
            self.report({"WARNING"}, msg)
        else:
            self.report({"INFO"}, msg)
        return {"FINISHED"}
