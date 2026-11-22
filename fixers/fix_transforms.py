"""Headless-safe, multi-user-aware object transform scale baker."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import bmesh
import bpy
from mathutils import Matrix

from .. import constants
from . import iter_scene_mesh_objects, update_view_layer

_MATCH_EPS = 1e-4


def _needs_scale_fix(obj: bpy.types.Object) -> bool:
    """Return True if ``obj`` has unapplied or negative scale exceeding tolerance."""
    sx, sy, sz = (float(obj.scale.x), float(obj.scale.y), float(obj.scale.z))
    tol = constants.SCALE_TOLERANCE
    return (
        abs(sx - 1.0) > tol
        or abs(sy - 1.0) > tol
        or abs(sz - 1.0) > tol
    )


def _apply_scale_to_mesh(
    mesh: bpy.types.Mesh,
    sx: float,
    sy: float,
    sz: float,
) -> None:
    """Bake diagonal scale ``(sx, sy, sz)`` into ``mesh`` and reverse faces if negative det."""
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


def _reset_object_scale_and_children(
    obj: bpy.types.Object,
    sx: float,
    sy: float,
    sz: float,
) -> None:
    """Reset ``obj.scale`` to unity and compensate direct OBJECT-parented child locations."""
    obj.scale = (1.0, 1.0, 1.0)
    for child in obj.children:
        if child.parent_type == "OBJECT":
            child.location.x *= sx
            child.location.y *= sy
            child.location.z *= sz


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
    """Bake local scale into ``obj.data`` safely, splitting if shared with other users."""
    if obj is None or obj.type != "MESH" or obj.data is None:
        return False
    if not _needs_scale_fix(obj):
        return False

    sx, sy, sz = (float(obj.scale.x), float(obj.scale.y), float(obj.scale.z))
    users = [u for u in bpy.data.objects if u.data == obj.data]
    if len(users) > 1:
        obj.data = obj.data.copy()

    _apply_scale_to_mesh(obj.data, sx, sy, sz)
    _reset_object_scale_and_children(obj, sx, sy, sz)
    for user_scene in getattr(obj, "users_scene", ()):
        update_view_layer(user_scene)
    if not getattr(obj, "users_scene", ()):
        update_view_layer(None)
    return True


def fix_scene_transforms(
    scene: Optional[bpy.types.Scene],
    selected_only: bool = False,
) -> Dict[str, int]:
    """Bake object scales into meshes across ``scene`` without fragile bpy.ops calls."""
    if scene is None:
        return {"objects_fixed": 0, "meshes_split": 0}

    update_view_layer(scene)

    targets: List[bpy.types.Object] = []
    for obj in iter_scene_mesh_objects(scene, selected_only=selected_only):
        if obj.data is None:
            continue
        det = float(obj.matrix_world.to_3x3().determinant())
        if _needs_scale_fix(obj) or (det < 0.0 and any(s < 0.0 for s in obj.scale)):
            targets.append(obj)

    if not targets:
        return {"objects_fixed": 0, "meshes_split": 0}

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

        ref_scale = (
            float(group_objs[0].scale.x),
            float(group_objs[0].scale.y),
            float(group_objs[0].scale.z),
        )
        uniform_group_scale = all(
            _scales_match(
                (float(o.scale.x), float(o.scale.y), float(o.scale.z)),
                ref_scale,
            )
            for o in group_objs
        )

        if all_in_fix_set and uniform_group_scale:
            _apply_scale_to_mesh(mesh, ref_scale[0], ref_scale[1], ref_scale[2])
            for user in group_objs:
                _reset_object_scale_and_children(
                    user, ref_scale[0], ref_scale[1], ref_scale[2]
                )
                objects_fixed += 1
        else:
            for user in group_objs:
                sx, sy, sz = (
                    float(user.scale.x),
                    float(user.scale.y),
                    float(user.scale.z),
                )
                user.data = user.data.copy()
                meshes_split += 1
                _apply_scale_to_mesh(user.data, sx, sy, sz)
                _reset_object_scale_and_children(user, sx, sy, sz)
                objects_fixed += 1

    update_view_layer(scene)
    return {"objects_fixed": objects_fixed, "meshes_split": meshes_split}


class FLIGHTCHECK_OT_fix_transforms(bpy.types.Operator):
    """Bake unapplied and inverted object scales safely into mesh geometry."""

    bl_idname = "flightcheck.fix_transforms"
    bl_label = "Apply & Fix Transforms"
    bl_description = (
        "Bake object scale into mesh geometry and reverse inverted face normals cleanly"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        from ..core import run_full_scan
        from ..hud import tag_view3d_redraw

        scene = context.scene
        fc = getattr(scene, "flightcheck", None)
        selected_only = bool(fc.selected_only) if fc is not None else False

        result = fix_scene_transforms(scene, selected_only=selected_only)
        run_full_scan(scene, write_texel_attr=True)
        tag_view3d_redraw()

        self.report(
            {"INFO"},
            (
                f"FlightCheck: Fixed {result['objects_fixed']} object transform(s); "
                f"split {result['meshes_split']} multi-user mesh(es)."
            ),
        )
        return {"FINISHED"}
