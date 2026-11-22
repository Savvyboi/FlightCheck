"""Per-UV-island texel density equalizer using bmesh and world-space triangle areas."""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import bmesh
import bpy

from .. import constants
from . import iter_scene_mesh_objects, update_view_layer

_UV_SEAM_TOL_SQ = 1e-10


def _partition_uv_islands(bm: bmesh.types.BMesh, uv_layer) -> List[List[int]]:
    """Partition faces of ``bm`` into UV islands via BFS over UV-continuous edges."""
    bm.faces.ensure_lookup_table()
    face_count = len(bm.faces)
    if face_count == 0:
        return []

    face_vert_uvs: List[Dict[int, object]] = [
        {loop.vert.index: loop[uv_layer].uv.copy() for loop in face.loops}
        for face in bm.faces
    ]

    visited = [False] * face_count
    islands: List[List[int]] = []

    for start_idx in range(face_count):
        if visited[start_idx]:
            continue
        visited[start_idx] = True
        queue: deque[int] = deque([start_idx])
        island: List[int] = []

        while queue:
            f_idx = queue.popleft()
            island.append(f_idx)
            face = bm.faces[f_idx]
            uvs_a = face_vert_uvs[f_idx]

            for edge in face.edges:
                v0 = edge.verts[0].index
                v1 = edge.verts[1].index
                uv_a0 = uvs_a.get(v0)
                uv_a1 = uvs_a.get(v1)
                if uv_a0 is None or uv_a1 is None:
                    continue

                for other in edge.link_faces:
                    o_idx = other.index
                    if visited[o_idx]:
                        continue
                    uvs_b = face_vert_uvs[o_idx]
                    uv_b0 = uvs_b.get(v0)
                    uv_b1 = uvs_b.get(v1)
                    if uv_b0 is None or uv_b1 is None:
                        continue
                    if (
                        (uv_a0 - uv_b0).length_squared <= _UV_SEAM_TOL_SQ
                        and (uv_a1 - uv_b1).length_squared <= _UV_SEAM_TOL_SQ
                    ):
                        visited[o_idx] = True
                        queue.append(o_idx)

        islands.append(island)

    return islands


def equalize_object_texels(
    obj: Optional[bpy.types.Object],
    target_px_per_m: float = constants.TEXEL_DEFAULT_TARGET,
    default_tex_size: int = constants.TEXEL_DEFAULT_RES,
    *,
    _flush_view_layer: bool = True,
) -> int:
    """Scale each UV island on ``obj`` around its UV centroid to match ``target_px_per_m``."""
    if obj is None or obj.type != "MESH" or obj.data is None or target_px_per_m <= 0.0:
        return 0

    mesh = obj.data
    if len(mesh.polygons) == 0:
        return 0

    if _flush_view_layer:
        scenes = getattr(obj, "users_scene", ())
        if scenes:
            update_view_layer(scenes[0])
        else:
            update_view_layer(None)

    from ..core import texel_analyzer

    slot_res_cache: Dict[int, Tuple[int, int]] = {}
    for idx, slot in enumerate(obj.material_slots):
        slot_res_cache[idx] = texel_analyzer.get_material_texture_resolution(
            slot.material, default_size=default_tex_size
        )
    default_res = (int(default_tex_size), int(default_tex_size))

    world_mat = obj.matrix_world.copy()
    islands_scaled = 0

    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        uv_layer = bm.loops.layers.uv.active
        if uv_layer is None or len(bm.faces) == 0:
            return 0

        bm.faces.ensure_lookup_table()
        islands = _partition_uv_islands(bm, uv_layer)

        face_tris: List[List[Tuple[bmesh.types.BMLoop, bmesh.types.BMLoop, bmesh.types.BMLoop]]] = [
            [] for _ in range(len(bm.faces))
        ]
        for tri in bm.calc_loop_triangles():
            face_tris[tri[0].face.index].append(tri)

        for island in islands:
            island_world_area = 0.0
            island_uv_area = 0.0
            w_weighted = 0.0
            h_weighted = 0.0

            for f_idx in island:
                face = bm.faces[f_idx]
                tex_w, tex_h = slot_res_cache.get(face.material_index, default_res)
                for l0, l1, l2 in face_tris[f_idx]:
                    p0 = world_mat @ l0.vert.co
                    p1 = world_mat @ l1.vert.co
                    p2 = world_mat @ l2.vert.co
                    tri_world = 0.5 * (p1 - p0).cross(p2 - p0).length

                    uv0 = l0[uv_layer].uv
                    uv1 = l1[uv_layer].uv
                    uv2 = l2[uv_layer].uv
                    tri_uv = 0.5 * abs(
                        (uv1.x - uv0.x) * (uv2.y - uv0.y)
                        - (uv2.x - uv0.x) * (uv1.y - uv0.y)
                    )

                    island_world_area += tri_world
                    island_uv_area += tri_uv
                    w_weighted += tex_w * tri_world
                    h_weighted += tex_h * tri_world

            if island_world_area <= 1e-12 or island_uv_area <= 1e-12:
                continue

            avg_w = w_weighted / island_world_area
            avg_h = h_weighted / island_world_area
            density_area = (island_uv_area * avg_w * avg_h) / island_world_area
            if density_area <= 1e-12:
                continue

            density = math.sqrt(density_area)
            if density <= 1e-6:
                continue

            s_uv = float(target_px_per_m) / density
            if abs(s_uv - 1.0) <= 1e-4:
                continue

            sum_u = 0.0
            sum_v = 0.0
            loop_count = 0
            for f_idx in island:
                for loop in bm.faces[f_idx].loops:
                    uv = loop[uv_layer].uv
                    sum_u += uv.x
                    sum_v += uv.y
                    loop_count += 1

            if loop_count == 0:
                continue

            cx = sum_u / loop_count
            cy = sum_v / loop_count
            for f_idx in island:
                for loop in bm.faces[f_idx].loops:
                    uv = loop[uv_layer].uv
                    uv.x = cx + (uv.x - cx) * s_uv
                    uv.y = cy + (uv.y - cy) * s_uv

            islands_scaled += 1

        if islands_scaled > 0:
            bm.to_mesh(mesh)
            mesh.update()
    finally:
        bm.free()

    texel_analyzer.analyze_object_texel_density(
        obj,
        target_px_per_m=target_px_per_m,
        default_tex_size=default_tex_size,
        write_attribute=True,
    )
    return islands_scaled


def _object_texel_signature(obj: bpy.types.Object) -> Tuple[Tuple[float, ...], Tuple[int, ...]]:
    """Build a scale and material resolution signature for shared-mesh deduplication."""
    scale_sig = tuple(round(abs(float(s)), 4) for s in obj.matrix_world.to_scale())
    mat_sig = tuple(
        slot.material.as_pointer() if slot.material is not None else 0
        for slot in obj.material_slots
    )
    return (scale_sig, mat_sig)


def equalize_scene_texels(
    scene: Optional[bpy.types.Scene],
    target_px_per_m: float = constants.TEXEL_DEFAULT_TARGET,
    default_tex_size: int = constants.TEXEL_DEFAULT_RES,
    selected_only: bool = False,
) -> Dict[str, int]:
    """Equalize UV island texel density across mesh objects in ``scene``."""
    if scene is None:
        return {"objects_equalized": 0, "islands_scaled": 0, "meshes_split": 0}

    update_view_layer(scene)
    from ..core import texel_analyzer

    targets = iter_scene_mesh_objects(scene, selected_only=selected_only)
    target_ptrs = {obj.as_pointer() for obj in targets}

    seen_meshes: Dict[int, Tuple[Tuple[float, ...], Tuple[int, ...]]] = {}
    objects_equalized = 0
    islands_scaled = 0
    meshes_split = 0

    for obj in targets:
        if obj.data is None:
            continue

        # Protect unselected co-users when operating in selected-only mode.
        if selected_only and obj.data.users > 1:
            has_unselected_user = any(
                u.data == obj.data and u.as_pointer() not in target_ptrs
                for u in bpy.data.objects
            )
            if has_unselected_user:
                obj.data = obj.data.copy()
                meshes_split += 1

        ptr = obj.data.as_pointer()
        sig = _object_texel_signature(obj)

        if ptr in seen_meshes:
            if seen_meshes[ptr] == sig:
                texel_analyzer.analyze_object_texel_density(
                    obj,
                    target_px_per_m=target_px_per_m,
                    default_tex_size=default_tex_size,
                    write_attribute=True,
                )
                continue
            obj.data = obj.data.copy()
            meshes_split += 1
            ptr = obj.data.as_pointer()

        seen_meshes[ptr] = sig
        scaled = equalize_object_texels(
            obj,
            target_px_per_m=target_px_per_m,
            default_tex_size=default_tex_size,
            _flush_view_layer=False,
        )
        if scaled > 0:
            objects_equalized += 1
            islands_scaled += scaled

    return {
        "objects_equalized": objects_equalized,
        "islands_scaled": islands_scaled,
        "meshes_split": meshes_split,
    }


class FLIGHTCHECK_OT_equalize_texels(bpy.types.Operator):
    """Equalize UV island texel densities to the scene target px/m."""

    bl_idname = "flightcheck.equalize_texels"
    bl_label = "Equalize Texel Density"
    bl_description = (
        "Uniformly scale each UV island around its centroid to match Target Texel Density"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        from ..core import run_full_scan
        from ..hud import tag_view3d_redraw

        scene = context.scene
        fc = getattr(scene, "flightcheck", None)
        selected_only = bool(fc.selected_only) if fc is not None else False
        target_td = (
            float(fc.target_texel_density)
            if fc is not None
            else constants.TEXEL_DEFAULT_TARGET
        )
        default_tex = (
            int(fc.default_texture_size)
            if fc is not None
            else constants.TEXEL_DEFAULT_RES
        )

        result = equalize_scene_texels(
            scene,
            target_px_per_m=target_td,
            default_tex_size=default_tex,
            selected_only=selected_only,
        )
        run_full_scan(scene, write_texel_attr=True)
        tag_view3d_redraw()

        self.report(
            {"INFO"},
            (
                f"FlightCheck: Equalized {result['islands_scaled']} UV island(s) "
                f"across {result['objects_equalized']} object(s)."
            ),
        )
        return {"FINISHED"}
