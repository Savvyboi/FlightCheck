"""Per-UV-island texel density equalizer using bmesh and world-space triangle areas."""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import bmesh
import bpy
from bpy.props import BoolProperty, EnumProperty
import numpy as np

from .. import constants
from ..state import internal_mutation
from . import iter_scene_mesh_objects, sync_wm_issues, update_view_layer

_UV_SEAM_TOL_SQ = 1e-10
_PACK_MARGIN = 0.005
_PACK_INNER_SPAN = 1.0 - 2.0 * _PACK_MARGIN  # 0.99 -> [0.005, 0.995]


def check_uvs_exceed_unit_square(mesh: Optional[bpy.types.Mesh]) -> bool:
    """Return True if any active UV coordinate lies outside [-1e-4, 1.0 + 1e-4]."""
    if mesh is None or not hasattr(mesh, "uv_layers"):
        return False

    if getattr(mesh, "is_editmode", False):
        bm = bmesh.from_edit_mesh(mesh)
        uv_layer = bm.loops.layers.uv.active
        if uv_layer is None:
            return False
        for face in bm.faces:
            for loop in face.loops:
                uv = loop[uv_layer].uv
                if uv.x < -1e-4 or uv.x > 1.0 + 1e-4 or uv.y < -1e-4 or uv.y > 1.0 + 1e-4:
                    return True
        return False

    uv_layer = mesh.uv_layers.active
    if uv_layer is None:
        return False
    loop_count = len(mesh.loops)
    if loop_count == 0:
        return False

    uvs = np.empty(loop_count * 2, dtype=np.float32)
    uv_layer.uv.foreach_get("vector", uvs)
    return bool(np.any((uvs < -1e-4) | (uvs > 1.0 + 1e-4)))


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


def _compute_island_aabb(
    bm: bmesh.types.BMesh,
    uv_layer,
    island: List[int],
) -> Tuple[float, float, float, float]:
    min_u = math.inf
    max_u = -math.inf
    min_v = math.inf
    max_v = -math.inf
    has_loops = False
    for f_idx in island:
        for loop in bm.faces[f_idx].loops:
            has_loops = True
            uv = loop[uv_layer].uv
            if uv.x < min_u:
                min_u = uv.x
            if uv.x > max_u:
                max_u = uv.x
            if uv.y < min_v:
                min_v = uv.y
            if uv.y > max_v:
                max_v = uv.y
    if not has_loops:
        return (0.0, 0.0, 0.0, 0.0)
    return (min_u, max_u, min_v, max_v)


def _islands_already_packed(aabbs: List[Tuple[float, float, float, float]]) -> bool:
    """Return True if all islands lie within [0, 1] and have mutually disjoint AABBs."""
    if not aabbs:
        return True
    lo_bound = -1e-5
    hi_bound = 1.0 + 1e-5
    for min_u, max_u, min_v, max_v in aabbs:
        if min_u < lo_bound or max_u > hi_bound or min_v < lo_bound or max_v > hi_bound:
            return False
    n = len(aabbs)
    if n == 1:
        return True
    for i in range(n):
        u0, u1, v0, v1 = aabbs[i]
        for j in range(i + 1, n):
            a0, a1, b0, b1 = aabbs[j]
            if min(u1, a1) > max(u0, a0) + 1e-5 and min(v1, b1) > max(v0, b0) + 1e-5:
                return False
    return True


def _run_shelf_layout(
    dims: List[Tuple[float, float]],
    order: List[int],
    shelf_width: float,
    pad: float,
) -> Tuple[List[Tuple[float, float]], float, float]:
    """Place bounding boxes onto horizontal shelves and return (placements, total_w, total_h)."""
    placements: List[Tuple[float, float]] = [(0.0, 0.0)] * len(dims)
    cursor_x = 0.0
    cursor_y = 0.0
    shelf_h = 0.0
    packed_w = 0.0
    packed_h = 0.0

    for idx in order:
        w_i, h_i = dims[idx]
        if cursor_x > 0.0 and (cursor_x + w_i) > shelf_width + 1e-9:
            cursor_y += shelf_h + pad
            cursor_x = 0.0
            shelf_h = 0.0

        placements[idx] = (cursor_x, cursor_y)
        packed_w = max(packed_w, cursor_x + w_i)
        packed_h = max(packed_h, cursor_y + h_i)
        cursor_x += w_i + pad
        shelf_h = max(shelf_h, h_i)

    return placements, packed_w, packed_h


def _pack_uv_islands_shelf(
    bm: bmesh.types.BMesh,
    uv_layer,
    islands: List[List[int]],
    force_pack: bool,
) -> Tuple[Set[int], float]:
    """Pack UV islands into [0.005, 0.995] using pure-bmesh shelf packing.

    Scales uniformly only when the packed bounding box exceeds ``_PACK_INNER_SPAN`` (0.99).
    Returns ``(moved_island_indices, pack_scale)``.
    """
    if not islands:
        return set(), 1.0

    aabbs = [_compute_island_aabb(bm, uv_layer, isl) for isl in islands]
    if not force_pack and _islands_already_packed(aabbs):
        return set(), 1.0

    dims = [
        (max(a[1] - a[0], 1e-6), max(a[3] - a[2], 1e-6))
        for a in aabbs
    ]
    order = sorted(range(len(islands)), key=lambda i: (-dims[i][1], -dims[i][0], i))

    total_area = sum(w * h for w, h in dims)
    max_w = max(w for w, _ in dims)
    max_h = max(h for _, h in dims)

    if total_area <= (_PACK_INNER_SPAN * _PACK_INNER_SPAN) and max_w <= _PACK_INNER_SPAN and max_h <= _PACK_INNER_SPAN:
        pad = 0.005
        placements, packed_w, packed_h = _run_shelf_layout(dims, order, _PACK_INNER_SPAN, pad)
        if packed_h > _PACK_INNER_SPAN:
            # Re-balance aspect ratio before uniform shrink so we do not stack into a narrow column.
            balanced_w = max(_PACK_INNER_SPAN, max_w, math.sqrt(packed_w * packed_h * 1.15))
            placements, packed_w, packed_h = _run_shelf_layout(dims, order, balanced_w, pad)
    else:
        est_side = max(max_w, max_h, math.sqrt(total_area * 1.20))
        pad = max(1e-4, 0.005 * (est_side / _PACK_INNER_SPAN))
        placements, packed_w, packed_h = _run_shelf_layout(dims, order, est_side, pad)

    max_dim = max(packed_w, packed_h)
    pack_scale = (_PACK_INNER_SPAN / max_dim) if max_dim > _PACK_INNER_SPAN else 1.0

    moved_indices: Set[int] = set()
    for idx, island in enumerate(islands):
        min_u, _max_u, min_v, _max_v = aabbs[idx]
        px, py = placements[idx]
        island_moved = False
        for f_idx in island:
            for loop in bm.faces[f_idx].loops:
                uv = loop[uv_layer].uv
                new_u = _PACK_MARGIN + (px + (uv.x - min_u)) * pack_scale
                new_v = _PACK_MARGIN + (py + (uv.y - min_v)) * pack_scale
                if abs(new_u - uv.x) > 1e-6 or abs(new_v - uv.y) > 1e-6:
                    island_moved = True
                uv.x = new_u
                uv.y = new_v
        if island_moved:
            moved_indices.add(idx)

    return moved_indices, pack_scale


def equalize_object_texels(
    obj: Optional[bpy.types.Object],
    target_px_per_m: float = constants.TEXEL_DEFAULT_TARGET,
    default_tex_size: int = constants.TEXEL_DEFAULT_RES,
    *,
    mode: str = "RELATIVE",
    pack_islands: bool = True,
    _flush_view_layer: bool = True,
) -> int:
    """Equalize UV island texel density on ``obj`` in ``RELATIVE`` or ``ABSOLUTE`` mode."""
    if obj is None or obj.type != "MESH" or obj.data is None or target_px_per_m <= 0.0:
        return 0
    if getattr(obj, "mode", "OBJECT") == "EDIT":
        return 0

    mesh = obj.data
    if not constants.is_editable_id(mesh):
        return 0
    if len(mesh.polygons) == 0:
        return 0

    eff_mode = str(mode or "RELATIVE").upper()
    if eff_mode not in {"RELATIVE", "ABSOLUTE"}:
        eff_mode = "RELATIVE"

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
    uv_modified = False
    scaled_indices: Set[int] = set()

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

        island_densities: List[Optional[float]] = []
        total_world_area = 0.0
        weighted_density_sum = 0.0

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
                island_densities.append(None)
                continue

            avg_w = w_weighted / island_world_area
            avg_h = h_weighted / island_world_area
            density_area = (island_uv_area * avg_w * avg_h) / island_world_area
            if density_area <= 1e-12:
                island_densities.append(None)
                continue

            density = math.sqrt(density_area)
            if density <= 1e-6:
                island_densities.append(None)
                continue

            island_densities.append(density)
            total_world_area += island_world_area
            weighted_density_sum += density * island_world_area

        avg_density = (weighted_density_sum / total_world_area) if total_world_area > 0.0 else 0.0

        for idx, (island, density) in enumerate(zip(islands, island_densities)):
            if density is None:
                continue

            if eff_mode == "RELATIVE":
                if avg_density <= 1e-6:
                    continue
                s_uv = avg_density / density
            else:
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

            scaled_indices.add(idx)
            uv_modified = True

        if eff_mode == "RELATIVE" and pack_islands:
            packed_indices, pack_scale = _pack_uv_islands_shelf(
                bm,
                uv_layer,
                islands,
                force_pack=bool(scaled_indices),
            )
            if packed_indices:
                uv_modified = True
                scaled_indices.update(packed_indices)
            if abs(pack_scale - 1.0) > 1e-4:
                scaled_indices.update(range(len(islands)))

        if uv_modified:
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
    return len(scaled_indices)


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
    mode: str = "RELATIVE",
    pack_islands: bool = True,
) -> Dict[str, int]:
    """Equalize UV island texel density across mesh objects in ``scene``."""
    if scene is None:
        return {
            "objects_equalized": 0,
            "islands_scaled": 0,
            "meshes_split": 0,
            "uvs_exceed_unit_square": 0,
        }

    update_view_layer(scene)
    from ..core import texel_analyzer

    targets = iter_scene_mesh_objects(scene, selected_only=selected_only)
    target_ptrs = {obj.as_pointer() for obj in targets}

    seen_meshes: Dict[int, Tuple[Tuple[float, ...], Tuple[int, ...]]] = {}
    objects_equalized = 0
    islands_scaled = 0
    meshes_split = 0
    uvs_exceed_unit_square = 0

    for obj in targets:
        if obj.data is None:
            continue
        if getattr(obj, "mode", "OBJECT") == "EDIT":
            continue
        if not constants.is_editable_id(obj.data):
            continue

        # Protect unselected co-users when operating in selected-only mode.
        if selected_only and obj.data.users > 1:
            has_unselected_user = any(
                u.data == obj.data and u.as_pointer() not in target_ptrs
                for u in bpy.data.objects
            )
            if has_unselected_user:
                if not constants.is_editable_id(obj):
                    continue
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
            if not constants.is_editable_id(obj):
                continue
            obj.data = obj.data.copy()
            meshes_split += 1
            ptr = obj.data.as_pointer()

        seen_meshes[ptr] = sig
        scaled = equalize_object_texels(
            obj,
            target_px_per_m=target_px_per_m,
            default_tex_size=default_tex_size,
            mode=mode,
            pack_islands=pack_islands,
            _flush_view_layer=False,
        )
        if scaled > 0:
            objects_equalized += 1
            islands_scaled += scaled
        if check_uvs_exceed_unit_square(obj.data):
            uvs_exceed_unit_square += 1

    return {
        "objects_equalized": objects_equalized,
        "islands_scaled": islands_scaled,
        "meshes_split": meshes_split,
        "uvs_exceed_unit_square": uvs_exceed_unit_square,
    }


class FLIGHTCHECK_OT_equalize_texels(bpy.types.Operator):
    """Equalize UV island texel densities in Relative or Absolute mode."""

    bl_idname = "flightcheck.equalize_texels"
    bl_label = "Equalize Texel Density"
    bl_description = (
        "Equalize UV island texel density in Relative (pack inside [0, 1]) or Absolute (target px/m) mode"
    )
    bl_options = {"REGISTER", "UNDO"}

    mode: EnumProperty(
        name="Equalize Mode",
        description="Choose between non-overlapping relative packing or absolute target px/m scaling",
        items=[
            (
                "RELATIVE",
                "Relative (Preserve 0-1 Layout)",
                "Equalize UV islands to object average density and repack cleanly inside [0, 1]",
            ),
            (
                "ABSOLUTE",
                "Absolute (Target px/m)",
                "Scale UV islands to exact Target Texel Density (may tile beyond [0, 1])",
            ),
        ],
        default="RELATIVE",
        options={"SKIP_SAVE"},
    )
    pack_islands: BoolProperty(
        name="Pack Islands",
        description="Pack UV islands cleanly inside [0.005, 0.995] in Relative mode",
        default=True,
        options={"SKIP_SAVE"},
    )

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return context.scene is not None and context.mode == "OBJECT"

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
        eff_mode = (
            self.mode
            if self.properties.is_property_set("mode")
            else (getattr(fc, "equalize_mode", "RELATIVE") if fc is not None else "RELATIVE")
        )

        with internal_mutation():
            result = equalize_scene_texels(
                scene,
                target_px_per_m=target_td,
                default_tex_size=default_tex,
                selected_only=selected_only,
                mode=eff_mode,
                pack_islands=bool(self.pack_islands),
            )
            report = run_full_scan(scene, write_texel_attr=True)
            sync_wm_issues(context, report)
            tag_view3d_redraw()

        msg = (
            f"FlightCheck ({eff_mode}): Equalized {result['islands_scaled']} UV island(s) "
            f"across {result['objects_equalized']} object(s)."
        )
        warn = result["meshes_split"] > 0 or result["uvs_exceed_unit_square"] > 0
        if result["meshes_split"] > 0:
            msg += f" Split {result['meshes_split']} shared mesh(es)."
        if result["uvs_exceed_unit_square"] > 0:
            msg += f" {result['uvs_exceed_unit_square']} mesh(es) have UVs exceeding [0, 1] (tiling)."

        self.report({"WARNING"} if warn else {"INFO"}, msg)
        return {"FINISHED"}
