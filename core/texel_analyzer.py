"""Per-object and per-face texel density calculation and mesh attribute baking."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Dict, List, Optional, Set, Tuple

import bmesh
import bpy
import numpy as np

from .. import constants
from ..state import Category, DiagnosticIssue, Severity, TexelObjectStats
from . import graph_crawler


def _first_valid_image_in_tree(
    node_tree: Optional[bpy.types.NodeTree],
    visited_trees: Optional[Set[int]] = None,
) -> Optional[Tuple[int, int]]:
    if node_tree is None:
        return None
    if visited_trees is None:
        visited_trees = set()
    ptr = node_tree.as_pointer()
    if ptr in visited_trees:
        return None
    visited_trees.add(ptr)

    for node in node_tree.nodes:
        if node.type == "TEX_IMAGE" and not getattr(node, "mute", False):
            img = getattr(node, "image", None)
            if img is not None and len(img.size) >= 2:
                w, h = int(img.size[0]), int(img.size[1])
                if w > 0 and h > 0:
                    return (w, h)

    for node in node_tree.nodes:
        if node.type == "GROUP" and not getattr(node, "mute", False):
            sub_tree = getattr(node, "node_tree", None)
            found = _first_valid_image_in_tree(sub_tree, visited_trees)
            if found is not None:
                return found

    return None


def get_material_texture_resolution(
    mat: Optional[bpy.types.Material],
    default_size: int = constants.TEXEL_DEFAULT_RES,
) -> Tuple[int, int]:
    """Return ``(width, height)`` of ``mat``'s Base Color image, fallback any valid image or default."""
    fallback = (int(default_size), int(default_size))
    if mat is None or mat.node_tree is None:
        return fallback

    for bsdf in graph_crawler.find_principled_bsdfs(mat.node_tree):
        base_sock = graph_crawler.get_socket(bsdf.inputs, "Base Color")
        for _node, img in graph_crawler.trace_upstream_images(base_sock):
            if img is not None and len(img.size) >= 2:
                w, h = int(img.size[0]), int(img.size[1])
                if w > 0 and h > 0:
                    return (w, h)

    any_img_res = _first_valid_image_in_tree(mat.node_tree)
    if any_img_res is not None:
        return any_img_res
    return fallback


def _iter_target_mesh_objects(
    scene: Optional[bpy.types.Scene],
    selected_only: bool = False,
) -> List[bpy.types.Object]:
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


def _clear_texel_attribute(mesh: bpy.types.Mesh) -> None:
    if not constants.is_editable_id(mesh):
        return
    attr = mesh.attributes.get(constants.TEXEL_ATTR_NAME)
    if attr is None:
        return
    n_loops = len(mesh.loops)
    if attr.data_type == "FLOAT" and attr.domain == "CORNER" and len(attr.data) == n_loops:
        attr.data.foreach_set("value", np.zeros(n_loops, dtype=np.float32))
    else:
        mesh.attributes.remove(attr)


def _write_corner_attribute(mesh: bpy.types.Mesh, loop_values: np.ndarray) -> None:
    if not constants.is_editable_id(mesh):
        return
    attr = mesh.attributes.get(constants.TEXEL_ATTR_NAME)
    if attr is not None and (attr.data_type != "FLOAT" or attr.domain != "CORNER"):
        mesh.attributes.remove(attr)
        attr = None
    if attr is None:
        attr = mesh.attributes.new(
            name=constants.TEXEL_ATTR_NAME,
            type="FLOAT",
            domain="CORNER",
        )
    if len(loop_values) == len(attr.data):
        attr.data.foreach_set("value", np.ascontiguousarray(loop_values, dtype=np.float32))


def _no_uv_result(
    obj_name: str,
    face_count: int,
    target: float,
) -> Tuple[TexelObjectStats, List[DiagnosticIssue]]:
    issue = DiagnosticIssue(
        code="TEXEL_NO_UV",
        category=Category.TEXEL,
        severity=Severity.WARNING,
        message=f"Object '{obj_name}' has {face_count} faces with no active UV map",
        object_name=obj_name,
        expected_range="Active UV map required",
        fixable=False,
    )
    stats = TexelObjectStats(
        object_name=obj_name,
        avg_density_px_per_m=0.0,
        min_density_px_per_m=0.0,
        max_density_px_per_m=0.0,
        target_px_per_m=target,
        face_count=face_count,
        out_of_spec_faces=face_count,
        has_uvs=False,
        severity=Severity.WARNING,
    )
    return stats, [issue]


def _finalize_texel_stats(
    obj_name: str,
    target: float,
    face_count: int,
    face_world_areas: np.ndarray,
    face_uv_areas: np.ndarray,
    face_tex_areas: np.ndarray,
) -> Tuple[TexelObjectStats, List[DiagnosticIssue], np.ndarray]:
    """Compute per-face encoded values, summary stats, and diagnostic issues via NumPy."""
    safe_target = max(target, 1e-6)
    valid_mask = face_world_areas > 1e-12
    valid_world_area = float(np.sum(face_world_areas[valid_mask])) if np.any(valid_mask) else 0.0

    face_encoded = np.zeros(face_count, dtype=np.float32)

    if valid_world_area > 0.0:
        density_area = np.zeros(face_count, dtype=np.float64)
        density_area[valid_mask] = (
            face_uv_areas[valid_mask] * face_tex_areas[valid_mask]
        ) / face_world_areas[valid_mask]
        face_density = np.sqrt(np.maximum(density_area, 0.0))

        valid_densities = face_density[valid_mask]
        valid_areas = face_world_areas[valid_mask]
        avg_density = float(np.sum(valid_densities * valid_areas)) / valid_world_area
        min_density = float(np.min(valid_densities))
        max_density = float(np.max(valid_densities))

        ratios = valid_densities / safe_target
        low_faces = int(np.sum(ratios < constants.TEXEL_RATIO_MIN))
        high_faces = int(np.sum(ratios > constants.TEXEL_RATIO_MAX))
        out_of_spec_faces = low_faces + high_faces

        encoded_valid = (np.log2(np.maximum(ratios, 1e-6)) + 2.0) / 4.0
        face_encoded[valid_mask] = np.clip(
            encoded_valid, constants.TEXEL_ATTR_MIN_VALID, 1.0
        ).astype(np.float32)
    else:
        avg_density = 0.0
        min_density = 0.0
        max_density = 0.0
        low_faces = 0
        high_faces = 0
        out_of_spec_faces = 0

    issues: List[DiagnosticIssue] = []
    avg_ratio = avg_density / safe_target
    low_bound = constants.TEXEL_RATIO_MIN * target
    high_bound = constants.TEXEL_RATIO_MAX * target

    if valid_world_area > 0.0:
        if avg_ratio < constants.TEXEL_RATIO_MIN or (
            low_faces > face_count * 0.5 and low_faces >= high_faces
        ):
            issues.append(
                DiagnosticIssue(
                    code="TEXEL_LOW",
                    category=Category.TEXEL,
                    severity=Severity.WARNING,
                    message=(
                        f"Object '{obj_name}' texel density {avg_density:.1f} px/m "
                        f"is below 0.5x target ({target:.0f} px/m)"
                    ),
                    object_name=obj_name,
                    measured_value=avg_density,
                    expected_range=f"[{low_bound:.0f}, {high_bound:.0f}] px/m",
                    fixable=True,
                )
            )
        elif avg_ratio > constants.TEXEL_RATIO_MAX or (
            high_faces > face_count * 0.5 and high_faces > low_faces
        ):
            issues.append(
                DiagnosticIssue(
                    code="TEXEL_HIGH",
                    category=Category.TEXEL,
                    severity=Severity.WARNING,
                    message=(
                        f"Object '{obj_name}' texel density {avg_density:.1f} px/m "
                        f"exceeds 2.0x target ({target:.0f} px/m)"
                    ),
                    object_name=obj_name,
                    measured_value=avg_density,
                    expected_range=f"[{low_bound:.0f}, {high_bound:.0f}] px/m",
                    fixable=True,
                )
            )

    severity = Severity.WARNING if issues else Severity.OK
    stats = TexelObjectStats(
        object_name=obj_name,
        avg_density_px_per_m=avg_density,
        min_density_px_per_m=min_density,
        max_density_px_per_m=max_density,
        target_px_per_m=target,
        face_count=face_count,
        out_of_spec_faces=out_of_spec_faces,
        has_uvs=True,
        severity=severity,
    )
    return stats, issues, face_encoded


def _slot_texture_areas(
    obj: bpy.types.Object,
    default_tex_size: int,
) -> Tuple[np.ndarray, float]:
    default_area = float(default_tex_size) * float(default_tex_size)
    areas = [
        float(w) * float(h)
        for w, h in (
            get_material_texture_resolution(slot.material, default_size=default_tex_size)
            for slot in obj.material_slots
        )
    ]
    return np.array(areas, dtype=np.float64), default_area


def _analyze_edit_mode(
    obj: bpy.types.Object,
    target: float,
    default_tex_size: int,
) -> Tuple[TexelObjectStats, List[DiagnosticIssue]]:
    """Read live Edit Mode bmesh without freeing or mutating it; never writes attributes."""
    bm = bmesh.from_edit_mesh(obj.data)
    face_count = len(bm.faces)
    uv_layer = bm.loops.layers.uv.active
    if face_count == 0:
        return (
            TexelObjectStats(
                object_name=obj.name,
                avg_density_px_per_m=0.0,
                min_density_px_per_m=0.0,
                max_density_px_per_m=0.0,
                target_px_per_m=target,
                face_count=0,
                out_of_spec_faces=0,
                has_uvs=(uv_layer is not None),
                severity=Severity.OK,
            ),
            [],
        )

    if uv_layer is None:
        return _no_uv_result(obj.name, face_count, target)

    bm.faces.ensure_lookup_table()
    world_mat = obj.matrix_world
    face_world_areas = np.zeros(face_count, dtype=np.float64)
    face_uv_areas = np.zeros(face_count, dtype=np.float64)

    for tri in bm.calc_loop_triangles():
        l0, l1, l2 = tri
        f_idx = l0.face.index
        p0 = world_mat @ l0.vert.co
        p1 = world_mat @ l1.vert.co
        p2 = world_mat @ l2.vert.co
        a_world = 0.5 * (p1 - p0).cross(p2 - p0).length
        uv0, uv1, uv2 = l0[uv_layer].uv, l1[uv_layer].uv, l2[uv_layer].uv
        a_uv = 0.5 * abs((uv1.x - uv0.x) * (uv2.y - uv0.y) - (uv2.x - uv0.x) * (uv1.y - uv0.y))
        face_world_areas[f_idx] += a_world
        face_uv_areas[f_idx] += a_uv

    slot_areas, default_area = _slot_texture_areas(obj, default_tex_size)
    if len(slot_areas) > 0:
        poly_mats = np.array([face.material_index for face in bm.faces], dtype=np.int32)
        valid_slot = (poly_mats >= 0) & (poly_mats < len(slot_areas))
        face_tex_areas = np.where(valid_slot, slot_areas[np.where(valid_slot, poly_mats, 0)], default_area)
    else:
        face_tex_areas = np.full(face_count, default_area, dtype=np.float64)

    stats, issues, _encoded = _finalize_texel_stats(
        obj.name, target, face_count, face_world_areas, face_uv_areas, face_tex_areas
    )
    return stats, issues


def analyze_object_texel_density(
    obj: Optional[bpy.types.Object],
    target_px_per_m: float = constants.TEXEL_DEFAULT_TARGET,
    default_tex_size: int = constants.TEXEL_DEFAULT_RES,
    write_attribute: bool = True,
) -> Tuple[TexelObjectStats, List[DiagnosticIssue]]:
    """Compute world-space texel density per face and optionally write ``__flightcheck_td``."""
    target = float(target_px_per_m)
    if obj is None or obj.type != "MESH" or obj.data is None:
        name = obj.name if obj is not None else ""
        return (
            TexelObjectStats(
                object_name=name,
                avg_density_px_per_m=0.0,
                min_density_px_per_m=0.0,
                max_density_px_per_m=0.0,
                target_px_per_m=target,
                face_count=0,
                out_of_spec_faces=0,
                has_uvs=False,
                severity=Severity.OK,
            ),
            [],
        )

    if getattr(obj, "mode", "OBJECT") == "EDIT":
        return _analyze_edit_mode(obj, target, default_tex_size)

    mesh = obj.data
    can_write = bool(write_attribute and constants.is_editable_id(mesh))
    face_count = len(mesh.polygons)
    uv_layer = mesh.uv_layers.active

    if face_count == 0:
        if can_write:
            _clear_texel_attribute(mesh)
        return (
            TexelObjectStats(
                object_name=obj.name,
                avg_density_px_per_m=0.0,
                min_density_px_per_m=0.0,
                max_density_px_per_m=0.0,
                target_px_per_m=target,
                face_count=0,
                out_of_spec_faces=0,
                has_uvs=(uv_layer is not None),
                severity=Severity.OK,
            ),
            [],
        )

    if uv_layer is None:
        if can_write:
            _clear_texel_attribute(mesh)
        return _no_uv_result(obj.name, face_count, target)

    mesh.calc_loop_triangles()
    n_verts = len(mesh.vertices)
    n_tris = len(mesh.loop_triangles)
    n_loops = len(mesh.loops)

    if n_tris > 0 and n_verts > 0 and n_loops > 0:
        vert_cos = np.empty(n_verts * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", vert_cos)
        verts = vert_cos.reshape(n_verts, 3).astype(np.float64, copy=False)

        world_mat = np.array(obj.matrix_world, dtype=np.float64)
        world_verts = verts @ world_mat[:3, :3].T + world_mat[:3, 3]

        tri_vert_idx = np.empty(n_tris * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", tri_vert_idx)
        tri_vert_idx = tri_vert_idx.reshape(n_tris, 3)

        tri_loop_idx = np.empty(n_tris * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("loops", tri_loop_idx)
        tri_loop_idx = tri_loop_idx.reshape(n_tris, 3)

        tri_poly_idx = np.empty(n_tris, dtype=np.int32)
        mesh.loop_triangles.foreach_get("polygon_index", tri_poly_idx)

        uv_flat = np.empty(n_loops * 2, dtype=np.float32)
        uv_layer.uv.foreach_get("vector", uv_flat)
        uvs = uv_flat.reshape(n_loops, 2).astype(np.float64, copy=False)

        p0 = world_verts[tri_vert_idx[:, 0]]
        p1 = world_verts[tri_vert_idx[:, 1]]
        p2 = world_verts[tri_vert_idx[:, 2]]
        tri_world_areas = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)

        uv0 = uvs[tri_loop_idx[:, 0]]
        uv1 = uvs[tri_loop_idx[:, 1]]
        uv2 = uvs[tri_loop_idx[:, 2]]
        du1 = uv1 - uv0
        du2 = uv2 - uv0
        tri_uv_areas = 0.5 * np.abs(du1[:, 0] * du2[:, 1] - du2[:, 0] * du1[:, 1])

        face_world_areas = np.bincount(tri_poly_idx, weights=tri_world_areas, minlength=face_count)
        face_uv_areas = np.bincount(tri_poly_idx, weights=tri_uv_areas, minlength=face_count)
    else:
        face_world_areas = np.zeros(face_count, dtype=np.float64)
        face_uv_areas = np.zeros(face_count, dtype=np.float64)

    slot_areas, default_area = _slot_texture_areas(obj, default_tex_size)
    if len(slot_areas) > 0:
        poly_mats = np.empty(face_count, dtype=np.int32)
        mesh.polygons.foreach_get("material_index", poly_mats)
        valid_slot = (poly_mats >= 0) & (poly_mats < len(slot_areas))
        face_tex_areas = np.where(
            valid_slot,
            slot_areas[np.where(valid_slot, poly_mats, 0)],
            default_area,
        )
    else:
        face_tex_areas = np.full(face_count, default_area, dtype=np.float64)

    stats, issues, face_encoded = _finalize_texel_stats(
        obj.name, target, face_count, face_world_areas, face_uv_areas, face_tex_areas
    )

    if can_write:
        poly_loop_totals = np.empty(face_count, dtype=np.int32)
        mesh.polygons.foreach_get("loop_total", poly_loop_totals)
        loop_values = np.repeat(face_encoded, poly_loop_totals)
        _write_corner_attribute(mesh, loop_values)

    return stats, issues


def scan_scene_texels(
    scene: Optional[bpy.types.Scene],
    target_px_per_m: float = constants.TEXEL_DEFAULT_TARGET,
    default_tex_size: int = constants.TEXEL_DEFAULT_RES,
    selected_only: bool = False,
    write_attribute: bool = True,
) -> Tuple[Dict[str, TexelObjectStats], List[DiagnosticIssue]]:
    """Scan mesh objects in ``scene`` for texel density compliance and shared-mesh conflicts."""
    stats_map: Dict[str, TexelObjectStats] = {}
    issues: List[DiagnosticIssue] = []

    if scene is None:
        return stats_map, issues

    targets = _iter_target_mesh_objects(scene, selected_only=selected_only)
    if not targets:
        return stats_map, issues

    scene_mesh_users: Dict[int, List[bpy.types.Object]] = defaultdict(list)
    for obj in scene.objects:
        if obj.type == "MESH" and obj.data is not None:
            scene_mesh_users[obj.data.as_pointer()].append(obj)

    grouped_targets: Dict[int, List[bpy.types.Object]] = defaultdict(list)
    no_data_targets: List[bpy.types.Object] = []
    for obj in targets:
        if obj.data is None:
            no_data_targets.append(obj)
        else:
            grouped_targets[obj.data.as_pointer()].append(obj)

    for obj in no_data_targets:
        stats, obj_issues = analyze_object_texel_density(
            obj,
            target_px_per_m=target_px_per_m,
            default_tex_size=default_tex_size,
            write_attribute=False,
        )
        stats_map[obj.name] = stats
        issues.extend(obj_issues)

    for mesh_ptr, group_objs in grouped_targets.items():
        sorted_group = sorted(group_objs, key=lambda o: o.name)
        co_users = sorted(scene_mesh_users.get(mesh_ptr, sorted_group), key=lambda o: o.name)

        if len(co_users) > 1:
            scales = [
                (abs(float(s.x)), abs(float(s.y)), abs(float(s.z)))
                for s in (u.matrix_world.to_scale() for u in co_users)
            ]
            max_diff = max(
                max(sc[axis] for sc in scales) - min(sc[axis] for sc in scales)
                for axis in range(3)
            )
            if max_diff > 1e-3:
                mesh_name = sorted_group[0].data.name
                issues.append(
                    DiagnosticIssue(
                        code="TEXEL_SHARED_MESH_SCALE",
                        category=Category.TEXEL,
                        severity=Severity.INFO,
                        message=(
                            f"Mesh '{mesh_name}' is shared by {len(co_users)} objects with differing "
                            f"world scales (max diff {max_diff:.3f}); texel attribute reflects "
                            f"'{sorted_group[0].name}'"
                        ),
                        object_name=sorted_group[0].name,
                        measured_value=max_diff,
                        expected_range="<= 0.001 scale diff",
                        fixable=False,
                    )
                )

        for idx, obj in enumerate(sorted_group):
            stats, obj_issues = analyze_object_texel_density(
                obj,
                target_px_per_m=target_px_per_m,
                default_tex_size=default_tex_size,
                write_attribute=(write_attribute and idx == 0),
            )
            stats_map[obj.name] = stats
            issues.extend(obj_issues)

    return stats_map, issues


def remove_texel_attributes(scene: Optional[bpy.types.Scene] = None) -> int:
    """Remove ``__flightcheck_td`` attributes from editable meshes in ``scene`` or all meshes."""
    removed = 0
    if scene is None:
        meshes = list(bpy.data.meshes)
    else:
        seen: Set[int] = set()
        meshes = []
        for obj in scene.objects:
            if obj.type != "MESH" or obj.data is None:
                continue
            ptr = obj.data.as_pointer()
            if ptr in seen:
                continue
            seen.add(ptr)
            meshes.append(obj.data)

    for mesh in meshes:
        if not constants.is_editable_id(mesh):
            continue
        attr = mesh.attributes.get(constants.TEXEL_ATTR_NAME)
        if attr is not None:
            mesh.attributes.remove(attr)
            removed += 1

    return removed
