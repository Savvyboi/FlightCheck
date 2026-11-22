"""Per-object and per-face texel density calculation and mesh attribute baking."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import bmesh
import bpy

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
        if node.type == "TEX_IMAGE":
            img = getattr(node, "image", None)
            if img is not None and len(img.size) >= 2:
                w, h = int(img.size[0]), int(img.size[1])
                if w > 0 and h > 0:
                    return (w, h)

    for node in node_tree.nodes:
        if node.type == "GROUP":
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
    attr = mesh.attributes.get(constants.TEXEL_ATTR_NAME)
    if attr is None:
        return
    if attr.data_type == "FLOAT" and attr.domain == "CORNER" and len(attr.data) == len(mesh.loops):
        attr.data.foreach_set("value", [0.0] * len(mesh.loops))
    else:
        mesh.attributes.remove(attr)


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

    mesh = obj.data
    face_count = len(mesh.polygons)
    if face_count == 0:
        if write_attribute:
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
                has_uvs=bool(mesh.uv_layers.active),
                severity=Severity.OK,
            ),
            [],
        )

    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bmesh.ops.transform(bm, matrix=obj.matrix_world, verts=bm.verts)
        bm.faces.ensure_lookup_table()
        uv_layer = bm.loops.layers.uv.active

        if uv_layer is None:
            if write_attribute:
                _clear_texel_attribute(mesh)
            issue = DiagnosticIssue(
                code="TEXEL_NO_UV",
                category=Category.TEXEL,
                severity=Severity.WARNING,
                message=f"Object '{obj.name}' has {face_count} faces with no active UV map",
                object_name=obj.name,
                expected_range="Active UV map required",
                fixable=False,
            )
            stats = TexelObjectStats(
                object_name=obj.name,
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

        default_res = (int(default_tex_size), int(default_tex_size))
        slot_res: List[Tuple[int, int]] = [
            get_material_texture_resolution(slot.material, default_size=default_tex_size)
            for slot in obj.material_slots
        ]

        n_faces = len(bm.faces)
        face_world_areas = [0.0] * n_faces
        face_uv_areas = [0.0] * n_faces

        for tri in bm.calc_loop_triangles():
            l0, l1, l2 = tri
            f_idx = l0.face.index
            p0, p1, p2 = l0.vert.co, l1.vert.co, l2.vert.co
            a_world = 0.5 * (p1 - p0).cross(p2 - p0).length
            uv0, uv1, uv2 = l0[uv_layer].uv, l1[uv_layer].uv, l2[uv_layer].uv
            a_uv = 0.5 * abs((uv1.x - uv0.x) * (uv2.y - uv0.y) - (uv2.x - uv0.x) * (uv1.y - uv0.y))
            face_world_areas[f_idx] += a_world
            face_uv_areas[f_idx] += a_uv

        valid_world_area = 0.0
        weighted_density_sum = 0.0
        min_density = math.inf
        max_density = 0.0
        low_faces = 0
        high_faces = 0
        out_of_spec_faces = 0
        loop_values: List[float] = []
        safe_target = max(target, 1e-6)

        for face in bm.faces:
            a_world = face_world_areas[face.index]
            a_uv = face_uv_areas[face.index]
            mat_idx = face.material_index
            w, h = slot_res[mat_idx] if 0 <= mat_idx < len(slot_res) else default_res

            if a_world <= 1e-12:
                encoded = 0.0
            else:
                density_area = (a_uv * float(w) * float(h)) / a_world
                density_px_per_m = math.sqrt(max(density_area, 0.0))
                ratio = density_px_per_m / safe_target
                encoded = constants.normalize_texel_ratio(ratio)

                valid_world_area += a_world
                weighted_density_sum += density_px_per_m * a_world
                if density_px_per_m < min_density:
                    min_density = density_px_per_m
                if density_px_per_m > max_density:
                    max_density = density_px_per_m

                if ratio < constants.TEXEL_RATIO_MIN:
                    low_faces += 1
                    out_of_spec_faces += 1
                elif ratio > constants.TEXEL_RATIO_MAX:
                    high_faces += 1
                    out_of_spec_faces += 1

            if write_attribute:
                loop_values.extend([encoded] * len(face.loops))

        if valid_world_area > 0.0:
            avg_density = weighted_density_sum / valid_world_area
        else:
            avg_density = 0.0
            min_density = 0.0
            max_density = 0.0

        if write_attribute:
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
                attr.data.foreach_set("value", loop_values)

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
                            f"Object '{obj.name}' texel density {avg_density:.1f} px/m "
                            f"is below 0.5x target ({target:.0f} px/m)"
                        ),
                        object_name=obj.name,
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
                            f"Object '{obj.name}' texel density {avg_density:.1f} px/m "
                            f"exceeds 2.0x target ({target:.0f} px/m)"
                        ),
                        object_name=obj.name,
                        measured_value=avg_density,
                        expected_range=f"[{low_bound:.0f}, {high_bound:.0f}] px/m",
                        fixable=True,
                    )
                )

        severity = Severity.WARNING if issues else Severity.OK
        stats = TexelObjectStats(
            object_name=obj.name,
            avg_density_px_per_m=avg_density,
            min_density_px_per_m=min_density,
            max_density_px_per_m=max_density,
            target_px_per_m=target,
            face_count=face_count,
            out_of_spec_faces=out_of_spec_faces,
            has_uvs=True,
            severity=severity,
        )
        return stats, issues
    finally:
        bm.free()


def scan_scene_texels(
    scene: Optional[bpy.types.Scene],
    target_px_per_m: float = constants.TEXEL_DEFAULT_TARGET,
    default_tex_size: int = constants.TEXEL_DEFAULT_RES,
    selected_only: bool = False,
    write_attribute: bool = True,
) -> Tuple[Dict[str, TexelObjectStats], List[DiagnosticIssue]]:
    """Scan mesh objects in ``scene`` for texel density compliance and missing UVs."""
    stats_map: Dict[str, TexelObjectStats] = {}
    issues: List[DiagnosticIssue] = []

    for obj in _iter_target_mesh_objects(scene, selected_only=selected_only):
        stats, obj_issues = analyze_object_texel_density(
            obj,
            target_px_per_m=target_px_per_m,
            default_tex_size=default_tex_size,
            write_attribute=write_attribute,
        )
        stats_map[obj.name] = stats
        issues.extend(obj_issues)

    return stats_map, issues


def remove_texel_attributes(scene: Optional[bpy.types.Scene] = None) -> int:
    """Remove ``__flightcheck_td`` attributes from meshes in ``scene`` or all ``bpy.data.meshes``."""
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
        attr = mesh.attributes.get(constants.TEXEL_ATTR_NAME)
        if attr is not None:
            mesh.attributes.remove(attr)
            removed += 1

    return removed
