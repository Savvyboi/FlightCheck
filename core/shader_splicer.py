"""Non-destructive viewport overlay splicing and PBR clamp restoration engine."""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Set, Tuple

import bpy

from .. import constants
from .pbr_shader_nodes import (
    ensure_ctrl_group,
    ensure_overlay_group,
    get_socket,
    remove_flightcheck_node_groups,
)

__all__ = [
    "is_material_spliced",
    "is_material_clamped",
    "splice_material",
    "resync_material",
    "unsplice_material",
    "splice_scene_materials",
    "unsplice_scene_materials",
    "unsplice_clamp_material",
    "cleanup_overlays",
    "cleanup_all",
]

_TAPPED_BSDF_SOCKETS: Tuple[str, ...] = (
    "Base Color",
    "Metallic",
    "Roughness",
    "Specular IOR Level",
    "IOR",
    "Transmission Weight",
)

_CLAMP_SOCKET_MAP: Tuple[Tuple[str, str], ...] = (
    ("Color", "Color"),
    ("Metallic", "Metallic"),
    ("Roughness", "Roughness"),
    ("Specular IOR Level", "Specular IOR Level"),
    ("IOR", "IOR"),
)


def _is_overlay_node(node: Optional[bpy.types.Node]) -> bool:
    if node is None:
        return False
    if node.name == constants.OVERLAY_NODE_NAME:
        return True
    return (
        node.type == "GROUP"
        and getattr(node, "node_tree", None) is not None
        and node.node_tree.name == constants.OVERLAY_GROUP_NAME
    )


def _is_ctrl_node(node: Optional[bpy.types.Node]) -> bool:
    if node is None:
        return False
    if node.name == constants.CTRL_NODE_NAME:
        return True
    return (
        node.type == "GROUP"
        and getattr(node, "node_tree", None) is not None
        and node.node_tree.name == constants.CTRL_GROUP_NAME
    )


def _is_clamp_node(node: Optional[bpy.types.Node]) -> bool:
    if node is None:
        return False
    if node.name == constants.CLAMP_NODE_NAME:
        return True
    return (
        node.type == "GROUP"
        and getattr(node, "node_tree", None) is not None
        and node.node_tree.name == constants.CLAMP_GROUP_NAME
    )


def _find_active_material_output(node_tree: bpy.types.NodeTree) -> Optional[bpy.types.Node]:
    outputs = [n for n in node_tree.nodes if n.type == "OUTPUT_MATERIAL"]
    if not outputs:
        return None
    for node in outputs:
        if getattr(node, "is_active_output", False) and getattr(node, "target", "ALL") == "ALL":
            return node
    for node in outputs:
        if getattr(node, "is_active_output", False):
            return node
    return outputs[0]


def _find_active_principled_bsdf(
    node_tree: bpy.types.NodeTree,
    mat_output: Optional[bpy.types.Node],
) -> Optional[bpy.types.Node]:
    if mat_output is not None:
        surf_in = get_socket(mat_output.inputs, "Surface")
        if surf_in is not None and surf_in.is_linked:
            queue: deque[bpy.types.Node] = deque()
            visited: Set[bpy.types.Node] = set()
            for link in surf_in.links:
                if link.from_node is not None:
                    queue.append(link.from_node)
            while queue:
                node = queue.popleft()
                if node in visited or _is_ctrl_node(node):
                    continue
                visited.add(node)
                if node.type == "BSDF_PRINCIPLED":
                    return node
                if _is_overlay_node(node):
                    shader_in = get_socket(node.inputs, "Shader")
                    if shader_in is not None:
                        for link in shader_in.links:
                            if link.from_node is not None and link.from_node not in visited:
                                queue.append(link.from_node)
                    continue
                for sock in node.inputs:
                    if sock.is_linked:
                        for link in sock.links:
                            if link.from_node is not None and link.from_node not in visited:
                                queue.append(link.from_node)

    for node in node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    return None


def _clear_socket_links(nt: bpy.types.NodeTree, socket: Optional[bpy.types.NodeSocket]) -> None:
    if socket is None or not socket.is_linked:
        return
    for link in list(socket.links):
        nt.links.remove(link)


def _copy_socket_default(src: bpy.types.NodeSocket, dst: bpy.types.NodeSocket) -> None:
    if dst.type == "RGBA" and src.type == "RGBA":
        dst.default_value = tuple(src.default_value)
    elif dst.type == "VALUE" and src.type == "VALUE":
        dst.default_value = float(src.default_value)
    elif dst.type == "VECTOR" and src.type == "VECTOR":
        dst.default_value = tuple(src.default_value)


def _ensure_group_node(
    nt: bpy.types.NodeTree,
    node_name: str,
    group_tree: bpy.types.ShaderNodeTree,
    default_loc: Tuple[float, float],
) -> bpy.types.Node:
    node = nt.nodes.get(node_name)
    if node is not None and node.type != "GROUP":
        nt.nodes.remove(node)
        node = None
    if node is None:
        node = nt.nodes.new("ShaderNodeGroup")
        node.name = node_name
        node.label = node_name
        node.location = default_loc
    if node.node_tree != group_tree:
        node.node_tree = group_tree
    return node


def _sync_metallic_linked_input(
    nt: bpy.types.NodeTree,
    bsdf: bpy.types.Node,
    overlay_node: bpy.types.Node,
) -> None:
    ov_met_linked = get_socket(overlay_node.inputs, "Metallic Linked", socket_type="VALUE")
    if ov_met_linked is None:
        return
    _clear_socket_links(nt, ov_met_linked)

    bsdf_met = get_socket(bsdf.inputs, "Metallic")
    if bsdf_met is None or not bsdf_met.is_linked:
        ov_met_linked.default_value = 0.0
        return

    src_node = bsdf_met.links[0].from_node
    if _is_clamp_node(src_node):
        clamp_met_in = get_socket(src_node.inputs, "Metallic")
        if clamp_met_in is not None and clamp_met_in.is_linked:
            ov_met_linked.default_value = 1.0
            return
        clamp_linked_in = get_socket(src_node.inputs, "Metallic Linked", socket_type="VALUE")
        if clamp_linked_in is not None:
            if clamp_linked_in.is_linked:
                nt.links.new(clamp_linked_in.links[0].from_socket, ov_met_linked)
                return
            ov_met_linked.default_value = 1.0 if float(clamp_linked_in.default_value) >= 0.5 else 0.0
            return
        ov_met_linked.default_value = 0.0
        return

    ov_met_linked.default_value = 1.0


def is_material_spliced(mat: Optional[bpy.types.Material]) -> bool:
    """Return True if ``mat`` has an active ``__FLIGHTCHECK_OVERLAY_NODE__`` wired to output."""
    if mat is None or getattr(mat, "node_tree", None) is None:
        return False
    nt = mat.node_tree
    overlay_node = nt.nodes.get(constants.OVERLAY_NODE_NAME)
    if overlay_node is None or overlay_node.type != "GROUP":
        return False
    if getattr(overlay_node, "node_tree", None) is None or overlay_node.node_tree.name != constants.OVERLAY_GROUP_NAME:
        return False
    mat_output = _find_active_material_output(nt)
    if mat_output is None:
        return False
    surface_in = get_socket(mat_output.inputs, "Surface", socket_type="SHADER")
    if surface_in is None or not surface_in.is_linked:
        return False
    return any(link.from_node == overlay_node for link in surface_in.links)


def is_material_clamped(mat: Optional[bpy.types.Material]) -> bool:
    """Return True if ``mat`` contains an active ``__FLIGHTCHECK_CLAMP_NODE__``."""
    if mat is None or getattr(mat, "node_tree", None) is None:
        return False
    return any(_is_clamp_node(node) for node in mat.node_tree.nodes)


def splice_material(mat: Optional[bpy.types.Material]) -> bool:
    """Splice or re-sync the non-destructive diagnostic overlay node in ``mat``."""
    if mat is None or getattr(mat, "node_tree", None) is None:
        return False

    nt = mat.node_tree
    mat_output = _find_active_material_output(nt)
    if mat_output is None:
        return False

    bsdf = _find_active_principled_bsdf(nt, mat_output)
    if bsdf is None:
        return False

    ctrl_group = ensure_ctrl_group()
    overlay_group = ensure_overlay_group()

    out_loc = mat_output.location
    overlay_node = _ensure_group_node(
        nt,
        constants.OVERLAY_NODE_NAME,
        overlay_group,
        (out_loc.x - 240.0, out_loc.y),
    )
    ctrl_node = _ensure_group_node(
        nt,
        constants.CTRL_NODE_NAME,
        ctrl_group,
        (overlay_node.location.x - 220.0, overlay_node.location.y - 260.0),
    )

    ctrl_mode_out = get_socket(ctrl_node.outputs, "Mode", socket_type="VALUE")
    ov_mode_in = get_socket(overlay_node.inputs, "Mode", socket_type="VALUE")
    if ctrl_mode_out is not None and ov_mode_in is not None:
        if not (ov_mode_in.is_linked and ov_mode_in.links[0].from_socket == ctrl_mode_out):
            _clear_socket_links(nt, ov_mode_in)
            nt.links.new(ctrl_mode_out, ov_mode_in)

    surface_in = get_socket(mat_output.inputs, "Surface", socket_type="SHADER")
    ov_shader_in = get_socket(overlay_node.inputs, "Shader", socket_type="SHADER")
    ov_shader_out = get_socket(overlay_node.outputs, "Shader", socket_type="SHADER")
    if surface_in is None or ov_shader_in is None or ov_shader_out is None:
        return False

    already_wired = surface_in.is_linked and any(
        link.from_node == overlay_node for link in surface_in.links
    )
    if not already_wired:
        orig_from = surface_in.links[0].from_socket if surface_in.is_linked else None
        _clear_socket_links(nt, surface_in)
        _clear_socket_links(nt, ov_shader_in)
        if orig_from is not None and orig_from.node != overlay_node:
            nt.links.new(orig_from, ov_shader_in)
        nt.links.new(ov_shader_out, surface_in)

    for sock_name in _TAPPED_BSDF_SOCKETS:
        bsdf_in = get_socket(bsdf.inputs, sock_name)
        ov_in = get_socket(overlay_node.inputs, sock_name)
        if bsdf_in is None or ov_in is None:
            continue
        _clear_socket_links(nt, ov_in)
        if bsdf_in.is_linked:
            src_sock = bsdf_in.links[0].from_socket
            if src_sock.node != overlay_node:
                nt.links.new(src_sock, ov_in)
        else:
            _copy_socket_default(bsdf_in, ov_in)

    _sync_metallic_linked_input(nt, bsdf, overlay_node)
    return True


def resync_material(mat: Optional[bpy.types.Material]) -> bool:
    """Re-sync tapped Principled BSDF connections on an already spliced material."""
    if not is_material_spliced(mat):
        return False
    return splice_material(mat)


def unsplice_material(mat: Optional[bpy.types.Material]) -> bool:
    """Restore original Material Output links and remove overlay/ctrl nodes from ``mat``."""
    if mat is None or getattr(mat, "node_tree", None) is None:
        return False

    nt = mat.node_tree
    overlay_nodes = [n for n in nt.nodes if _is_overlay_node(n)]
    ctrl_nodes = [n for n in nt.nodes if _is_ctrl_node(n)]
    if not overlay_nodes and not ctrl_nodes:
        return False

    for overlay_node in overlay_nodes:
        ov_shader_in = get_socket(overlay_node.inputs, "Shader", socket_type="SHADER")
        ov_shader_out = get_socket(overlay_node.outputs, "Shader", socket_type="SHADER")
        orig_from = (
            ov_shader_in.links[0].from_socket
            if ov_shader_in is not None and ov_shader_in.is_linked
            else None
        )
        if ov_shader_out is not None and ov_shader_out.is_linked:
            targets = [link.to_socket for link in list(ov_shader_out.links)]
            _clear_socket_links(nt, ov_shader_out)
            if orig_from is not None and orig_from.node != overlay_node:
                for dst_sock in targets:
                    nt.links.new(orig_from, dst_sock)
        nt.nodes.remove(overlay_node)

    for ctrl_node in ctrl_nodes:
        nt.nodes.remove(ctrl_node)

    return True


def unsplice_clamp_material(mat: Optional[bpy.types.Material]) -> bool:
    """Restore pre-clamp links or default values, remove clamp nodes, and re-sync if spliced."""
    if mat is None or getattr(mat, "node_tree", None) is None:
        return False

    nt = mat.node_tree
    clamp_nodes = [n for n in nt.nodes if _is_clamp_node(n)]
    if not clamp_nodes:
        return False

    for clamp_node in clamp_nodes:
        for in_name, out_name in _CLAMP_SOCKET_MAP:
            c_in = get_socket(clamp_node.inputs, in_name)
            c_out = get_socket(clamp_node.outputs, out_name)
            if c_out is None or not c_out.is_linked:
                continue

            targets = [link.to_socket for link in list(c_out.links)]
            _clear_socket_links(nt, c_out)

            if c_in is not None and c_in.is_linked:
                orig_from = c_in.links[0].from_socket
                if orig_from.node != clamp_node:
                    for dst_sock in targets:
                        nt.links.new(orig_from, dst_sock)
            elif c_in is not None:
                for dst_sock in targets:
                    _copy_socket_default(c_in, dst_sock)

        nt.nodes.remove(clamp_node)

    if is_material_spliced(mat):
        resync_material(mat)
    return True


def _iter_scene_materials(
    scene: Optional[bpy.types.Scene],
    selected_only: bool = False,
) -> List[bpy.types.Material]:
    if scene is None:
        return []

    view_layer = None
    if selected_only:
        ctx = getattr(bpy, "context", None)
        if ctx is not None and getattr(ctx, "scene", None) == scene and getattr(ctx, "view_layer", None) is not None:
            view_layer = ctx.view_layer
        elif len(scene.view_layers) > 0:
            view_layer = scene.view_layers[0]

    seen: Set[int] = set()
    materials: List[bpy.types.Material] = []
    for obj in scene.objects:
        if selected_only:
            try:
                is_selected = obj.select_get(view_layer=view_layer) if view_layer is not None else obj.select_get()
            except RuntimeError:
                is_selected = False
            if not is_selected:
                continue
        for slot in getattr(obj, "material_slots", ()):
            mat = slot.material
            if mat is None:
                continue
            ptr = mat.as_pointer()
            if ptr not in seen:
                seen.add(ptr)
                materials.append(mat)
    return materials


def splice_scene_materials(
    scene: Optional[bpy.types.Scene],
    selected_only: bool = False,
) -> int:
    """Splice diagnostic overlay nodes into unique materials used in ``scene``."""
    if scene is None:
        ctx = getattr(bpy, "context", None)
        scene = getattr(ctx, "scene", None) if ctx is not None else None
    count = 0
    for mat in _iter_scene_materials(scene, selected_only=selected_only):
        if splice_material(mat):
            count += 1
    return count


def unsplice_scene_materials(scene: Optional[bpy.types.Scene] = None) -> int:
    """Unsplice diagnostic overlay nodes from scene materials (or all materials if ``scene`` is None)."""
    targets: Iterable[bpy.types.Material] = (
        bpy.data.materials if scene is None else _iter_scene_materials(scene, selected_only=False)
    )
    count = 0
    for mat in list(targets):
        if unsplice_material(mat):
            count += 1
    return count


def cleanup_overlays(scene: Optional[bpy.types.Scene] = None) -> int:
    """Unsplice all viewport diagnostic overlays and remove unused FlightCheck node groups."""
    count = unsplice_scene_materials(scene=scene)
    remove_flightcheck_node_groups(force=False)
    return count


def cleanup_all(remove_clamps: bool = True) -> Dict[str, int]:
    """Remove all overlay splices, optionally revert PBR clamps, and purge unused FlightCheck groups."""
    overlays_removed = 0
    clamps_removed = 0
    for mat in list(bpy.data.materials):
        if unsplice_material(mat):
            overlays_removed += 1
        if remove_clamps and unsplice_clamp_material(mat):
            clamps_removed += 1
    groups_removed = remove_flightcheck_node_groups(force=False)
    return {
        "overlays_removed": overlays_removed,
        "clamps_removed": clamps_removed,
        "groups_removed": groups_removed,
    }
