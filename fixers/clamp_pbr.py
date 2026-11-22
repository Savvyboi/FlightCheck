"""Non-destructive soft-knee PBR clamping fixer for Principled BSDF materials."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import bpy

from .. import constants
from . import collect_scene_materials

_CLAMP_SOCKET_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("Base Color", "Color"),
    ("Metallic", "Metallic"),
    ("Roughness", "Roughness"),
    ("Specular IOR Level", "Specular IOR Level"),
    ("IOR", "IOR"),
)


def _find_existing_clamp_nodes(node_tree: bpy.types.NodeTree) -> List[bpy.types.Node]:
    """Return existing FlightCheck PBR clamp group nodes in ``node_tree``."""
    clamps: List[bpy.types.Node] = []
    for node in node_tree.nodes:
        if node.name == constants.CLAMP_NODE_NAME:
            clamps.append(node)
        elif (
            node.type == "GROUP"
            and getattr(node, "node_tree", None) is not None
            and node.node_tree.name == constants.CLAMP_GROUP_NAME
        ):
            clamps.append(node)
    return clamps


def _update_clamp_parameters(
    clamp_node: bpy.types.Node,
    clamp_group: bpy.types.NodeTree,
    dielectric_knee: float,
    dielectric_max: float,
    get_socket,
) -> None:
    """Ensure clamp node group reference and update knee/max inputs by socket name."""
    if getattr(clamp_node, "node_tree", None) != clamp_group:
        clamp_node.node_tree = clamp_group

    knee_sock = get_socket(clamp_node.inputs, "Dielectric Knee")
    if knee_sock is not None:
        knee_sock.default_value = float(dielectric_knee)

    max_sock = get_socket(clamp_node.inputs, "Dielectric Max")
    if max_sock is not None:
        max_sock.default_value = float(dielectric_max)


def _bsdf_needs_clamping(bsdf: bpy.types.Node, get_socket) -> bool:
    """Return True if any clamped socket is linked or has an out-of-spec unlinked value."""
    for bsdf_name, _ in _CLAMP_SOCKET_PAIRS:
        sock = get_socket(bsdf.inputs, bsdf_name)
        if sock is not None and sock.is_linked:
            return True

    sock_color = get_socket(bsdf.inputs, "Base Color")
    sock_metal = get_socket(bsdf.inputs, "Metallic")
    sock_rough = get_socket(bsdf.inputs, "Roughness")
    sock_spec = get_socket(bsdf.inputs, "Specular IOR Level")
    sock_ior = get_socket(bsdf.inputs, "IOR")
    sock_trans = get_socket(bsdf.inputs, "Transmission Weight")

    metallic = float(sock_metal.default_value) if sock_metal is not None else 0.0
    if metallic < 0.0 or metallic > 1.0:
        return True
    if constants.SEMI_METAL_LOW < metallic < constants.SEMI_METAL_HIGH:
        return True

    base_col = tuple(sock_color.default_value) if sock_color is not None else (0.5, 0.5, 0.5, 1.0)
    luma = constants.linear_luminance(base_col)
    if metallic >= constants.METALLIC_CONDUCTOR_THRESHOLD:
        if luma < constants.CONDUCTOR_MIN_LUMA:
            return True
    else:
        if luma < constants.DIELECTRIC_MIN_LUMA or luma > constants.DIELECTRIC_WARN_LUMA:
            return True

    trans_linked = bool(sock_trans is not None and sock_trans.is_linked)
    transmission = (
        float(sock_trans.default_value)
        if (sock_trans is not None and not trans_linked)
        else 0.0
    )
    if not trans_linked and transmission < constants.TRANSMISSION_THRESHOLD:
        roughness = float(sock_rough.default_value) if sock_rough is not None else 0.5
        if roughness < constants.ROUGHNESS_MIN or roughness > constants.ROUGHNESS_MAX:
            return True

    specular = float(sock_spec.default_value) if sock_spec is not None else 0.5
    if specular < constants.SPECULAR_MIN or specular > constants.SPECULAR_MAX:
        return True

    ior = float(sock_ior.default_value) if sock_ior is not None else 1.5
    if ior < constants.IOR_MIN or ior > constants.IOR_MAX:
        return True

    return False


def _insert_clamp_for_bsdf(
    node_tree: bpy.types.NodeTree,
    bsdf: bpy.types.Node,
    clamp_group: bpy.types.NodeTree,
    dielectric_knee: float,
    dielectric_max: float,
    get_socket,
) -> bpy.types.Node:
    """Splice a __FLIGHTCHECK_CLAMP_NODE__ immediately upstream of ``bsdf``."""
    clamp_node = node_tree.nodes.new("ShaderNodeGroup")
    clamp_node.node_tree = clamp_group
    if constants.CLAMP_NODE_NAME not in node_tree.nodes:
        clamp_node.name = constants.CLAMP_NODE_NAME
    clamp_node.label = "FlightCheck PBR Clamp"
    clamp_node.location = (bsdf.location.x - 240.0, bsdf.location.y)

    metallic_was_linked = False

    for bsdf_name, clamp_name in _CLAMP_SOCKET_PAIRS:
        bsdf_in = get_socket(bsdf.inputs, bsdf_name)
        clamp_in = get_socket(clamp_node.inputs, clamp_name)
        clamp_out = get_socket(clamp_node.outputs, clamp_name)
        if bsdf_in is None or clamp_in is None or clamp_out is None:
            continue

        if bsdf_in.is_linked:
            if bsdf_name == "Metallic":
                metallic_was_linked = True
            upstream_sockets = [link.from_socket for link in bsdf_in.links]
            for link in list(bsdf_in.links):
                node_tree.links.remove(link)
            for from_sock in upstream_sockets:
                node_tree.links.new(from_sock, clamp_in)
        else:
            if clamp_name == "Color":
                vals = list(bsdf_in.default_value)
                rgba = (
                    float(vals[0]) if len(vals) > 0 else 0.0,
                    float(vals[1]) if len(vals) > 1 else 0.0,
                    float(vals[2]) if len(vals) > 2 else 0.0,
                    float(vals[3]) if len(vals) > 3 else 1.0,
                )
                clamp_in.default_value = rgba
            else:
                clamp_in.default_value = float(bsdf_in.default_value)

        node_tree.links.new(clamp_out, bsdf_in)

    # Read Transmission Weight only to drive roughness bypass; never overwrite BSDF input.
    trans_bsdf_in = get_socket(bsdf.inputs, "Transmission Weight")
    trans_clamp_in = get_socket(clamp_node.inputs, "Transmission Weight")
    if trans_bsdf_in is not None and trans_clamp_in is not None:
        if trans_bsdf_in.is_linked:
            node_tree.links.new(trans_bsdf_in.links[0].from_socket, trans_clamp_in)
        else:
            trans_clamp_in.default_value = float(trans_bsdf_in.default_value)

    ml_sock = get_socket(clamp_node.inputs, "Metallic Linked")
    if ml_sock is not None:
        ml_sock.default_value = 1.0 if metallic_was_linked else 0.0

    _update_clamp_parameters(
        clamp_node, clamp_group, dielectric_knee, dielectric_max, get_socket
    )
    return clamp_node


def clamp_material_pbr(
    mat: Optional[bpy.types.Material],
    dielectric_knee: float = constants.CLAMP_DIELECTRIC_KNEE,
    dielectric_max: float = constants.CLAMP_DIELECTRIC_MAX,
) -> bool:
    """Insert or update non-destructive soft-knee PBR clamping on ``mat``."""
    if mat is None or getattr(mat, "node_tree", None) is None:
        return False

    from ..core import graph_crawler, pbr_shader_nodes, shader_splicer

    nt = mat.node_tree
    get_socket = pbr_shader_nodes.get_socket

    existing_clamps = _find_existing_clamp_nodes(nt)
    if existing_clamps:
        clamp_group = pbr_shader_nodes.ensure_clamp_group()
        for clamp_node in existing_clamps:
            _update_clamp_parameters(
                clamp_node, clamp_group, dielectric_knee, dielectric_max, get_socket
            )
        if shader_splicer.is_material_spliced(mat):
            shader_splicer.resync_material(mat)
        return True

    bsdfs = graph_crawler.find_principled_bsdfs(nt)
    if not bsdfs:
        return False

    inserted = False
    clamp_group: Optional[bpy.types.NodeTree] = None
    for bsdf in bsdfs:
        if not _bsdf_needs_clamping(bsdf, get_socket):
            continue
        if clamp_group is None:
            clamp_group = pbr_shader_nodes.ensure_clamp_group()
        _insert_clamp_for_bsdf(
            nt, bsdf, clamp_group, dielectric_knee, dielectric_max, get_socket
        )
        inserted = True

    if inserted and shader_splicer.is_material_spliced(mat):
        shader_splicer.resync_material(mat)

    return inserted


def clamp_scene_pbr(
    scene: Optional[bpy.types.Scene],
    selected_only: bool = False,
) -> Dict[str, int]:
    """Clamp out-of-spec PBR materials across mesh objects in ``scene``."""
    if scene is None:
        return {"materials_clamped": 0}

    fc = getattr(scene, "flightcheck", None)
    dielectric_knee = (
        float(fc.clamp_dielectric_knee)
        if fc is not None
        else constants.CLAMP_DIELECTRIC_KNEE
    )
    dielectric_max = (
        float(fc.clamp_dielectric_max)
        if fc is not None
        else constants.CLAMP_DIELECTRIC_MAX
    )

    clamped_count = 0
    for mat in collect_scene_materials(scene, selected_only=selected_only):
        if clamp_material_pbr(
            mat, dielectric_knee=dielectric_knee, dielectric_max=dielectric_max
        ):
            clamped_count += 1

    return {"materials_clamped": clamped_count}


class FLIGHTCHECK_OT_clamp_pbr(bpy.types.Operator):
    """Apply non-destructive soft-knee PBR clamping to scene materials."""

    bl_idname = "flightcheck.clamp_pbr"
    bl_label = "Clamp PBR Values"
    bl_description = (
        "Insert non-destructive __FLIGHTCHECK_CLAMP_NODE__ groups on out-of-spec materials"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        from ..core import run_full_scan
        from ..hud import tag_view3d_redraw

        scene = context.scene
        fc = getattr(scene, "flightcheck", None)
        selected_only = bool(fc.selected_only) if fc is not None else False

        result = clamp_scene_pbr(scene, selected_only=selected_only)
        run_full_scan(scene, write_texel_attr=True)
        tag_view3d_redraw()

        self.report(
            {"INFO"},
            f"FlightCheck: Clamped {result['materials_clamped']} material(s).",
        )
        return {"FINISHED"}
