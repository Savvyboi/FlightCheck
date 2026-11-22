"""Non-destructive soft-knee PBR clamping fixer for Principled BSDF materials."""

from __future__ import annotations

import inspect
from typing import Dict, List, Optional, Set, Tuple

import bpy
import numpy as np

from .. import constants
from ..state import internal_mutation
from . import collect_scene_materials, sync_wm_issues

_CLAMP_SOCKET_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("Base Color", "Color"),
    ("Metallic", "Metallic"),
    ("Roughness", "Roughness"),
    ("Specular IOR Level", "Specular IOR Level"),
    ("IOR", "IOR"),
)


def _is_clamp_node(node: Optional[bpy.types.Node]) -> bool:
    if node is None:
        return False
    if node.name == constants.CLAMP_NODE_NAME:
        return True
    nt = getattr(node, "node_tree", None)
    return (
        node.type == "GROUP"
        and nt is not None
        and nt.name == constants.CLAMP_GROUP_NAME
    )


def _find_existing_clamp_nodes(node_tree: bpy.types.NodeTree) -> List[bpy.types.Node]:
    """Return existing FlightCheck PBR clamp group nodes in ``node_tree``."""
    return [node for node in node_tree.nodes if _is_clamp_node(node)]


def _find_bsdf_clamp_nodes(bsdf: bpy.types.Node, get_socket) -> List[bpy.types.Node]:
    """Return clamp nodes directly feeding any clamped input socket of ``bsdf``."""
    found: List[bpy.types.Node] = []
    seen_ptrs: Set[int] = set()
    for bsdf_name, _ in _CLAMP_SOCKET_PAIRS:
        sock = get_socket(bsdf.inputs, bsdf_name)
        if sock is None or not sock.is_linked:
            continue
        for link in sock.links:
            fn = getattr(link, "from_node", None)
            if _is_clamp_node(fn):
                ptr = fn.as_pointer()
                if ptr not in seen_ptrs:
                    seen_ptrs.add(ptr)
                    found.append(fn)
    return found


def _enforce_clamp_bounds(
    dielectric_knee: float,
    dielectric_max: float,
) -> Tuple[float, float]:
    """Clamp knee <= 0.79 and enforce dielectric_max >= dielectric_knee + 0.01."""
    knee = min(float(dielectric_knee), 0.79)
    ceiling = max(float(dielectric_max), knee + 0.01)
    return knee, ceiling


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


def _eval_bounds(eval_res, is_color: bool = False) -> Tuple[float, float]:
    """Return robust (low, high) bounds favoring p01/p99, or computing from samples."""
    if hasattr(eval_res, "p01") and hasattr(eval_res, "p99"):
        return float(eval_res.p01), float(eval_res.p99)
    samples = getattr(eval_res, "samples", None)
    if samples is not None and len(samples) > 0:
        if is_color and samples.ndim == 2 and samples.shape[1] >= 3:
            vals = (
                constants.LUMA_WEIGHTS[0] * samples[:, 0]
                + constants.LUMA_WEIGHTS[1] * samples[:, 1]
                + constants.LUMA_WEIGHTS[2] * samples[:, 2]
            )
        else:
            vals = samples.reshape(-1)
        p01, p99 = np.percentile(vals, [1.0, 99.0])
        return float(p01), float(p99)
    return float(eval_res.min_val), float(eval_res.max_val)


def _eval_scalar_avg(sock, graph_crawler, pixel_cache: Dict[int, np.ndarray], default: float) -> Tuple[Optional[object], float]:
    if sock is None:
        return None, default
    ev = graph_crawler.evaluate_scalar_socket(sock, pixel_cache)
    if ev is None:
        return None, default
    return ev, float(ev.avg_val)


def _fallback_bsdf_out_of_spec_sockets(
    bsdf: bpy.types.Node,
    graph_crawler,
    get_socket,
    pixel_cache: Dict[int, np.ndarray],
) -> Set[str]:
    """Evaluate Principled BSDF inputs with p01/p99 and per-sample dielectric/conductor checks."""
    out_of_spec: Set[str] = set()

    base_sock = get_socket(bsdf.inputs, "Base Color")
    met_sock = get_socket(bsdf.inputs, "Metallic")
    rough_sock = get_socket(bsdf.inputs, "Roughness")
    spec_sock = get_socket(bsdf.inputs, "Specular IOR Level")
    ior_sock = get_socket(bsdf.inputs, "IOR")
    trans_sock = get_socket(bsdf.inputs, "Transmission Weight")

    base_eval = graph_crawler.evaluate_color_socket(base_sock, pixel_cache) if base_sock is not None else None
    met_eval, met_val = _eval_scalar_avg(met_sock, graph_crawler, pixel_cache, 0.0)
    rough_eval = graph_crawler.evaluate_scalar_socket(rough_sock, pixel_cache) if rough_sock is not None else None
    spec_eval = graph_crawler.evaluate_scalar_socket(spec_sock, pixel_cache) if spec_sock is not None else None
    ior_eval = graph_crawler.evaluate_scalar_socket(ior_sock, pixel_cache) if ior_sock is not None else None
    _trans_eval, trans_val = _eval_scalar_avg(trans_sock, graph_crawler, pixel_cache, 0.0)

    if met_eval is not None:
        m_low, m_high = _eval_bounds(met_eval)
        if m_low < 0.0 or m_high > 1.0:
            out_of_spec.add("Metallic")
        elif (not met_eval.is_linked) and (
            constants.SEMI_METAL_LOW < met_eval.avg_val < constants.SEMI_METAL_HIGH
        ):
            out_of_spec.add("Metallic")

    if base_eval is not None:
        base_samples = getattr(base_eval, "samples", None)
        met_samples = getattr(met_eval, "samples", None) if met_eval is not None else None
        if base_samples is not None and met_samples is not None:
            n_b = len(base_samples)
            n_m = len(met_samples)
            if n_b > 0 and n_m > 0:
                if n_b != n_m:
                    idx = np.linspace(0, n_m - 1, n_b, dtype=np.int64)
                    met_aligned = met_samples[idx]
                else:
                    met_aligned = met_samples
                lumas = (
                    constants.LUMA_WEIGHTS[0] * base_samples[:, 0]
                    + constants.LUMA_WEIGHTS[1] * base_samples[:, 1]
                    + constants.LUMA_WEIGHTS[2] * base_samples[:, 2]
                )
                diel_mask = met_aligned < constants.METALLIC_CONDUCTOR_THRESHOLD
                cond_mask = ~diel_mask
                if np.any(diel_mask):
                    d_p01, d_p99 = np.percentile(lumas[diel_mask], [1.0, 99.0])
                    if d_p01 < constants.DIELECTRIC_MIN_LUMA or d_p99 > constants.DIELECTRIC_WARN_LUMA:
                        out_of_spec.add("Base Color")
                if np.any(cond_mask):
                    c_p01 = float(np.percentile(lumas[cond_mask], 1.0))
                    if c_p01 < constants.CONDUCTOR_MIN_LUMA:
                        out_of_spec.add("Base Color")
        else:
            b_low, b_high = _eval_bounds(base_eval, is_color=True)
            if met_val < constants.METALLIC_CONDUCTOR_THRESHOLD:
                if b_low < constants.DIELECTRIC_MIN_LUMA or b_high > constants.DIELECTRIC_WARN_LUMA:
                    out_of_spec.add("Base Color")
            else:
                if b_low < constants.CONDUCTOR_MIN_LUMA:
                    out_of_spec.add("Base Color")

    if rough_eval is not None and trans_val < constants.TRANSMISSION_THRESHOLD:
        r_low, r_high = _eval_bounds(rough_eval)
        if r_low < constants.ROUGHNESS_MIN or r_high > constants.ROUGHNESS_MAX:
            out_of_spec.add("Roughness")

    if spec_eval is not None:
        s_low, s_high = _eval_bounds(spec_eval)
        if s_low < constants.SPECULAR_MIN or s_high > constants.SPECULAR_MAX:
            out_of_spec.add("Specular IOR Level")

    if (
        ior_eval is not None
        and met_val < constants.METALLIC_CONDUCTOR_THRESHOLD
        and trans_val < constants.TRANSMISSION_THRESHOLD
    ):
        i_low, i_high = _eval_bounds(ior_eval)
        if i_low < constants.IOR_MIN or i_high > constants.IOR_MAX:
            out_of_spec.add("IOR")

    return out_of_spec


def _get_bsdf_out_of_spec_sockets(
    mat: bpy.types.Material,
    bsdf: bpy.types.Node,
    graph_crawler,
    get_socket,
    pixel_cache: Dict[int, np.ndarray],
) -> Set[str]:
    """Return out-of-spec PBR socket names for ``bsdf``, gating IOR on dielectric non-transmissive."""
    fn = getattr(graph_crawler, "bsdf_out_of_spec_sockets", None)
    if callable(fn):
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        kwargs = {"pixel_cache": pixel_cache} if "pixel_cache" in params else {}
        if params and params[0] in ("mat", "material"):
            out = set(fn(mat, bsdf, **kwargs))
        else:
            out = set(fn(bsdf, **kwargs))
    else:
        out = _fallback_bsdf_out_of_spec_sockets(bsdf, graph_crawler, get_socket, pixel_cache)

    # Enforce metallic / transmissive IOR and transmissive roughness guards.
    if "IOR" in out or "Roughness" in out:
        met_sock = get_socket(bsdf.inputs, "Metallic")
        trans_sock = get_socket(bsdf.inputs, "Transmission Weight")
        _me, met_val = _eval_scalar_avg(met_sock, graph_crawler, pixel_cache, 0.0)
        _te, trans_val = _eval_scalar_avg(trans_sock, graph_crawler, pixel_cache, 0.0)
        if met_val >= constants.METALLIC_CONDUCTOR_THRESHOLD or trans_val >= constants.TRANSMISSION_THRESHOLD:
            out.discard("IOR")
        if trans_val >= constants.TRANSMISSION_THRESHOLD:
            out.discard("Roughness")

    return out


def _sync_readonly_input(
    node_tree: bpy.types.NodeTree,
    src_bsdf_sock: Optional[bpy.types.NodeSocket],
    clamp_in_sock: Optional[bpy.types.NodeSocket],
) -> None:
    """Mirror a BSDF socket into a clamp node input without altering the BSDF socket."""
    if src_bsdf_sock is None or clamp_in_sock is None:
        return
    if src_bsdf_sock.is_linked:
        src_from = src_bsdf_sock.links[0].from_socket
        if src_from.node == clamp_in_sock.node:
            return
        already = (
            clamp_in_sock.is_linked
            and any(link.from_socket == src_from for link in clamp_in_sock.links)
        )
        if not already:
            for link in list(clamp_in_sock.links):
                node_tree.links.remove(link)
            node_tree.links.new(src_from, clamp_in_sock)
    else:
        for link in list(clamp_in_sock.links):
            node_tree.links.remove(link)
        clamp_in_sock.default_value = float(src_bsdf_sock.default_value)


def _route_socket_through_clamp(
    node_tree: bpy.types.NodeTree,
    bsdf_in: bpy.types.NodeSocket,
    clamp_in: bpy.types.NodeSocket,
    clamp_out: bpy.types.NodeSocket,
    clamp_name: str,
) -> bool:
    """Splice ``clamp_in -> clamp_out`` onto ``bsdf_in``; return True if ``bsdf_in`` was linked."""
    was_linked = False
    if bsdf_in.is_linked and any(link.from_node == clamp_out.node for link in bsdf_in.links):
        return bool(clamp_in.is_linked)

    if bsdf_in.is_linked:
        was_linked = True
        upstream_sockets = [link.from_socket for link in bsdf_in.links]
        for link in list(bsdf_in.links):
            node_tree.links.remove(link)
        for link in list(clamp_in.links):
            node_tree.links.remove(link)
        for from_sock in upstream_sockets:
            node_tree.links.new(from_sock, clamp_in)
    else:
        for link in list(clamp_in.links):
            node_tree.links.remove(link)
        if clamp_name == "Color":
            vals = list(bsdf_in.default_value)
            clamp_in.default_value = (
                float(vals[0]) if len(vals) > 0 else 0.0,
                float(vals[1]) if len(vals) > 1 else 0.0,
                float(vals[2]) if len(vals) > 2 else 0.0,
                float(vals[3]) if len(vals) > 3 else 1.0,
            )
        else:
            clamp_in.default_value = float(bsdf_in.default_value)

    node_tree.links.new(clamp_out, bsdf_in)
    return was_linked


def _sync_clamp_node_metadata(
    node_tree: bpy.types.NodeTree,
    bsdf: bpy.types.Node,
    clamp_node: bpy.types.Node,
    metallic_was_linked: Optional[bool],
    get_socket,
) -> None:
    """Keep read-only Metallic/Transmission pass-through and Metallic Linked flag synced."""
    bsdf_met = get_socket(bsdf.inputs, "Metallic")
    clamp_met_in = get_socket(clamp_node.inputs, "Metallic")
    if bsdf_met is not None and clamp_met_in is not None:
        if not (bsdf_met.is_linked and any(l.from_node == clamp_node for l in bsdf_met.links)):
            _sync_readonly_input(node_tree, bsdf_met, clamp_met_in)
            if metallic_was_linked is None:
                metallic_was_linked = bool(bsdf_met.is_linked)
        elif metallic_was_linked is None:
            metallic_was_linked = bool(clamp_met_in.is_linked)

    trans_bsdf_in = get_socket(bsdf.inputs, "Transmission Weight")
    trans_clamp_in = get_socket(clamp_node.inputs, "Transmission Weight")
    _sync_readonly_input(node_tree, trans_bsdf_in, trans_clamp_in)

    ml_sock = get_socket(clamp_node.inputs, "Metallic Linked")
    if ml_sock is not None and metallic_was_linked is not None:
        ml_sock.default_value = 1.0 if metallic_was_linked else 0.0


def _insert_clamp_for_bsdf(
    node_tree: bpy.types.NodeTree,
    bsdf: bpy.types.Node,
    clamp_group: bpy.types.NodeTree,
    out_of_spec: Set[str],
    dielectric_knee: float,
    dielectric_max: float,
    get_socket,
) -> bpy.types.Node:
    """Splice a __FLIGHTCHECK_CLAMP_NODE__ immediately upstream of ``bsdf`` for out-of-spec sockets."""
    clamp_node = node_tree.nodes.new("ShaderNodeGroup")
    clamp_node.node_tree = clamp_group
    clamp_node.name = constants.CLAMP_NODE_NAME
    clamp_node.label = "FlightCheck PBR Clamp"
    clamp_node.location = (bsdf.location.x - 240.0, bsdf.location.y)

    metallic_was_linked = False
    for bsdf_name, clamp_name in _CLAMP_SOCKET_PAIRS:
        if bsdf_name not in out_of_spec:
            continue
        bsdf_in = get_socket(bsdf.inputs, bsdf_name)
        clamp_in = get_socket(clamp_node.inputs, clamp_name)
        clamp_out = get_socket(clamp_node.outputs, clamp_name)
        if bsdf_in is None or clamp_in is None or clamp_out is None:
            continue
        linked = _route_socket_through_clamp(node_tree, bsdf_in, clamp_in, clamp_out, clamp_name)
        if bsdf_name == "Metallic":
            metallic_was_linked = linked

    if "Metallic" not in out_of_spec:
        bsdf_met = get_socket(bsdf.inputs, "Metallic")
        metallic_was_linked = bool(bsdf_met is not None and bsdf_met.is_linked)

    _sync_clamp_node_metadata(node_tree, bsdf, clamp_node, metallic_was_linked, get_socket)
    _update_clamp_parameters(
        clamp_node, clamp_group, dielectric_knee, dielectric_max, get_socket
    )
    return clamp_node


def clamp_material_pbr(
    mat: Optional[bpy.types.Material],
    dielectric_knee: float = constants.CLAMP_DIELECTRIC_KNEE,
    dielectric_max: float = constants.CLAMP_DIELECTRIC_MAX,
) -> bool:
    """Insert or update non-destructive soft-knee PBR clamping on ``mat`` per Principled BSDF."""
    if not constants.is_editable_id(mat) or not constants.is_editable_id(getattr(mat, "node_tree", None)):
        return False

    from ..core import graph_crawler, pbr_shader_nodes, shader_splicer

    nt = mat.node_tree
    get_socket = pbr_shader_nodes.get_socket
    knee_f, max_f = _enforce_clamp_bounds(dielectric_knee, dielectric_max)
    pixel_cache: Dict[int, np.ndarray] = {}

    bsdfs = graph_crawler.find_principled_bsdfs(nt)
    changed = False
    updated_clamp_ptrs: Set[int] = set()
    seen_bsdf_ptrs: Set[int] = set()
    clamp_group: Optional[bpy.types.NodeTree] = None

    for bsdf in bsdfs:
        bsdf_ptr = bsdf.as_pointer()
        if bsdf_ptr in seen_bsdf_ptrs:
            continue
        seen_bsdf_ptrs.add(bsdf_ptr)

        bsdf_nt = getattr(bsdf, "id_data", None) or nt
        if not constants.is_editable_id(bsdf_nt):
            continue

        bsdf_clamps = _find_bsdf_clamp_nodes(bsdf, get_socket)
        out_of_spec = _get_bsdf_out_of_spec_sockets(mat, bsdf, graph_crawler, get_socket, pixel_cache)

        if bsdf_clamps:
            if clamp_group is None:
                clamp_group = pbr_shader_nodes.ensure_clamp_group()
            primary_clamp = bsdf_clamps[0]
            for clamp_node in bsdf_clamps:
                _update_clamp_parameters(clamp_node, clamp_group, knee_f, max_f, get_socket)
                updated_clamp_ptrs.add(clamp_node.as_pointer())
            if out_of_spec:
                met_linked: Optional[bool] = None
                for bsdf_name, clamp_name in _CLAMP_SOCKET_PAIRS:
                    if bsdf_name not in out_of_spec:
                        continue
                    bsdf_in = get_socket(bsdf.inputs, bsdf_name)
                    clamp_in = get_socket(primary_clamp.inputs, clamp_name)
                    clamp_out = get_socket(primary_clamp.outputs, clamp_name)
                    if bsdf_in is None or clamp_in is None or clamp_out is None:
                        continue
                    was_linked = _route_socket_through_clamp(
                        bsdf_nt, bsdf_in, clamp_in, clamp_out, clamp_name
                    )
                    if bsdf_name == "Metallic":
                        met_linked = was_linked
                _sync_clamp_node_metadata(bsdf_nt, bsdf, primary_clamp, met_linked, get_socket)
            else:
                _sync_clamp_node_metadata(bsdf_nt, bsdf, primary_clamp, None, get_socket)
            changed = True
            continue

        if not out_of_spec:
            continue

        if clamp_group is None:
            clamp_group = pbr_shader_nodes.ensure_clamp_group()
        clamp_node = _insert_clamp_for_bsdf(
            bsdf_nt, bsdf, clamp_group, out_of_spec, knee_f, max_f, get_socket
        )
        updated_clamp_ptrs.add(clamp_node.as_pointer())
        changed = True

    for orphan_clamp in _find_existing_clamp_nodes(nt):
        if orphan_clamp.as_pointer() not in updated_clamp_ptrs:
            if clamp_group is None:
                clamp_group = pbr_shader_nodes.ensure_clamp_group()
            _update_clamp_parameters(orphan_clamp, clamp_group, knee_f, max_f, get_socket)
            changed = True

    if changed and shader_splicer.is_material_spliced(mat):
        shader_splicer.resync_material(mat)

    return changed


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
    dielectric_knee, dielectric_max = _enforce_clamp_bounds(dielectric_knee, dielectric_max)

    clamped_count = 0
    for mat in collect_scene_materials(scene, selected_only=selected_only):
        if not constants.is_editable_id(mat) or not constants.is_editable_id(getattr(mat, "node_tree", None)):
            continue
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

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return context.scene is not None

    def execute(self, context: bpy.types.Context) -> set[str]:
        from ..core import run_full_scan
        from ..hud import tag_view3d_redraw

        scene = context.scene
        fc = getattr(scene, "flightcheck", None)
        selected_only = bool(fc.selected_only) if fc is not None else False

        with internal_mutation():
            result = clamp_scene_pbr(scene, selected_only=selected_only)
            report = run_full_scan(scene, write_texel_attr=True)
            sync_wm_issues(context, report)
            tag_view3d_redraw()

        self.report(
            {"INFO"},
            f"FlightCheck: Clamped {result['materials_clamped']} material(s).",
        )
        return {"FINISHED"}
