"""Shader node-graph traversal, evaluation, colorspace mapping, and PBR analysis."""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

import bpy
import numpy as np

from .. import constants
from ..state import Category, DiagnosticIssue, Severity

_DATA_NODE_TAGS: Dict[str, str] = {
    "NORMAL_MAP": "Normal",
    "BUMP": "Bump",
    "DISPLACEMENT": "Displacement",
    "VECTOR_DISPLACEMENT": "Displacement",
}


@dataclass
class SocketEval:
    """Evaluated value range and linkage metadata for a shader input socket."""

    min_val: float
    max_val: float
    avg_val: float
    is_linked: bool
    is_textured: bool = False
    avg_rgb: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    samples: Optional[np.ndarray] = None


def get_socket(
    sockets: Any,
    name: str,
    socket_type: Optional[str] = None,
    occurrence: int = 0,
) -> Optional[bpy.types.NodeSocket]:
    """Return the ``occurrence``-th enabled socket matching ``name`` (and optional ``socket_type``)."""
    if sockets is None:
        return None
    count = 0
    for sock in sockets:
        if sock.name != name or not sock.enabled:
            continue
        if socket_type is not None and sock.type != socket_type:
            continue
        if count == occurrence:
            return sock
        count += 1
    return None


def _first_enabled_socket(sockets: Any, preferred_name: Optional[str] = None) -> Optional[bpy.types.NodeSocket]:
    """Return ``preferred_name`` if enabled, else the first enabled socket in ``sockets``."""
    if preferred_name is not None:
        sock = get_socket(sockets, preferred_name)
        if sock is not None:
            return sock
    if sockets is None:
        return None
    for sock in sockets:
        if sock.enabled and sock.type != "CUSTOM":
            return sock
    return None


def _is_overlay_node(node: Optional[bpy.types.Node]) -> bool:
    if node is None:
        return False
    if node.name == constants.OVERLAY_NODE_NAME:
        return True
    nt = getattr(node, "node_tree", None)
    return node.type == "GROUP" and nt is not None and nt.name == constants.OVERLAY_GROUP_NAME


def _is_clamp_node(node: Optional[bpy.types.Node]) -> bool:
    if node is None:
        return False
    if node.name == constants.CLAMP_NODE_NAME:
        return True
    nt = getattr(node, "node_tree", None)
    return node.type == "GROUP" and nt is not None and nt.name == constants.CLAMP_GROUP_NAME


def _valid_links(socket: Optional[bpy.types.NodeSocket]) -> List[bpy.types.NodeLink]:
    if socket is None or not socket.is_linked:
        return []
    result: List[bpy.types.NodeLink] = []
    for link in socket.links:
        if not getattr(link, "is_valid", True):
            continue
        if getattr(link, "is_muted", False):
            continue
        if link.from_node is None or link.from_socket is None:
            continue
        result.append(link)
    return result


def _match_group_node_input(
    group_node: bpy.types.Node,
    group_input_out_socket: bpy.types.NodeSocket,
) -> Optional[bpy.types.NodeSocket]:
    """Map an output socket on an internal ``NodeGroupInput`` back to ``group_node.inputs``."""
    ident = getattr(group_input_out_socket, "identifier", "")
    for sock in group_node.inputs:
        if sock.enabled and ident and getattr(sock, "identifier", "") == ident:
            return sock
    return get_socket(group_node.inputs, group_input_out_socket.name)


def _match_group_output_inputs(
    group_tree: bpy.types.NodeTree,
    group_output_socket: bpy.types.NodeSocket,
) -> List[bpy.types.NodeSocket]:
    """Map an output socket on ``ShaderNodeGroup`` to internal ``NodeGroupOutput`` input sockets."""
    out_nodes = [n for n in group_tree.nodes if n.type == "GROUP_OUTPUT"]
    if not out_nodes:
        return []
    active_outs = [n for n in out_nodes if getattr(n, "is_active_output", False)]
    targets = active_outs if active_outs else out_nodes

    ident = getattr(group_output_socket, "identifier", "")
    matched: List[bpy.types.NodeSocket] = []
    for go in targets:
        found = None
        if ident:
            for sock in go.inputs:
                if sock.enabled and sock.type != "CUSTOM" and getattr(sock, "identifier", "") == ident:
                    found = sock
                    break
        if found is None:
            found = get_socket(go.inputs, group_output_socket.name)
        if found is not None:
            matched.append(found)
    return matched


def find_active_material_output(node_tree: Optional[bpy.types.NodeTree]) -> Optional[bpy.types.Node]:
    """Return the active ``OUTPUT_MATERIAL`` node, falling back to the first one in ``node_tree``."""
    if node_tree is None:
        return None
    outputs = [n for n in node_tree.nodes if n.type == "OUTPUT_MATERIAL"]
    if not outputs:
        return None
    for node in outputs:
        if getattr(node, "is_active_output", False):
            return node
    return outputs[0]


def find_principled_bsdfs(node_tree: Optional[bpy.types.NodeTree]) -> List[bpy.types.Node]:
    """Return ``BSDF_PRINCIPLED`` nodes reachable upstream from active output, or all in tree."""
    if node_tree is None:
        return []

    active_output = find_active_material_output(node_tree)
    reachable: List[bpy.types.Node] = []

    if active_output is not None:
        visited: Set[int] = set()
        stack: List[bpy.types.Node] = [active_output]
        while stack:
            node = stack.pop()
            ptr = node.as_pointer()
            if ptr in visited:
                continue
            visited.add(ptr)

            if node.type == "BSDF_PRINCIPLED":
                reachable.append(node)

            if _is_overlay_node(node):
                shader_sock = get_socket(node.inputs, "Shader")
                for link in _valid_links(shader_sock):
                    stack.append(link.from_node)
                continue

            for sock in node.inputs:
                if not sock.enabled:
                    continue
                for link in _valid_links(sock):
                    stack.append(link.from_node)

    if reachable:
        return reachable
    return [n for n in node_tree.nodes if n.type == "BSDF_PRINCIPLED"]


def _trace_images_recursive(
    socket: Optional[bpy.types.NodeSocket],
    visited: Set[Tuple[int, Tuple[int, ...]]],
    group_stack: Tuple[bpy.types.Node, ...],
    seen_pairs: Set[Tuple[int, int]],
    results: List[Tuple[bpy.types.Node, bpy.types.Image]],
) -> None:
    if socket is None:
        return

    key = (socket.as_pointer(), tuple(g.as_pointer() for g in group_stack))
    if key in visited:
        return
    visited.add(key)

    for link in _valid_links(socket):
        from_node = link.from_node
        from_socket = link.from_socket

        if from_node.type == "TEX_IMAGE":
            img = getattr(from_node, "image", None)
            if img is not None:
                pair_key = (from_node.as_pointer(), img.as_pointer())
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    results.append((from_node, img))
            continue

        if from_node.type == "REROUTE":
            in_sock = _first_enabled_socket(from_node.inputs, "Input")
            _trace_images_recursive(in_sock, visited, group_stack, seen_pairs, results)
            continue

        if from_node.type == "GROUP_INPUT":
            if group_stack:
                parent_node = group_stack[-1]
                parent_sock = _match_group_node_input(parent_node, from_socket)
                _trace_images_recursive(parent_sock, visited, group_stack[:-1], seen_pairs, results)
            continue

        if _is_overlay_node(from_node):
            shader_sock = get_socket(from_node.inputs, "Shader")
            _trace_images_recursive(shader_sock, visited, group_stack, seen_pairs, results)
            continue

        if from_node.type == "GROUP":
            sub_tree = getattr(from_node, "node_tree", None)
            if sub_tree is not None:
                for go_sock in _match_group_output_inputs(sub_tree, from_socket):
                    _trace_images_recursive(
                        go_sock,
                        visited,
                        group_stack + (from_node,),
                        seen_pairs,
                        results,
                    )
            continue

        for in_sock in from_node.inputs:
            if in_sock.enabled and in_sock.is_linked:
                _trace_images_recursive(in_sock, visited, group_stack, seen_pairs, results)


def trace_upstream_images(
    socket: Optional[bpy.types.NodeSocket],
    visited: Optional[Set[Any]] = None,
) -> List[Tuple[bpy.types.Node, bpy.types.Image]]:
    """Recursively walk upstream links (including reroutes and node groups) for ``TEX_IMAGE`` nodes."""
    if visited is None:
        visited = set()
    results: List[Tuple[bpy.types.Node, bpy.types.Image]] = []
    seen_pairs: Set[Tuple[int, int]] = set()
    _trace_images_recursive(socket, visited, (), seen_pairs, results)
    return results


def _collect_socket_images_with_tags(
    socket: Optional[bpy.types.NodeSocket],
    data_tags: FrozenSet[str],
    visited: Set[Tuple[int, Tuple[int, ...], FrozenSet[str]]],
    group_stack: Tuple[bpy.types.Node, ...],
    out_matches: List[Tuple[bpy.types.Node, bpy.types.Image, FrozenSet[str]]],
) -> None:
    if socket is None:
        return

    key = (socket.as_pointer(), tuple(g.as_pointer() for g in group_stack), data_tags)
    if key in visited:
        return
    visited.add(key)

    for link in _valid_links(socket):
        from_node = link.from_node
        from_socket = link.from_socket

        if from_node.type == "TEX_IMAGE":
            img = getattr(from_node, "image", None)
            if img is not None:
                out_matches.append((from_node, img, data_tags))
            continue

        if from_node.type == "REROUTE":
            in_sock = _first_enabled_socket(from_node.inputs, "Input")
            _collect_socket_images_with_tags(in_sock, data_tags, visited, group_stack, out_matches)
            continue

        if from_node.type == "GROUP_INPUT":
            if group_stack:
                parent_node = group_stack[-1]
                parent_sock = _match_group_node_input(parent_node, from_socket)
                _collect_socket_images_with_tags(
                    parent_sock, data_tags, visited, group_stack[:-1], out_matches
                )
            continue

        if _is_overlay_node(from_node):
            shader_sock = get_socket(from_node.inputs, "Shader")
            _collect_socket_images_with_tags(shader_sock, data_tags, visited, group_stack, out_matches)
            continue

        next_tags = data_tags
        if from_node.type in constants.DATA_ONLY_NODE_TYPES:
            tag = _DATA_NODE_TAGS.get(from_node.type)
            if tag:
                next_tags = data_tags | {tag}

        if from_node.type == "GROUP":
            sub_tree = getattr(from_node, "node_tree", None)
            if sub_tree is not None:
                for go_sock in _match_group_output_inputs(sub_tree, from_socket):
                    _collect_socket_images_with_tags(
                        go_sock,
                        next_tags,
                        visited,
                        group_stack + (from_node,),
                        out_matches,
                    )
            continue

        for in_sock in from_node.inputs:
            if in_sock.enabled and in_sock.is_linked:
                _collect_socket_images_with_tags(in_sock, next_tags, visited, group_stack, out_matches)


def collect_colorspace_map(
    materials: Iterable[Optional[bpy.types.Material]],
) -> Dict[bpy.types.Image, Dict[str, Set[str]]]:
    """Map each referenced ``bpy.types.Image`` to its data/color socket usage across ``materials``."""
    cs_map: Dict[bpy.types.Image, Dict[str, Set[str]]] = {}

    def _ensure_entry(image: bpy.types.Image) -> Dict[str, Set[str]]:
        entry = cs_map.get(image)
        if entry is None:
            entry = {
                "data_sockets": set(),
                "color_sockets": set(),
                "materials": set(),
                "nodes": set(),
            }
            cs_map[image] = entry
        return entry

    def _record_socket(mat_name: str, sock: bpy.types.NodeSocket) -> None:
        if not sock.enabled or not sock.is_linked:
            return
        matches: List[Tuple[bpy.types.Node, bpy.types.Image, FrozenSet[str]]] = []
        _collect_socket_images_with_tags(sock, frozenset(), set(), (), matches)
        for tex_node, img, path_tags in matches:
            entry = _ensure_entry(img)
            entry["materials"].add(mat_name)
            entry["nodes"].add(tex_node.name)
            entry["data_sockets"].update(path_tags)
            if sock.name in constants.DATA_MAP_SOCKETS:
                entry["data_sockets"].add(sock.name)
            elif sock.name in constants.COLOR_MAP_SOCKETS and not path_tags:
                entry["color_sockets"].add(sock.name)

    for mat in materials:
        if mat is None or mat.node_tree is None:
            continue
        for bsdf in find_principled_bsdfs(mat.node_tree):
            for sock in bsdf.inputs:
                _record_socket(mat.name, sock)

        active_out = find_active_material_output(mat.node_tree)
        if active_out is not None:
            disp_sock = get_socket(active_out.inputs, "Displacement")
            if disp_sock is not None:
                _record_socket(mat.name, disp_sock)

    return cs_map


def _sample_image_rgba(
    image: Optional[bpy.types.Image],
    pixel_cache: Optional[Dict[int, np.ndarray]] = None,
) -> Optional[np.ndarray]:
    """Sample up to 4096 RGBA pixels evenly from ``image`` (linearizing RGB when ``sRGB``)."""
    if image is None:
        return None
    cache_key = image.as_pointer()
    if pixel_cache is not None and cache_key in pixel_cache:
        return pixel_cache[cache_key]

    try:
        n_floats = len(image.pixels)
    except Exception:
        return None
    if n_floats < 4 or (n_floats % 4) != 0:
        return None

    total_pixels = n_floats // 4
    buf = np.empty(n_floats, dtype=np.float32)
    image.pixels.foreach_get(buf)
    pix = buf.reshape(total_pixels, 4)

    if total_pixels > 4096:
        indices = np.linspace(0, total_pixels - 1, 4096, dtype=np.int64)
        sampled = pix[indices].astype(np.float64, copy=True)
    else:
        sampled = pix.astype(np.float64, copy=True)

    cs_name = getattr(getattr(image, "colorspace_settings", None), "name", "")
    if cs_name == "sRGB":
        rgb = sampled[:, :3]
        sampled[:, :3] = np.where(
            rgb <= 0.04045,
            np.maximum(0.0, rgb) / 12.92,
            ((rgb + 0.055) / 1.055) ** 2.4,
        )

    if pixel_cache is not None:
        pixel_cache[cache_key] = sampled
    return sampled


def _eval_color_from_samples(samples: np.ndarray, is_linked: bool) -> SocketEval:
    rgb = samples[:, :3]
    lumas = (
        constants.LUMA_WEIGHTS[0] * rgb[:, 0]
        + constants.LUMA_WEIGHTS[1] * rgb[:, 1]
        + constants.LUMA_WEIGHTS[2] * rgb[:, 2]
    )
    avg_rgb = (float(np.mean(rgb[:, 0])), float(np.mean(rgb[:, 1])), float(np.mean(rgb[:, 2])))
    return SocketEval(
        min_val=float(np.min(lumas)),
        max_val=float(np.max(lumas)),
        avg_val=float(np.mean(lumas)),
        is_linked=is_linked,
        is_textured=True,
        avg_rgb=avg_rgb,
        samples=rgb,
    )


def _eval_scalar_from_samples(values: np.ndarray, is_linked: bool) -> SocketEval:
    flat = values.reshape(-1)
    return SocketEval(
        min_val=float(np.min(flat)),
        max_val=float(np.max(flat)),
        avg_val=float(np.mean(flat)),
        is_linked=is_linked,
        is_textured=True,
        samples=flat,
    )


def _eval_color_constant(rgb: Tuple[float, float, float], is_linked: bool) -> SocketEval:
    luma = constants.linear_luminance(rgb)
    return SocketEval(
        min_val=luma,
        max_val=luma,
        avg_val=luma,
        is_linked=is_linked,
        is_textured=False,
        avg_rgb=rgb,
        samples=None,
    )


def _eval_scalar_constant(val: float, is_linked: bool) -> SocketEval:
    v = float(val)
    return SocketEval(
        min_val=v,
        max_val=v,
        avg_val=v,
        is_linked=is_linked,
        is_textured=False,
        samples=None,
    )


def _combine_color_eval(
    node: bpy.types.Node,
    group_stack: Tuple[bpy.types.Node, ...],
    visited: Set[Tuple[int, Tuple[int, ...]]],
    pixel_cache: Dict[int, np.ndarray],
    top_linked: bool,
) -> Optional[SocketEval]:
    mode = getattr(node, "mode", "RGB")
    r_sock = get_socket(node.inputs, "Red") or _first_enabled_socket(node.inputs)
    g_sock = get_socket(node.inputs, "Green")
    b_sock = get_socket(node.inputs, "Blue")
    if r_sock is None or g_sock is None or b_sock is None:
        return None

    r_eval = _eval_scalar_recursive(r_sock, group_stack, visited, pixel_cache, top_linked)
    g_eval = _eval_scalar_recursive(g_sock, group_stack, visited, pixel_cache, top_linked)
    b_eval = _eval_scalar_recursive(b_sock, group_stack, visited, pixel_cache, top_linked)
    if r_eval is None or g_eval is None or b_eval is None:
        return None

    if r_eval.samples is None and g_eval.samples is None and b_eval.samples is None:
        c0, c1, c2 = r_eval.avg_val, g_eval.avg_val, b_eval.avg_val
        if mode == "HSV":
            rgb = colorsys.hsv_to_rgb(c0, c1, c2)
        elif mode == "HSL":
            rgb = colorsys.hls_to_rgb(c0, c2, c1)
        else:
            rgb = (c0, c1, c2)
        return _eval_color_constant(rgb, top_linked)

    lengths = [len(e.samples) for e in (r_eval, g_eval, b_eval) if e.samples is not None]
    n = max(lengths)
    c0_arr = r_eval.samples if r_eval.samples is not None else np.full(n, r_eval.avg_val, dtype=np.float64)
    c1_arr = g_eval.samples if g_eval.samples is not None else np.full(n, g_eval.avg_val, dtype=np.float64)
    c2_arr = b_eval.samples if b_eval.samples is not None else np.full(n, b_eval.avg_val, dtype=np.float64)
    if len(c0_arr) != n or len(c1_arr) != n or len(c2_arr) != n:
        c0, c1, c2 = r_eval.avg_val, g_eval.avg_val, b_eval.avg_val
        if mode == "HSV":
            rgb = colorsys.hsv_to_rgb(c0, c1, c2)
        elif mode == "HSL":
            rgb = colorsys.hls_to_rgb(c0, c2, c1)
        else:
            rgb = (c0, c1, c2)
        return _eval_color_constant(rgb, top_linked)

    if mode == "HSV":
        rgbs = np.array([colorsys.hsv_to_rgb(float(a), float(b), float(c)) for a, b, c in zip(c0_arr, c1_arr, c2_arr)])
    elif mode == "HSL":
        rgbs = np.array([colorsys.hls_to_rgb(float(a), float(c), float(b)) for a, b, c in zip(c0_arr, c1_arr, c2_arr)])
    else:
        rgbs = np.stack([c0_arr, c1_arr, c2_arr], axis=1)

    dummy_rgba = np.ones((len(rgbs), 4), dtype=np.float64)
    dummy_rgba[:, :3] = rgbs
    return _eval_color_from_samples(dummy_rgba, top_linked)


def _eval_clamp_color(
    clamp_node: bpy.types.Node,
    group_stack: Tuple[bpy.types.Node, ...],
    visited: Set[Tuple[int, Tuple[int, ...]]],
    pixel_cache: Dict[int, np.ndarray],
) -> Optional[SocketEval]:
    color_in = get_socket(clamp_node.inputs, "Color")
    if color_in is None:
        return None
    upstream_linked = color_in.is_linked
    color_eval = _eval_color_recursive(color_in, group_stack, visited.copy(), pixel_cache, upstream_linked)
    if color_eval is None:
        return None

    met_in = get_socket(clamp_node.inputs, "Metallic")
    met_eval = (
        _eval_scalar_recursive(met_in, group_stack, visited.copy(), pixel_cache, met_in.is_linked)
        if met_in is not None
        else None
    )
    raw_met = met_eval.avg_val if met_eval is not None else 0.0
    met_linked = met_in.is_linked if met_in is not None else False
    clamped_met = constants.soft_knee_clamp_metallic(raw_met, is_linked=met_linked)

    knee_in = get_socket(clamp_node.inputs, "Dielectric Knee")
    knee_eval = (
        _eval_scalar_recursive(knee_in, group_stack, visited.copy(), pixel_cache, knee_in.is_linked)
        if knee_in is not None
        else None
    )
    knee = knee_eval.avg_val if knee_eval is not None else constants.CLAMP_DIELECTRIC_KNEE

    max_in = get_socket(clamp_node.inputs, "Dielectric Max")
    max_eval = (
        _eval_scalar_recursive(max_in, group_stack, visited.copy(), pixel_cache, max_in.is_linked)
        if max_in is not None
        else None
    )
    ceil = max_eval.avg_val if max_eval is not None else constants.CLAMP_DIELECTRIC_MAX

    if color_eval.samples is not None:
        clamped_rgbs = np.array(
            [constants.soft_knee_clamp_color(rgb, clamped_met, knee, ceil) for rgb in color_eval.samples],
            dtype=np.float64,
        )
        lumas = (
            constants.LUMA_WEIGHTS[0] * clamped_rgbs[:, 0]
            + constants.LUMA_WEIGHTS[1] * clamped_rgbs[:, 1]
            + constants.LUMA_WEIGHTS[2] * clamped_rgbs[:, 2]
        )
        avg_rgb = (
            float(np.mean(clamped_rgbs[:, 0])),
            float(np.mean(clamped_rgbs[:, 1])),
            float(np.mean(clamped_rgbs[:, 2])),
        )
        return SocketEval(
            min_val=float(np.min(lumas)),
            max_val=float(np.max(lumas)),
            avg_val=float(np.mean(lumas)),
            is_linked=upstream_linked,
            is_textured=True,
            avg_rgb=avg_rgb,
            samples=clamped_rgbs,
        )

    c_rgb = constants.soft_knee_clamp_color(color_eval.avg_rgb, clamped_met, knee, ceil)
    return _eval_color_constant(c_rgb, upstream_linked)


def _eval_clamp_scalar(
    clamp_node: bpy.types.Node,
    socket_name: str,
    group_stack: Tuple[bpy.types.Node, ...],
    visited: Set[Tuple[int, Tuple[int, ...]]],
    pixel_cache: Dict[int, np.ndarray],
) -> Optional[SocketEval]:
    in_sock = get_socket(clamp_node.inputs, socket_name)
    if in_sock is None:
        return None
    upstream_linked = in_sock.is_linked
    raw_eval = _eval_scalar_recursive(in_sock, group_stack, visited.copy(), pixel_cache, upstream_linked)
    if raw_eval is None:
        return None

    if socket_name == "Metallic":
        if raw_eval.samples is not None:
            clamped = np.array(
                [constants.soft_knee_clamp_metallic(float(v), is_linked=upstream_linked) for v in raw_eval.samples],
                dtype=np.float64,
            )
            return SocketEval(
                min_val=float(np.min(clamped)),
                max_val=float(np.max(clamped)),
                avg_val=float(np.mean(clamped)),
                is_linked=upstream_linked,
                is_textured=True,
                samples=clamped,
            )
        return _eval_scalar_constant(
            constants.soft_knee_clamp_metallic(raw_eval.avg_val, is_linked=upstream_linked),
            upstream_linked,
        )

    if socket_name == "Roughness":
        trans_in = get_socket(clamp_node.inputs, "Transmission Weight")
        trans_eval = (
            _eval_scalar_recursive(trans_in, group_stack, visited.copy(), pixel_cache, trans_in.is_linked)
            if trans_in is not None
            else None
        )
        trans = trans_eval.avg_val if trans_eval is not None else 0.0
        if raw_eval.samples is not None:
            clamped = np.array(
                [constants.soft_knee_clamp_roughness(float(v), trans) for v in raw_eval.samples],
                dtype=np.float64,
            )
            return SocketEval(
                min_val=float(np.min(clamped)),
                max_val=float(np.max(clamped)),
                avg_val=float(np.mean(clamped)),
                is_linked=upstream_linked,
                is_textured=True,
                samples=clamped,
            )
        return _eval_scalar_constant(
            constants.soft_knee_clamp_roughness(raw_eval.avg_val, trans),
            upstream_linked,
        )

    if socket_name == "Specular IOR Level":
        if raw_eval.samples is not None:
            clamped = np.array(
                [constants.soft_knee_clamp_specular(float(v)) for v in raw_eval.samples],
                dtype=np.float64,
            )
            return SocketEval(
                min_val=float(np.min(clamped)),
                max_val=float(np.max(clamped)),
                avg_val=float(np.mean(clamped)),
                is_linked=upstream_linked,
                is_textured=True,
                samples=clamped,
            )
        return _eval_scalar_constant(constants.soft_knee_clamp_specular(raw_eval.avg_val), upstream_linked)

    if socket_name == "IOR":
        if raw_eval.samples is not None:
            clamped = np.array(
                [constants.soft_knee_clamp_ior(float(v)) for v in raw_eval.samples],
                dtype=np.float64,
            )
            return SocketEval(
                min_val=float(np.min(clamped)),
                max_val=float(np.max(clamped)),
                avg_val=float(np.mean(clamped)),
                is_linked=upstream_linked,
                is_textured=True,
                samples=clamped,
            )
        return _eval_scalar_constant(constants.soft_knee_clamp_ior(raw_eval.avg_val), upstream_linked)

    return raw_eval


def _eval_color_recursive(
    socket: Optional[bpy.types.NodeSocket],
    group_stack: Tuple[bpy.types.Node, ...],
    visited: Set[Tuple[int, Tuple[int, ...]]],
    pixel_cache: Dict[int, np.ndarray],
    top_linked: bool,
) -> Optional[SocketEval]:
    if socket is None:
        return None

    key = (socket.as_pointer(), tuple(g.as_pointer() for g in group_stack))
    if key in visited:
        return None
    visited.add(key)

    links = _valid_links(socket)
    if not links:
        if hasattr(socket, "default_value"):
            dv = socket.default_value
            if hasattr(dv, "__len__") and len(dv) >= 3:
                return _eval_color_constant((float(dv[0]), float(dv[1]), float(dv[2])), top_linked)
            v = float(dv)
            return _eval_color_constant((v, v, v), top_linked)
        return None

    link = links[0]
    from_node = link.from_node
    from_socket = link.from_socket

    if from_node.type == "REROUTE":
        in_sock = _first_enabled_socket(from_node.inputs, "Input")
        return _eval_color_recursive(in_sock, group_stack, visited, pixel_cache, top_linked)

    if from_node.type == "RGB":
        out_sock = _first_enabled_socket(from_node.outputs, "Color")
        if out_sock is not None and hasattr(out_sock, "default_value"):
            dv = out_sock.default_value
            return _eval_color_constant((float(dv[0]), float(dv[1]), float(dv[2])), top_linked)
        return None

    if from_node.type == "VALUE":
        out_sock = _first_enabled_socket(from_node.outputs, "Value")
        if out_sock is not None and hasattr(out_sock, "default_value"):
            v = float(out_sock.default_value)
            return _eval_color_constant((v, v, v), top_linked)
        return None

    if from_node.type == "COMBINE_COLOR":
        return _combine_color_eval(from_node, group_stack, visited, pixel_cache, top_linked)

    if from_node.type == "TEX_IMAGE":
        samples = _sample_image_rgba(getattr(from_node, "image", None), pixel_cache)
        if samples is None:
            return None
        if from_socket.name == "Alpha":
            alpha_rgb = np.repeat(samples[:, 3:4], 4, axis=1)
            return _eval_color_from_samples(alpha_rgb, top_linked)
        return _eval_color_from_samples(samples, top_linked)

    if from_node.type == "GROUP_INPUT":
        if group_stack:
            parent_node = group_stack[-1]
            parent_sock = _match_group_node_input(parent_node, from_socket)
            return _eval_color_recursive(parent_sock, group_stack[:-1], visited, pixel_cache, top_linked)
        return None

    if _is_clamp_node(from_node):
        return _eval_clamp_color(from_node, group_stack, visited, pixel_cache)

    if from_node.type == "GROUP":
        sub_tree = getattr(from_node, "node_tree", None)
        if sub_tree is not None:
            for go_sock in _match_group_output_inputs(sub_tree, from_socket):
                res = _eval_color_recursive(go_sock, group_stack + (from_node,), visited, pixel_cache, top_linked)
                if res is not None:
                    return res
        return None

    return None


def _eval_scalar_recursive(
    socket: Optional[bpy.types.NodeSocket],
    group_stack: Tuple[bpy.types.Node, ...],
    visited: Set[Tuple[int, Tuple[int, ...]]],
    pixel_cache: Dict[int, np.ndarray],
    top_linked: bool,
) -> Optional[SocketEval]:
    if socket is None:
        return None

    key = (socket.as_pointer(), tuple(g.as_pointer() for g in group_stack))
    if key in visited:
        return None
    visited.add(key)

    links = _valid_links(socket)
    if not links:
        if hasattr(socket, "default_value"):
            dv = socket.default_value
            if hasattr(dv, "__len__") and len(dv) >= 3:
                return _eval_scalar_constant(
                    constants.linear_luminance((float(dv[0]), float(dv[1]), float(dv[2]))),
                    top_linked,
                )
            return _eval_scalar_constant(float(dv), top_linked)
        return None

    link = links[0]
    from_node = link.from_node
    from_socket = link.from_socket

    if from_node.type == "REROUTE":
        in_sock = _first_enabled_socket(from_node.inputs, "Input")
        return _eval_scalar_recursive(in_sock, group_stack, visited, pixel_cache, top_linked)

    if from_node.type == "VALUE":
        out_sock = _first_enabled_socket(from_node.outputs, "Value")
        if out_sock is not None and hasattr(out_sock, "default_value"):
            return _eval_scalar_constant(float(out_sock.default_value), top_linked)
        return None

    if from_node.type == "RGB":
        out_sock = _first_enabled_socket(from_node.outputs, "Color")
        if out_sock is not None and hasattr(out_sock, "default_value"):
            dv = out_sock.default_value
            return _eval_scalar_constant(
                constants.linear_luminance((float(dv[0]), float(dv[1]), float(dv[2]))),
                top_linked,
            )
        return None

    if from_node.type == "COMBINE_COLOR":
        c_eval = _combine_color_eval(from_node, group_stack, visited, pixel_cache, top_linked)
        if c_eval is None:
            return None
        return SocketEval(
            min_val=c_eval.min_val,
            max_val=c_eval.max_val,
            avg_val=c_eval.avg_val,
            is_linked=top_linked,
            is_textured=c_eval.is_textured,
            samples=None,
        )

    if from_node.type == "SEPARATE_COLOR":
        in_sock = get_socket(from_node.inputs, "Color") or _first_enabled_socket(from_node.inputs)
        c_eval = _eval_color_recursive(in_sock, group_stack, visited, pixel_cache, top_linked)
        if c_eval is None:
            return None
        ch_map = {"Red": 0, "Green": 1, "Blue": 2}
        ch = ch_map.get(from_socket.name, 0)
        if c_eval.samples is not None:
            return _eval_scalar_from_samples(c_eval.samples[:, ch], top_linked)
        return _eval_scalar_constant(c_eval.avg_rgb[ch], top_linked)

    if from_node.type == "TEX_IMAGE":
        samples = _sample_image_rgba(getattr(from_node, "image", None), pixel_cache)
        if samples is None:
            return None
        if from_socket.name == "Alpha":
            return _eval_scalar_from_samples(samples[:, 3], top_linked)
        lumas = (
            constants.LUMA_WEIGHTS[0] * samples[:, 0]
            + constants.LUMA_WEIGHTS[1] * samples[:, 1]
            + constants.LUMA_WEIGHTS[2] * samples[:, 2]
        )
        return _eval_scalar_from_samples(lumas, top_linked)

    if from_node.type == "GROUP_INPUT":
        if group_stack:
            parent_node = group_stack[-1]
            parent_sock = _match_group_node_input(parent_node, from_socket)
            return _eval_scalar_recursive(parent_sock, group_stack[:-1], visited, pixel_cache, top_linked)
        return None

    if _is_clamp_node(from_node):
        return _eval_clamp_scalar(from_node, from_socket.name, group_stack, visited, pixel_cache)

    if from_node.type == "GROUP":
        sub_tree = getattr(from_node, "node_tree", None)
        if sub_tree is not None:
            for go_sock in _match_group_output_inputs(sub_tree, from_socket):
                res = _eval_scalar_recursive(go_sock, group_stack + (from_node,), visited, pixel_cache, top_linked)
                if res is not None:
                    return res
        return None

    return None


def evaluate_color_socket(
    socket: Optional[bpy.types.NodeSocket],
    pixel_cache: Optional[Dict[int, np.ndarray]] = None,
) -> Optional[SocketEval]:
    """Evaluate a color socket to min/max/avg linear Rec.709 luminance and RGB."""
    if socket is None:
        return None
    cache = pixel_cache if pixel_cache is not None else {}
    return _eval_color_recursive(socket, (), set(), cache, socket.is_linked)


def evaluate_scalar_socket(
    socket: Optional[bpy.types.NodeSocket],
    pixel_cache: Optional[Dict[int, np.ndarray]] = None,
) -> Optional[SocketEval]:
    """Evaluate a float scalar socket to min/max/avg value and linkage state."""
    if socket is None:
        return None
    cache = pixel_cache if pixel_cache is not None else {}
    return _eval_scalar_recursive(socket, (), set(), cache, socket.is_linked)


def _outlier_value(eval_res: SocketEval, low: float, high: float) -> float:
    low_dist = max(0.0, low - eval_res.min_val)
    high_dist = max(0.0, eval_res.max_val - high)
    return eval_res.min_val if low_dist >= high_dist else eval_res.max_val


def analyze_material(
    mat: Optional[bpy.types.Material],
    object_name: str = "",
    colorspace_map: Optional[Dict[bpy.types.Image, Dict[str, Set[str]]]] = None,
    pixel_cache: Optional[Dict[int, np.ndarray]] = None,
) -> List[DiagnosticIssue]:
    """Analyze a single material's active Principled BSDF(s) and colorspaces."""
    if mat is None or mat.node_tree is None:
        return []

    cache = pixel_cache if pixel_cache is not None else {}
    global_cs = colorspace_map if colorspace_map is not None else collect_colorspace_map([mat])
    local_cs = collect_colorspace_map([mat])

    issues: List[DiagnosticIssue] = []

    for bsdf in find_principled_bsdfs(mat.node_tree):
        base_sock = get_socket(bsdf.inputs, "Base Color")
        met_sock = get_socket(bsdf.inputs, "Metallic")
        rough_sock = get_socket(bsdf.inputs, "Roughness")
        spec_sock = get_socket(bsdf.inputs, "Specular IOR Level")
        ior_sock = get_socket(bsdf.inputs, "IOR")
        trans_sock = get_socket(bsdf.inputs, "Transmission Weight")

        base_eval = evaluate_color_socket(base_sock, cache)
        met_eval = evaluate_scalar_socket(met_sock, cache)
        rough_eval = evaluate_scalar_socket(rough_sock, cache)
        spec_eval = evaluate_scalar_socket(spec_sock, cache)
        ior_eval = evaluate_scalar_socket(ior_sock, cache)
        trans_eval = evaluate_scalar_socket(trans_sock, cache)

        if met_eval is not None:
            if (not met_eval.is_linked) and (
                constants.SEMI_METAL_LOW < met_eval.avg_val < constants.SEMI_METAL_HIGH
            ):
                issues.append(
                    DiagnosticIssue(
                        code="SEMI_METAL",
                        category=Category.ALBEDO,
                        severity=Severity.CRITICAL,
                        message=(
                            f"Material '{mat.name}' has unlinked semi-metal Metallic={met_eval.avg_val:.3f}"
                        ),
                        object_name=object_name,
                        material_name=mat.name,
                        measured_value=met_eval.avg_val,
                        expected_range="0.0 or 1.0 (unlinked)",
                        fixable=True,
                    )
                )

            if base_eval is not None:
                metallic_val = met_eval.avg_val
                if metallic_val < constants.METALLIC_CONDUCTOR_THRESHOLD:
                    if base_eval.min_val < constants.DIELECTRIC_MIN_LUMA:
                        issues.append(
                            DiagnosticIssue(
                                code="DIELECTRIC_CRUSHED_BLACK",
                                category=Category.ALBEDO,
                                severity=Severity.CRITICAL,
                                message=(
                                    f"Material '{mat.name}' dielectric Base Color luminance "
                                    f"{base_eval.min_val:.4f} < {constants.DIELECTRIC_MIN_LUMA}"
                                ),
                                object_name=object_name,
                                material_name=mat.name,
                                measured_value=base_eval.min_val,
                                expected_range=f"[{constants.DIELECTRIC_MIN_LUMA}, {constants.DIELECTRIC_WARN_LUMA}]",
                                fixable=True,
                            )
                        )
                    if base_eval.max_val > constants.DIELECTRIC_CRIT_LUMA:
                        issues.append(
                            DiagnosticIssue(
                                code="BLOWN_ALBEDO_CRIT",
                                category=Category.ALBEDO,
                                severity=Severity.CRITICAL,
                                message=(
                                    f"Material '{mat.name}' dielectric Base Color luminance "
                                    f"{base_eval.max_val:.4f} > {constants.DIELECTRIC_CRIT_LUMA}"
                                ),
                                object_name=object_name,
                                material_name=mat.name,
                                measured_value=base_eval.max_val,
                                expected_range=f"<= {constants.DIELECTRIC_WARN_LUMA}",
                                fixable=True,
                            )
                        )
                    elif base_eval.max_val > constants.DIELECTRIC_WARN_LUMA:
                        issues.append(
                            DiagnosticIssue(
                                code="BLOWN_ALBEDO_WARN",
                                category=Category.ALBEDO,
                                severity=Severity.WARNING,
                                message=(
                                    f"Material '{mat.name}' dielectric Base Color luminance "
                                    f"{base_eval.max_val:.4f} > {constants.DIELECTRIC_WARN_LUMA}"
                                ),
                                object_name=object_name,
                                material_name=mat.name,
                                measured_value=base_eval.max_val,
                                expected_range=f"<= {constants.DIELECTRIC_WARN_LUMA}",
                                fixable=True,
                            )
                        )
                else:
                    if base_eval.min_val < constants.CONDUCTOR_MIN_LUMA:
                        issues.append(
                            DiagnosticIssue(
                                code="DARK_METAL",
                                category=Category.ALBEDO,
                                severity=Severity.CRITICAL,
                                message=(
                                    f"Material '{mat.name}' conductor Base Color luminance "
                                    f"{base_eval.min_val:.4f} < {constants.CONDUCTOR_MIN_LUMA}"
                                ),
                                object_name=object_name,
                                material_name=mat.name,
                                measured_value=base_eval.min_val,
                                expected_range=f">= {constants.CONDUCTOR_MIN_LUMA}",
                                fixable=True,
                            )
                        )

        trans_val = trans_eval.avg_val if trans_eval is not None else 0.0
        if rough_eval is not None and trans_val < constants.TRANSMISSION_THRESHOLD:
            if rough_eval.min_val < constants.ROUGHNESS_MIN:
                issues.append(
                    DiagnosticIssue(
                        code="MIRROR_ROUGHNESS",
                        category=Category.ROUGHNESS,
                        severity=Severity.WARNING,
                        message=(
                            f"Material '{mat.name}' roughness {rough_eval.min_val:.4f} "
                            f"< {constants.ROUGHNESS_MIN}"
                        ),
                        object_name=object_name,
                        material_name=mat.name,
                        measured_value=rough_eval.min_val,
                        expected_range=f"[{constants.ROUGHNESS_MIN}, {constants.ROUGHNESS_MAX}]",
                        fixable=True,
                    )
                )
            if rough_eval.max_val > constants.ROUGHNESS_MAX:
                issues.append(
                    DiagnosticIssue(
                        code="DEAD_ROUGHNESS",
                        category=Category.ROUGHNESS,
                        severity=Severity.WARNING,
                        message=(
                            f"Material '{mat.name}' roughness {rough_eval.max_val:.4f} "
                            f"> {constants.ROUGHNESS_MAX}"
                        ),
                        object_name=object_name,
                        material_name=mat.name,
                        measured_value=rough_eval.max_val,
                        expected_range=f"[{constants.ROUGHNESS_MIN}, {constants.ROUGHNESS_MAX}]",
                        fixable=True,
                    )
                )

        if spec_eval is not None and (
            spec_eval.min_val < constants.SPECULAR_MIN or spec_eval.max_val > constants.SPECULAR_MAX
        ):
            measured_spec = _outlier_value(spec_eval, constants.SPECULAR_MIN, constants.SPECULAR_MAX)
            issues.append(
                DiagnosticIssue(
                    code="SPECULAR_OUTLIER",
                    category=Category.ROUGHNESS,
                    severity=Severity.WARNING,
                    message=(
                        f"Material '{mat.name}' Specular IOR Level {measured_spec:.3f} "
                        f"outside [{constants.SPECULAR_MIN}, {constants.SPECULAR_MAX}]"
                    ),
                    object_name=object_name,
                    material_name=mat.name,
                    measured_value=measured_spec,
                    expected_range=f"[{constants.SPECULAR_MIN}, {constants.SPECULAR_MAX}]",
                    fixable=True,
                )
            )

        if ior_eval is not None and (
            ior_eval.min_val < constants.IOR_MIN or ior_eval.max_val > constants.IOR_MAX
        ):
            measured_ior = _outlier_value(ior_eval, constants.IOR_MIN, constants.IOR_MAX)
            issues.append(
                DiagnosticIssue(
                    code="IOR_OUTLIER",
                    category=Category.ROUGHNESS,
                    severity=Severity.WARNING,
                    message=(
                        f"Material '{mat.name}' IOR {measured_ior:.3f} "
                        f"outside [{constants.IOR_MIN}, {constants.IOR_MAX}]"
                    ),
                    object_name=object_name,
                    material_name=mat.name,
                    measured_value=measured_ior,
                    expected_range=f"[{constants.IOR_MIN}, {constants.IOR_MAX}]",
                    fixable=True,
                )
            )

    for img, local_entry in local_cs.items():
        if not local_entry["data_sockets"]:
            continue
        cs_name = getattr(getattr(img, "colorspace_settings", None), "name", "")
        if cs_name != "Non-Color":
            global_entry = global_cs.get(img, local_entry)
            is_shared = bool(global_entry["color_sockets"])
            sockets_str = ", ".join(sorted(local_entry["data_sockets"]))
            issues.append(
                DiagnosticIssue(
                    code="DATA_MAP_SRGB",
                    category=Category.COLORSPACE,
                    severity=Severity.CRITICAL,
                    message=(
                        f"Image '{img.name}' on '{mat.name}' ({sockets_str}) uses '{cs_name}' "
                        f"instead of 'Non-Color'"
                        + (" (shared with color socket)" if is_shared else "")
                    ),
                    object_name=object_name,
                    material_name=mat.name,
                    image_name=img.name,
                    expected_range="Non-Color",
                    fixable=not is_shared,
                    is_shared_image=is_shared,
                )
            )

    return issues


def _iter_scene_materials(
    scene: Optional[bpy.types.Scene],
    selected_only: bool = False,
) -> List[Tuple[bpy.types.Material, str]]:
    """Collect unique valid materials referenced by scene objects with their first object name."""
    if scene is None:
        return []

    view_layer = None
    if selected_only:
        ctx = getattr(bpy, "context", None)
        if ctx is not None and getattr(ctx, "scene", None) == scene and getattr(ctx, "view_layer", None) is not None:
            view_layer = ctx.view_layer
        elif len(scene.view_layers) > 0:
            view_layer = scene.view_layers[0]

    seen_mats: Set[int] = set()
    mat_list: List[Tuple[bpy.types.Material, str]] = []

    for obj in scene.objects:
        if selected_only:
            try:
                is_selected = obj.select_get(view_layer=view_layer) if view_layer is not None else obj.select_get()
            except RuntimeError:
                is_selected = False
            if not is_selected:
                continue

        slots = getattr(obj, "material_slots", None)
        if not slots:
            continue

        for slot in slots:
            mat = getattr(slot, "material", None)
            if mat is None or mat.node_tree is None:
                continue
            ptr = mat.as_pointer()
            if ptr in seen_mats:
                continue
            seen_mats.add(ptr)
            mat_list.append((mat, obj.name))

    return mat_list


def scan_scene_materials(
    scene: Optional[bpy.types.Scene],
    selected_only: bool = False,
) -> Tuple[List[DiagnosticIssue], int]:
    """Scan all scene materials for PBR shader and colorspace issues."""
    mat_entries = _iter_scene_materials(scene, selected_only=selected_only)
    if not mat_entries:
        return [], 0

    materials = [mat for mat, _obj_name in mat_entries]
    colorspace_map = collect_colorspace_map(materials)
    pixel_cache: Dict[int, np.ndarray] = {}

    deduped: List[DiagnosticIssue] = []
    seen_keys: Set[Tuple[str, str, str]] = set()

    for mat, obj_name in mat_entries:
        mat_issues = analyze_material(
            mat,
            object_name=obj_name,
            colorspace_map=colorspace_map,
            pixel_cache=pixel_cache,
        )
        for issue in mat_issues:
            dedup_key = (issue.code, issue.material_name, issue.image_name)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            deduped.append(issue)

    return deduped, len(mat_entries)
