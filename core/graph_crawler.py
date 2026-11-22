"""Shader node-graph traversal, evaluation, colorspace mapping, and PBR analysis."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import threading
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

_CLAMP_PASSTHROUGH_MAP: Dict[str, str] = {
    "Color": "Color",
    "Metallic": "Metallic",
    "Roughness": "Roughness",
    "Specular IOR Level": "Specular IOR Level",
    "IOR": "IOR",
}

_IMAGE_SAMPLE_CACHE_MAX = 64
_IMAGE_SAMPLE_CACHE: "OrderedDict[Tuple[Any, ...], np.ndarray]" = OrderedDict()
_IMAGE_SAMPLE_CACHE_LOCK = threading.RLock()

_MATH_1_ARG = frozenset({
    "SQRT",
    "INVERSE_SQRT",
    "ABSOLUTE",
    "EXPONENT",
    "SIGN",
    "ROUND",
    "FLOOR",
    "CEIL",
    "TRUNC",
    "FRACT",
    "SINE",
    "COSINE",
    "TANGENT",
    "ARCSINE",
    "ARCCOSINE",
    "ARCTANGENT",
    "SINH",
    "COSH",
    "TANH",
    "RADIANS",
    "DEGREES",
})

_MATH_2_ARG = frozenset({
    "ADD",
    "SUBTRACT",
    "MULTIPLY",
    "DIVIDE",
    "POWER",
    "LOGARITHM",
    "MINIMUM",
    "MAXIMUM",
    "LESS_THAN",
    "GREATER_THAN",
    "MODULO",
    "FLOORED_MODULO",
    "SNAP",
    "PINGPONG",
    "ARCTAN2",
})

_MATH_3_ARG = frozenset({
    "MULTIPLY_ADD",
    "COMPARE",
    "SMOOTH_MIN",
    "SMOOTH_MAX",
    "WRAP",
})

_SUPPORTED_MIX_BLEND_MODES = frozenset({
    "MIX",
    "MULTIPLY",
    "ADD",
    "SUBTRACT",
    "SCREEN",
    "OVERLAY",
    "DARKEN",
    "LIGHTEN",
    "DIFFERENCE",
    "DIVIDE",
    "COLOR",
    "HUE",
    "SATURATION",
    "VALUE",
})


@dataclass
class SocketEval:
    """Evaluated value range, percentiles, and linkage metadata for a shader input socket."""

    min_val: float
    max_val: float
    avg_val: float
    is_linked: bool
    is_textured: bool = False
    avg_rgb: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    samples: Optional[np.ndarray] = None
    p01: Optional[float] = None
    p99: Optional[float] = None
    alpha: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.p01 is None:
            self.p01 = self.min_val
        if self.p99 is None:
            self.p99 = self.max_val


def clear_image_sample_cache() -> None:
    """Clear the bounded cross-scan LRU image pixel sample cache."""
    with _IMAGE_SAMPLE_CACHE_LOCK:
        _IMAGE_SAMPLE_CACHE.clear()


def get_socket(
    sockets: Any,
    name: str,
    socket_type: Optional[str] = None,
    occurrence: int = 0,
) -> Optional[bpy.types.NodeSocket]:
    """Return the ``occurrence``-th enabled socket matching ``name`` (and optional ``socket_type``)."""
    if sockets is None or occurrence < 0:
        return None
    count = 0
    for sock in sockets:
        if sock.name != name or not getattr(sock, "enabled", True):
            continue
        if socket_type is not None and sock.type != socket_type:
            continue
        if count == occurrence:
            return sock
        count += 1
    return None


def _get_factor_socket(sockets: Any) -> Optional[bpy.types.NodeSocket]:
    """Return the active Factor/Fac float input socket (Blender 5.x 'Factor', 4.x 'Fac')."""
    return (
        get_socket(sockets, "Factor", socket_type="VALUE")
        or get_socket(sockets, "Fac", socket_type="VALUE")
    )


def _first_enabled_socket(sockets: Any, preferred_name: Optional[str] = None) -> Optional[bpy.types.NodeSocket]:
    """Return ``preferred_name`` if enabled, else the first enabled non-CUSTOM socket."""
    if preferred_name is not None:
        sock = get_socket(sockets, preferred_name)
        if sock is not None:
            return sock
    if sockets is None:
        return None
    for sock in sockets:
        if getattr(sock, "enabled", True) and sock.type != "CUSTOM":
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


def _muted_passthrough_input(
    node: bpy.types.Node,
    out_socket: Optional[bpy.types.NodeSocket],
) -> Optional[bpy.types.NodeSocket]:
    """Return the primary enabled input passed through when ``node.mute`` is True."""
    if node.type == "BSDF_PRINCIPLED":
        return None

    for ilink in getattr(node, "internal_links", ()):
        to_sock = getattr(ilink, "to_socket", None)
        from_sock = getattr(ilink, "from_socket", None)
        if from_sock is None or not getattr(from_sock, "enabled", True):
            continue
        if out_socket is None:
            return from_sock
        if to_sock == out_socket or (
            to_sock is not None
            and to_sock.name == out_socket.name
            and to_sock.type == out_socket.type
        ):
            return from_sock

    if out_socket is not None:
        same_name = get_socket(node.inputs, out_socket.name, socket_type=out_socket.type)
        if same_name is not None:
            return same_name
        for sock in node.inputs:
            if getattr(sock, "enabled", True) and sock.type == out_socket.type:
                return sock
    return None


def _match_group_node_input(
    group_node: bpy.types.Node,
    group_input_out_socket: bpy.types.NodeSocket,
) -> Optional[bpy.types.NodeSocket]:
    """Map an output socket on an internal ``NodeGroupInput`` back to ``group_node.inputs``."""
    ident = getattr(group_input_out_socket, "identifier", "")
    for sock in group_node.inputs:
        if getattr(sock, "enabled", True) and ident and getattr(sock, "identifier", "") == ident:
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
                if (
                    getattr(sock, "enabled", True)
                    and sock.type != "CUSTOM"
                    and getattr(sock, "identifier", "") == ident
                ):
                    found = sock
                    break
        if found is None:
            found = get_socket(go.inputs, group_output_socket.name)
        if found is not None:
            matched.append(found)
    return matched


def _resolve_engine_target(engine: Optional[str] = None) -> Optional[str]:
    if engine is None:
        ctx = getattr(bpy, "context", None)
        scene = getattr(ctx, "scene", None) if ctx is not None else None
        render = getattr(scene, "render", None) if scene is not None else None
        engine = getattr(render, "engine", None) if render is not None else None
    if engine in {"BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "EEVEE"}:
        return "EEVEE"
    if engine == "CYCLES":
        return "CYCLES"
    return None


def find_active_material_output(
    node_tree: Optional[bpy.types.NodeTree],
    engine: Optional[str] = None,
) -> Optional[bpy.types.Node]:
    """Return the active ``OUTPUT_MATERIAL`` node matching ``engine``, ``ALL``, or fallback."""
    if node_tree is None:
        return None
    outputs = [
        n
        for n in node_tree.nodes
        if n.type == "OUTPUT_MATERIAL" and not getattr(n, "mute", False)
    ]
    if not outputs:
        return None

    engine_target = _resolve_engine_target(engine)
    if engine_target is not None:
        for node in outputs:
            if (
                getattr(node, "is_active_output", False)
                and getattr(node, "target", "ALL") == engine_target
            ):
                return node
        for node in outputs:
            if getattr(node, "target", "ALL") == engine_target:
                return node
    for node in outputs:
        if getattr(node, "is_active_output", False) and getattr(node, "target", "ALL") == "ALL":
            return node
    for node in outputs:
        if getattr(node, "target", "ALL") == "ALL":
            return node
    for node in outputs:
        if getattr(node, "is_active_output", False):
            return node
    return outputs[0]


def _find_principled_bsdfs_with_context(
    node_tree: Optional[bpy.types.NodeTree],
    engine: Optional[str] = None,
) -> List[Tuple[bpy.types.Node, Tuple[bpy.types.Node, ...]]]:
    """Return active non-muted ``(BSDF_PRINCIPLED, group_stack)`` reachable from output."""
    if node_tree is None:
        return []

    active_output = find_active_material_output(node_tree, engine=engine)
    surface_sock = get_socket(active_output.inputs, "Surface") if active_output is not None else None
    surface_links = _valid_links(surface_sock)

    if active_output is not None and surface_links:
        reachable: List[Tuple[bpy.types.Node, Tuple[bpy.types.Node, ...]]] = []
        seen_bsdfs: Set[int] = set()
        visited: Set[Tuple[int, Tuple[int, ...]]] = set()
        stack: List[Tuple[bpy.types.NodeSocket, Tuple[bpy.types.Node, ...]]] = [
            (surface_sock, ())
        ]

        while stack:
            sock, g_stack = stack.pop()
            for link in _valid_links(sock):
                from_node = link.from_node
                from_socket = link.from_socket
                visit_key = (
                    from_socket.as_pointer(),
                    tuple(g.as_pointer() for g in g_stack),
                )
                if visit_key in visited:
                    continue
                visited.add(visit_key)

                if getattr(from_node, "mute", False):
                    pass_sock = _muted_passthrough_input(from_node, from_socket)
                    if pass_sock is not None:
                        stack.append((pass_sock, g_stack))
                    continue

                if from_node.type == "BSDF_PRINCIPLED":
                    ptr = from_node.as_pointer()
                    if ptr not in seen_bsdfs:
                        seen_bsdfs.add(ptr)
                        reachable.append((from_node, g_stack))
                    continue

                if _is_overlay_node(from_node):
                    sh_in = get_socket(from_node.inputs, "Shader")
                    if sh_in is not None:
                        stack.append((sh_in, g_stack))
                    continue

                if _is_clamp_node(from_node):
                    continue

                if from_node.type == "GROUP_INPUT":
                    if g_stack:
                        parent_node = g_stack[-1]
                        parent_sock = _match_group_node_input(parent_node, from_socket)
                        if parent_sock is not None:
                            stack.append((parent_sock, g_stack[:-1]))
                    continue

                if from_node.type == "GROUP":
                    sub_tree = getattr(from_node, "node_tree", None)
                    if sub_tree is not None and from_node not in g_stack:
                        for go_sock in _match_group_output_inputs(sub_tree, from_socket):
                            stack.append((go_sock, g_stack + (from_node,)))
                    continue

                for in_sock in from_node.inputs:
                    if getattr(in_sock, "enabled", True) and in_sock.is_linked:
                        stack.append((in_sock, g_stack))

        return reachable

    return [
        (n, ())
        for n in node_tree.nodes
        if n.type == "BSDF_PRINCIPLED" and not getattr(n, "mute", False)
    ]


def find_principled_bsdfs(
    node_tree: Optional[bpy.types.NodeTree],
    engine: Optional[str] = None,
) -> List[bpy.types.Node]:
    """Return active non-muted ``BSDF_PRINCIPLED`` nodes reachable from active output."""
    return [bsdf for bsdf, _stack in _find_principled_bsdfs_with_context(node_tree, engine=engine)]


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

        if getattr(from_node, "mute", False):
            if from_node.type == "TEX_IMAGE":
                continue
            pass_sock = _muted_passthrough_input(from_node, from_socket)
            if pass_sock is not None:
                _trace_images_recursive(pass_sock, visited, group_stack, seen_pairs, results)
            continue

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

        if _is_clamp_node(from_node):
            mapped_in_name = _CLAMP_PASSTHROUGH_MAP.get(from_socket.name)
            if mapped_in_name is not None:
                in_sock = get_socket(from_node.inputs, mapped_in_name)
                _trace_images_recursive(in_sock, visited, group_stack, seen_pairs, results)
            continue

        if from_node.type == "GROUP":
            sub_tree = getattr(from_node, "node_tree", None)
            if sub_tree is not None and from_node not in group_stack:
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
            if getattr(in_sock, "enabled", True) and in_sock.is_linked:
                _trace_images_recursive(in_sock, visited, group_stack, seen_pairs, results)


def trace_upstream_images(
    socket: Optional[bpy.types.NodeSocket],
    visited: Optional[Set[Any]] = None,
    group_stack: Tuple[bpy.types.Node, ...] = (),
) -> List[Tuple[bpy.types.Node, bpy.types.Image]]:
    """Recursively walk upstream links (including reroutes and node groups) for ``TEX_IMAGE`` nodes."""
    if visited is None:
        visited = set()
    results: List[Tuple[bpy.types.Node, bpy.types.Image]] = []
    seen_pairs: Set[Tuple[int, int]] = set()
    _trace_images_recursive(socket, visited, group_stack, seen_pairs, results)
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

        if getattr(from_node, "mute", False):
            if from_node.type == "TEX_IMAGE":
                continue
            pass_sock = _muted_passthrough_input(from_node, from_socket)
            if pass_sock is not None:
                _collect_socket_images_with_tags(
                    pass_sock, data_tags, visited, group_stack, out_matches
                )
            continue

        if from_node.type == "TEX_IMAGE":
            if from_socket.name == "Alpha":
                continue
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

        if _is_clamp_node(from_node):
            mapped_in_name = _CLAMP_PASSTHROUGH_MAP.get(from_socket.name)
            if mapped_in_name is not None:
                in_sock = get_socket(from_node.inputs, mapped_in_name)
                _collect_socket_images_with_tags(
                    in_sock, data_tags, visited, group_stack, out_matches
                )
            continue

        next_tags = data_tags
        if from_node.type in constants.DATA_ONLY_NODE_TYPES:
            tag = _DATA_NODE_TAGS.get(from_node.type)
            if tag:
                next_tags = data_tags | {tag}

        if from_node.type == "GROUP":
            sub_tree = getattr(from_node, "node_tree", None)
            if sub_tree is not None and from_node not in group_stack:
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
            if getattr(in_sock, "enabled", True) and in_sock.is_linked:
                _collect_socket_images_with_tags(in_sock, next_tags, visited, group_stack, out_matches)


def _collect_colorspace_map_per_material(
    materials: Iterable[Optional[bpy.types.Material]],
) -> Tuple[
    Dict[bpy.types.Image, Dict[str, Set[str]]],
    Dict[int, Dict[bpy.types.Image, Dict[str, Set[str]]]],
]:
    """Build both global and per-material colorspace maps in a single graph pass."""
    global_cs: Dict[bpy.types.Image, Dict[str, Set[str]]] = {}
    per_mat_cs: Dict[int, Dict[bpy.types.Image, Dict[str, Set[str]]]] = {}

    def _ensure_entry(
        target_map: Dict[bpy.types.Image, Dict[str, Set[str]]],
        image: bpy.types.Image,
    ) -> Dict[str, Set[str]]:
        entry = target_map.get(image)
        if entry is None:
            entry = {
                "data_sockets": set(),
                "color_sockets": set(),
                "materials": set(),
                "nodes": set(),
            }
            target_map[image] = entry
        return entry

    def _record_socket(
        mat_name: str,
        local_map: Dict[bpy.types.Image, Dict[str, Set[str]]],
        sock: bpy.types.NodeSocket,
        group_stack: Tuple[bpy.types.Node, ...],
    ) -> None:
        if not getattr(sock, "enabled", True) or not sock.is_linked:
            return
        matches: List[Tuple[bpy.types.Node, bpy.types.Image, FrozenSet[str]]] = []
        _collect_socket_images_with_tags(sock, frozenset(), set(), group_stack, matches)
        for tex_node, img, path_tags in matches:
            for target_map in (global_cs, local_map):
                entry = _ensure_entry(target_map, img)
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
        ptr = mat.as_pointer()
        local_map = per_mat_cs.get(ptr)
        if local_map is None:
            local_map = {}
            per_mat_cs[ptr] = local_map
        else:
            continue

        for bsdf, g_stack in _find_principled_bsdfs_with_context(mat.node_tree):
            for sock in bsdf.inputs:
                _record_socket(mat.name, local_map, sock, g_stack)

        active_out = find_active_material_output(mat.node_tree)
        if active_out is not None:
            disp_sock = get_socket(active_out.inputs, "Displacement")
            if disp_sock is not None:
                _record_socket(mat.name, local_map, disp_sock, ())

    return global_cs, per_mat_cs


def collect_colorspace_map(
    materials: Iterable[Optional[bpy.types.Material]],
) -> Dict[bpy.types.Image, Dict[str, Set[str]]]:
    """Map each referenced ``bpy.types.Image`` to its data/color socket usage across ``materials``."""
    global_cs, _per_mat = _collect_colorspace_map_per_material(materials)
    return global_cs


def is_valid_data_colorspace(image: Optional[bpy.types.Image]) -> bool:
    """Return True if ``image`` uses a non-color or linear data color space."""
    if image is None:
        return False
    cs_settings = getattr(image, "colorspace_settings", None)
    if cs_settings is None:
        return False
    if bool(getattr(cs_settings, "is_data", False)):
        return True
    cs_name = str(getattr(cs_settings, "name", ""))
    if cs_name == "Non-Color":
        return True
    if bool(getattr(image, "is_float", False)):
        lower = cs_name.lower()
        if any(tok in lower for tok in ("linear", "raw", "aces", "non-color")):
            return True
    return False


def _image_cache_key(image: bpy.types.Image) -> Tuple[Any, ...]:
    size = getattr(image, "size", (0, 0))
    w = int(size[0]) if len(size) > 0 else 0
    h = int(size[1]) if len(size) > 1 else 0
    cs_name = str(getattr(getattr(image, "colorspace_settings", None), "name", ""))
    is_float = bool(getattr(image, "is_float", False))
    return (
        image.as_pointer(),
        str(getattr(image, "filepath", "")),
        w,
        h,
        bool(getattr(image, "is_dirty", False)),
        cs_name,
        is_float,
    )


def _sample_image_rgba(
    image: Optional[bpy.types.Image],
    pixel_cache: Optional[Dict[Any, np.ndarray]] = None,
) -> Optional[np.ndarray]:
    """Sample up to 4096 RGBA pixels evenly from ``image`` (linearizing display sRGB bytes)."""
    if image is None:
        return None

    cache_key = _image_cache_key(image)
    if pixel_cache is not None and cache_key in pixel_cache:
        return pixel_cache[cache_key]

    with _IMAGE_SAMPLE_CACHE_LOCK:
        cached = _IMAGE_SAMPLE_CACHE.get(cache_key)
        if cached is not None:
            _IMAGE_SAMPLE_CACHE.move_to_end(cache_key)
            if pixel_cache is not None:
                pixel_cache[cache_key] = cached
            return cached

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

    cs_name = str(getattr(getattr(image, "colorspace_settings", None), "name", ""))
    is_float = bool(getattr(image, "is_float", False))
    if (not is_float) and (cs_name in constants.SRGB_DISPLAY_SPACES):
        rgb = sampled[:, :3]
        sampled[:, :3] = np.where(
            rgb <= 0.04045,
            np.maximum(0.0, rgb) / 12.92,
            ((rgb + 0.055) / 1.055) ** 2.4,
        )

    with _IMAGE_SAMPLE_CACHE_LOCK:
        _IMAGE_SAMPLE_CACHE[cache_key] = sampled
        _IMAGE_SAMPLE_CACHE.move_to_end(cache_key)
        while len(_IMAGE_SAMPLE_CACHE) > _IMAGE_SAMPLE_CACHE_MAX:
            _IMAGE_SAMPLE_CACHE.popitem(last=False)

    if pixel_cache is not None:
        pixel_cache[cache_key] = sampled
    return sampled


def _luma_array(rgb: np.ndarray) -> np.ndarray:
    return (
        constants.LUMA_WEIGHTS[0] * rgb[:, 0]
        + constants.LUMA_WEIGHTS[1] * rgb[:, 1]
        + constants.LUMA_WEIGHTS[2] * rgb[:, 2]
    )


def _resample_array(arr: np.ndarray, target_len: int) -> np.ndarray:
    if len(arr) == target_len:
        return arr
    idx = np.linspace(0, len(arr) - 1, target_len, dtype=np.int64)
    return arr[idx]


def _eval_color_from_samples(
    samples: np.ndarray,
    is_linked: bool,
    alpha: Optional[np.ndarray] = None,
) -> SocketEval:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] >= 4:
        if alpha is None:
            alpha = arr[:, 3]
        rgb = arr[:, :3]
    else:
        rgb = arr[:, :3]

    lumas = _luma_array(rgb)
    if alpha is not None and len(alpha) == len(lumas) and np.any(alpha > 1e-3):
        pct_vals = lumas[alpha > 1e-3]
    else:
        pct_vals = lumas

    p01, p99 = np.percentile(pct_vals, [1.0, 99.0])
    avg_rgb = (float(np.mean(rgb[:, 0])), float(np.mean(rgb[:, 1])), float(np.mean(rgb[:, 2])))
    return SocketEval(
        min_val=float(np.min(lumas)),
        max_val=float(np.max(lumas)),
        avg_val=float(np.mean(lumas)),
        is_linked=is_linked,
        is_textured=True,
        avg_rgb=avg_rgb,
        samples=rgb,
        p01=float(p01),
        p99=float(p99),
        alpha=alpha,
    )


def _eval_scalar_from_samples(
    values: np.ndarray,
    is_linked: bool,
    alpha: Optional[np.ndarray] = None,
) -> SocketEval:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(flat) == 0:
        return _eval_scalar_constant(0.0, is_linked)
    if alpha is not None and len(alpha) == len(flat) and np.any(alpha > 1e-3):
        pct_vals = flat[alpha > 1e-3]
    else:
        pct_vals = flat
    p01, p99 = np.percentile(pct_vals, [1.0, 99.0])
    return SocketEval(
        min_val=float(np.min(flat)),
        max_val=float(np.max(flat)),
        avg_val=float(np.mean(flat)),
        is_linked=is_linked,
        is_textured=True,
        samples=flat,
        p01=float(p01),
        p99=float(p99),
        alpha=alpha,
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
        p01=luma,
        p99=luma,
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
        p01=v,
        p99=v,
    )


def _color_eval_to_scalar(c_eval: SocketEval, top_linked: bool) -> SocketEval:
    if c_eval.samples is not None:
        return _eval_scalar_from_samples(
            _luma_array(c_eval.samples), top_linked, alpha=c_eval.alpha
        )
    return SocketEval(
        min_val=c_eval.min_val,
        max_val=c_eval.max_val,
        avg_val=c_eval.avg_val,
        is_linked=top_linked,
        is_textured=c_eval.is_textured,
        samples=None,
        p01=c_eval.p01,
        p99=c_eval.p99,
    )


def _scalar_eval_to_color(s_eval: SocketEval, top_linked: bool) -> SocketEval:
    if s_eval.samples is not None:
        rgb = np.repeat(s_eval.samples[:, None], 3, axis=1)
        return _eval_color_from_samples(rgb, top_linked, alpha=s_eval.alpha)
    v = s_eval.avg_val
    return SocketEval(
        min_val=s_eval.min_val,
        max_val=s_eval.max_val,
        avg_val=v,
        is_linked=top_linked,
        is_textured=s_eval.is_textured,
        avg_rgb=(v, v, v),
        samples=None,
        p01=s_eval.p01,
        p99=s_eval.p99,
    )


def _align_scalar_evals(*evals: SocketEval) -> Tuple[int, List[np.ndarray], Optional[np.ndarray]]:
    textured = [e for e in evals if e.samples is not None]
    n = max((len(e.samples) for e in textured), default=1)
    arrays: List[np.ndarray] = []
    alpha_out: Optional[np.ndarray] = None
    for e in evals:
        if e.samples is not None:
            arr = _resample_array(e.samples, n)
            if alpha_out is None and e.alpha is not None:
                alpha_out = _resample_array(e.alpha, n)
        else:
            arr = np.full(n, e.avg_val, dtype=np.float64)
        arrays.append(arr)
    return n, arrays, alpha_out


def _rgb_to_hsv_np(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    delta = maxc - minc
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(maxc > 0.0, delta / maxc, 0.0)
        rc = np.where(delta > 0.0, (maxc - r) / delta, 0.0)
        gc = np.where(delta > 0.0, (maxc - g) / delta, 0.0)
        bc = np.where(delta > 0.0, (maxc - b) / delta, 0.0)
    h = np.where(
        delta <= 0.0,
        0.0,
        np.where(
            r == maxc,
            bc - gc,
            np.where(g == maxc, 2.0 + rc - bc, 4.0 + gc - rc),
        ),
    )
    h = (h / 6.0) % 1.0
    return np.stack([h, s, v], axis=1)


def _hsv_to_rgb_np(hsv: np.ndarray) -> np.ndarray:
    h = hsv[:, 0] % 1.0
    s = hsv[:, 1]
    v = hsv[:, 2]
    i = np.floor(h * 6.0).astype(np.int64)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=1)


def _hls_to_rgb_np(h: np.ndarray, l: np.ndarray, s: np.ndarray) -> np.ndarray:
    m2 = np.where(l <= 0.5, l * (1.0 + s), l + s - l * s)
    m1 = 2.0 * l - m2

    def _chan(hue: np.ndarray) -> np.ndarray:
        hu = hue % 1.0
        return np.where(
            hu < (1.0 / 6.0),
            m1 + (m2 - m1) * hu * 6.0,
            np.where(
                hu < 0.5,
                m2,
                np.where(hu < (2.0 / 3.0), m1 + (m2 - m1) * ((2.0 / 3.0) - hu) * 6.0, m1),
            ),
        )

    return np.stack([_chan(h + (1.0 / 3.0)), _chan(h), _chan(h - (1.0 / 3.0))], axis=1)


def _np_soft_knee_upper(y: np.ndarray, knee: Any, ceiling: Any) -> np.ndarray:
    ceil_f = np.asarray(ceiling, dtype=np.float64)
    eff_knee = np.minimum(np.asarray(knee, dtype=np.float64), ceil_f - 1e-4)
    span = np.maximum(ceil_f - eff_knee, 1e-6)
    return np.where(y <= eff_knee, y, eff_knee + span * np.tanh((y - eff_knee) / span))


def _np_soft_knee_lower(y: np.ndarray, floor: Any, knee: Any) -> np.ndarray:
    floor_f = np.asarray(floor, dtype=np.float64)
    eff_knee = np.maximum(np.asarray(knee, dtype=np.float64), floor_f + 1e-4)
    span = np.maximum(eff_knee - floor_f, 1e-6)
    return np.where(y >= eff_knee, y, eff_knee - span * np.tanh((eff_knee - y) / span))


def _combine_color_eval(
    node: bpy.types.Node,
    group_stack: Tuple[bpy.types.Node, ...],
    visited: Set[Tuple[int, Tuple[int, ...]]],
    pixel_cache: Dict[Any, np.ndarray],
    top_linked: bool,
) -> Optional[SocketEval]:
    mode = getattr(node, "mode", "RGB")
    r_sock = get_socket(node.inputs, "Red") or _first_enabled_socket(node.inputs)
    g_sock = get_socket(node.inputs, "Green")
    b_sock = get_socket(node.inputs, "Blue")
    if r_sock is None or g_sock is None or b_sock is None:
        return None

    r_eval = _eval_scalar_recursive(r_sock, group_stack, visited.copy(), pixel_cache, top_linked)
    g_eval = _eval_scalar_recursive(g_sock, group_stack, visited.copy(), pixel_cache, top_linked)
    b_eval = _eval_scalar_recursive(b_sock, group_stack, visited.copy(), pixel_cache, top_linked)
    if r_eval is None or g_eval is None or b_eval is None:
        return None

    is_const = r_eval.samples is None and g_eval.samples is None and b_eval.samples is None
    _n, (c0_arr, c1_arr, c2_arr), alpha_out = _align_scalar_evals(r_eval, g_eval, b_eval)

    if mode == "HSV":
        rgbs = _hsv_to_rgb_np(np.stack([c0_arr, c1_arr, c2_arr], axis=1))
    elif mode == "HSL":
        rgbs = _hls_to_rgb_np(c0_arr, c2_arr, c1_arr)
    else:
        rgbs = np.stack([c0_arr, c1_arr, c2_arr], axis=1)

    if is_const:
        return _eval_color_constant((float(rgbs[0, 0]), float(rgbs[0, 1]), float(rgbs[0, 2])), top_linked)

    return _eval_color_from_samples(rgbs, top_linked, alpha=alpha_out)


def _eval_clamp_color(
    clamp_node: bpy.types.Node,
    group_stack: Tuple[bpy.types.Node, ...],
    visited: Set[Tuple[int, Tuple[int, ...]]],
    pixel_cache: Dict[Any, np.ndarray],
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
    met_linked = met_in.is_linked if met_in is not None else False
    met_linked_sock = get_socket(clamp_node.inputs, "Metallic Linked")
    if (not met_linked) and met_linked_sock is not None:
        ml_eval = _eval_scalar_recursive(
            met_linked_sock, group_stack, visited.copy(), pixel_cache, met_linked_sock.is_linked
        )
        if ml_eval is not None and ml_eval.avg_val >= 0.5:
            met_linked = True

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

    has_array = (color_eval.samples is not None) or (met_eval is not None and met_eval.samples is not None)
    if not has_array:
        raw_met = met_eval.avg_val if met_eval is not None else 0.0
        clamped_met = constants.soft_knee_clamp_metallic(raw_met, is_linked=met_linked)
        c_rgb = constants.soft_knee_clamp_color(color_eval.avg_rgb, clamped_met, knee, ceil)
        return _eval_color_constant(c_rgb, upstream_linked)

    n = max(
        len(color_eval.samples) if color_eval.samples is not None else 1,
        len(met_eval.samples) if (met_eval is not None and met_eval.samples is not None) else 1,
    )
    if color_eval.samples is not None:
        rgb = np.maximum(_resample_array(color_eval.samples, n), 0.0)
        alpha_out = _resample_array(color_eval.alpha, n) if color_eval.alpha is not None else None
    else:
        rgb = np.maximum(np.broadcast_to(np.array(color_eval.avg_rgb, dtype=np.float64), (n, 3)), 0.0)
        alpha_out = None

    if met_eval is not None and met_eval.samples is not None:
        raw_met_arr = _resample_array(met_eval.samples, n)
    else:
        raw_met_val = met_eval.avg_val if met_eval is not None else 0.0
        raw_met_arr = np.full(n, raw_met_val, dtype=np.float64)

    clamped_met_arr = np.clip(raw_met_arr, 0.0, 1.0)
    if not met_linked:
        semi_mask = (clamped_met_arr > constants.SEMI_METAL_LOW) & (
            clamped_met_arr < constants.SEMI_METAL_HIGH
        )
        clamped_met_arr = np.where(
            semi_mask,
            np.where(clamped_met_arr >= constants.METALLIC_CONDUCTOR_THRESHOLD, 1.0, 0.0),
            clamped_met_arr,
        )

    y = _luma_array(rgb)
    y_cond = _np_soft_knee_lower(
        y, constants.CLAMP_CONDUCTOR_FLOOR, constants.CLAMP_CONDUCTOR_FLOOR_KNEE
    )
    y_diel_up = _np_soft_knee_upper(y, knee, ceil)
    y_diel = _np_soft_knee_lower(
        y_diel_up, constants.CLAMP_DIELECTRIC_FLOOR, constants.CLAMP_DIELECTRIC_FLOOR_KNEE
    )
    y_target = np.where(clamped_met_arr >= constants.METALLIC_CONDUCTOR_THRESHOLD, y_cond, y_diel)

    scale = y_target / np.maximum(y, 1e-6)
    rgb_scaled = rgb * scale[:, None]
    t = np.clip(y / 1e-3, 0.0, 1.0)[:, None]
    gray = y_target[:, None]
    clamped_rgbs = np.where(y[:, None] >= 1e-3, rgb_scaled, gray * (1.0 - t) + rgb_scaled * t)

    return _eval_color_from_samples(clamped_rgbs, upstream_linked, alpha=alpha_out)


def _eval_clamp_scalar(
    clamp_node: bpy.types.Node,
    socket_name: str,
    group_stack: Tuple[bpy.types.Node, ...],
    visited: Set[Tuple[int, Tuple[int, ...]]],
    pixel_cache: Dict[Any, np.ndarray],
) -> Optional[SocketEval]:
    in_sock = get_socket(clamp_node.inputs, socket_name)
    if in_sock is None:
        return None
    upstream_linked = in_sock.is_linked
    raw_eval = _eval_scalar_recursive(in_sock, group_stack, visited.copy(), pixel_cache, upstream_linked)
    if raw_eval is None:
        return None

    if socket_name == "Metallic":
        met_linked = upstream_linked
        met_linked_sock = get_socket(clamp_node.inputs, "Metallic Linked")
        if (not met_linked) and met_linked_sock is not None:
            ml_eval = _eval_scalar_recursive(
                met_linked_sock, group_stack, visited.copy(), pixel_cache, met_linked_sock.is_linked
            )
            if ml_eval is not None and ml_eval.avg_val >= 0.5:
                met_linked = True

        if raw_eval.samples is None:
            return _eval_scalar_constant(
                constants.soft_knee_clamp_metallic(raw_eval.avg_val, is_linked=met_linked),
                upstream_linked,
            )
        clamped = np.clip(raw_eval.samples, 0.0, 1.0)
        if not met_linked:
            semi = (clamped > constants.SEMI_METAL_LOW) & (clamped < constants.SEMI_METAL_HIGH)
            clamped = np.where(
                semi,
                np.where(clamped >= constants.METALLIC_CONDUCTOR_THRESHOLD, 1.0, 0.0),
                clamped,
            )
        return _eval_scalar_from_samples(clamped, upstream_linked, alpha=raw_eval.alpha)

    if socket_name == "Roughness":
        trans_in = get_socket(clamp_node.inputs, "Transmission Weight")
        trans_eval = (
            _eval_scalar_recursive(trans_in, group_stack, visited.copy(), pixel_cache, trans_in.is_linked)
            if trans_in is not None
            else None
        )
        has_array = (raw_eval.samples is not None) or (
            trans_eval is not None and trans_eval.samples is not None
        )
        if not has_array:
            trans = trans_eval.avg_val if trans_eval is not None else 0.0
            return _eval_scalar_constant(
                constants.soft_knee_clamp_roughness(raw_eval.avg_val, trans),
                upstream_linked,
            )

        if trans_eval is not None:
            _n, (val_arr, trans_arr), alpha_out = _align_scalar_evals(raw_eval, trans_eval)
        else:
            val_arr = raw_eval.samples
            trans_arr = np.zeros_like(val_arr)
            alpha_out = raw_eval.alpha

        val_clipped = np.clip(val_arr, 0.0, 1.0)
        low_lifted = _np_soft_knee_lower(
            val_clipped, constants.CLAMP_ROUGHNESS_FLOOR, constants.CLAMP_ROUGHNESS_FLOOR_KNEE
        )
        rgh_clamped = _np_soft_knee_upper(
            low_lifted, constants.CLAMP_ROUGHNESS_CEIL_KNEE, constants.CLAMP_ROUGHNESS_CEIL
        )
        out_arr = np.where(trans_arr >= constants.TRANSMISSION_THRESHOLD, val_clipped, rgh_clamped)
        return _eval_scalar_from_samples(out_arr, upstream_linked, alpha=alpha_out)

    if socket_name == "Specular IOR Level":
        if raw_eval.samples is None:
            return _eval_scalar_constant(
                constants.soft_knee_clamp_specular(raw_eval.avg_val), upstream_linked
            )
        low_lifted = _np_soft_knee_lower(
            raw_eval.samples, constants.CLAMP_SPECULAR_MIN, constants.CLAMP_SPECULAR_MIN_KNEE
        )
        clamped = _np_soft_knee_upper(
            low_lifted, constants.CLAMP_SPECULAR_MAX_KNEE, constants.CLAMP_SPECULAR_MAX
        )
        return _eval_scalar_from_samples(clamped, upstream_linked, alpha=raw_eval.alpha)

    if socket_name == "IOR":
        if raw_eval.samples is None:
            return _eval_scalar_constant(
                constants.soft_knee_clamp_ior(raw_eval.avg_val), upstream_linked
            )
        low_lifted = _np_soft_knee_lower(
            raw_eval.samples, constants.CLAMP_IOR_MIN, constants.CLAMP_IOR_MIN_KNEE
        )
        clamped = _np_soft_knee_upper(
            low_lifted, constants.CLAMP_IOR_MAX_KNEE, constants.CLAMP_IOR_MAX
        )
        return _eval_scalar_from_samples(clamped, upstream_linked, alpha=raw_eval.alpha)

    return raw_eval


def _blend_rgb_np(blend_type: str, a: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    if blend_type == "MIX":
        return b
    if blend_type == "MULTIPLY":
        return a * b
    if blend_type == "ADD":
        return a + b
    if blend_type == "SUBTRACT":
        return a - b
    if blend_type == "SCREEN":
        return 1.0 - (1.0 - a) * (1.0 - b)
    if blend_type == "OVERLAY":
        return np.where(a < 0.5, 2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b))
    if blend_type == "DARKEN":
        return np.minimum(a, b)
    if blend_type == "LIGHTEN":
        return np.maximum(a, b)
    if blend_type == "DIFFERENCE":
        return np.abs(a - b)
    if blend_type == "DIVIDE":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(b != 0.0, a / np.where(b != 0.0, b, 1.0), a)
    if blend_type in {"COLOR", "HUE", "SATURATION", "VALUE"}:
        hsv_a = _rgb_to_hsv_np(a)
        hsv_b = _rgb_to_hsv_np(b)
        if blend_type == "HUE":
            out_hsv = np.stack([hsv_b[:, 0], hsv_a[:, 1], hsv_a[:, 2]], axis=1)
            return np.where((hsv_b[:, 1] != 0.0)[:, None], _hsv_to_rgb_np(out_hsv), a)
        if blend_type == "SATURATION":
            out_hsv = np.stack([hsv_a[:, 0], hsv_b[:, 1], hsv_a[:, 2]], axis=1)
            return np.where((hsv_a[:, 1] != 0.0)[:, None], _hsv_to_rgb_np(out_hsv), a)
        if blend_type == "VALUE":
            out_hsv = np.stack([hsv_a[:, 0], hsv_a[:, 1], hsv_b[:, 2]], axis=1)
            return _hsv_to_rgb_np(out_hsv)
        out_hsv = np.stack([hsv_b[:, 0], hsv_b[:, 1], hsv_a[:, 2]], axis=1)
        return np.where((hsv_b[:, 1] != 0.0)[:, None], _hsv_to_rgb_np(out_hsv), a)
    return None


def _envelope_color_evals(ev0: SocketEval, ev1: SocketEval, top_linked: bool) -> SocketEval:
    p01 = min(ev0.p01, ev1.p01)
    p99 = max(ev0.p99, ev1.p99)
    avg_rgb = tuple(0.5 * (ev0.avg_rgb[i] + ev1.avg_rgb[i]) for i in range(3))
    return SocketEval(
        min_val=min(ev0.min_val, ev1.min_val),
        max_val=max(ev0.max_val, ev1.max_val),
        avg_val=0.5 * (ev0.avg_val + ev1.avg_val),
        is_linked=top_linked,
        is_textured=ev0.is_textured or ev1.is_textured,
        avg_rgb=(avg_rgb[0], avg_rgb[1], avg_rgb[2]),
        samples=None,
        p01=p01,
        p99=p99,
    )


def _envelope_scalar_evals(ev0: SocketEval, ev1: SocketEval, top_linked: bool) -> SocketEval:
    return SocketEval(
        min_val=min(ev0.min_val, ev1.min_val),
        max_val=max(ev0.max_val, ev1.max_val),
        avg_val=0.5 * (ev0.avg_val + ev1.avg_val),
        is_linked=top_linked,
        is_textured=ev0.is_textured or ev1.is_textured,
        samples=None,
        p01=min(ev0.p01, ev1.p01),
        p99=max(ev0.p99, ev1.p99),
    )


def _eval_mix_float_node(
    node: bpy.types.Node,
    group_stack: Tuple[bpy.types.Node, ...],
    visited: Set[Tuple[int, Tuple[int, ...]]],
    pixel_cache: Dict[Any, np.ndarray],
    top_linked: bool,
) -> Optional[SocketEval]:
    fac_sock = _get_factor_socket(node.inputs)
    a_sock = get_socket(node.inputs, "A", socket_type="VALUE")
    b_sock = get_socket(node.inputs, "B", socket_type="VALUE")
    if a_sock is None or b_sock is None:
        return None

    a_eval = _eval_scalar_recursive(a_sock, group_stack, visited.copy(), pixel_cache, top_linked)
    b_eval = _eval_scalar_recursive(b_sock, group_stack, visited.copy(), pixel_cache, top_linked)
    if a_eval is None or b_eval is None:
        return None

    clamp_factor = bool(getattr(node, "clamp_factor", True))
    clamp_result = bool(getattr(node, "clamp_result", False))

    fac_eval = (
        _eval_scalar_recursive(fac_sock, group_stack, visited.copy(), pixel_cache, top_linked)
        if fac_sock is not None
        else _eval_scalar_constant(0.5, top_linked)
    )
    if fac_eval is None:
        if not clamp_factor:
            return None
        ev0 = a_eval
        ev1 = b_eval
        if clamp_result:
            ev0 = _clip_scalar_eval(ev0, 0.0, 1.0, top_linked)
            ev1 = _clip_scalar_eval(ev1, 0.0, 1.0, top_linked)
        return _envelope_scalar_evals(ev0, ev1, top_linked)

    is_const = fac_eval.samples is None and a_eval.samples is None and b_eval.samples is None
    _n, (f_arr, a_arr, b_arr), alpha_out = _align_scalar_evals(fac_eval, a_eval, b_eval)
    if clamp_factor:
        f_arr = np.clip(f_arr, 0.0, 1.0)
    out = (1.0 - f_arr) * a_arr + f_arr * b_arr
    if clamp_result:
        out = np.clip(out, 0.0, 1.0)
    if is_const:
        return _eval_scalar_constant(float(out[0]), top_linked)
    return _eval_scalar_from_samples(out, top_linked, alpha=alpha_out)


def _clip_scalar_eval(ev: SocketEval, lo: float, hi: float, top_linked: bool) -> SocketEval:
    if ev.samples is not None:
        return _eval_scalar_from_samples(np.clip(ev.samples, lo, hi), top_linked, alpha=ev.alpha)
    return _eval_scalar_constant(min(max(ev.avg_val, lo), hi), top_linked)


def _clip_color_eval(ev: SocketEval, lo: float, hi: float, top_linked: bool) -> SocketEval:
    if ev.samples is not None:
        return _eval_color_from_samples(np.clip(ev.samples, lo, hi), top_linked, alpha=ev.alpha)
    rgb = tuple(min(max(c, lo), hi) for c in ev.avg_rgb)
    return _eval_color_constant((rgb[0], rgb[1], rgb[2]), top_linked)


def _eval_mix_rgba_node(
    node: bpy.types.Node,
    group_stack: Tuple[bpy.types.Node, ...],
    visited: Set[Tuple[int, Tuple[int, ...]]],
    pixel_cache: Dict[Any, np.ndarray],
    top_linked: bool,
) -> Optional[SocketEval]:
    blend_type = getattr(node, "blend_type", "MIX")
    if blend_type not in _SUPPORTED_MIX_BLEND_MODES:
        return None

    fac_sock = _get_factor_socket(node.inputs)
    if node.type == "MIX_RGB":
        a_sock = get_socket(node.inputs, "Color1", socket_type="RGBA")
        b_sock = get_socket(node.inputs, "Color2", socket_type="RGBA")
        clamp_factor = True
        clamp_result = bool(getattr(node, "use_clamp", False))
    else:
        a_sock = get_socket(node.inputs, "A", socket_type="RGBA")
        b_sock = get_socket(node.inputs, "B", socket_type="RGBA")
        clamp_factor = bool(getattr(node, "clamp_factor", True))
        clamp_result = bool(getattr(node, "clamp_result", False))

    if a_sock is None or b_sock is None:
        return None

    a_eval = _eval_color_recursive(a_sock, group_stack, visited.copy(), pixel_cache, top_linked)
    b_eval = _eval_color_recursive(b_sock, group_stack, visited.copy(), pixel_cache, top_linked)
    if a_eval is None or b_eval is None:
        return None

    fac_eval = (
        _eval_scalar_recursive(fac_sock, group_stack, visited.copy(), pixel_cache, top_linked)
        if fac_sock is not None
        else _eval_scalar_constant(0.5, top_linked)
    )

    def _compute_mix(t_override: Optional[float]) -> Optional[SocketEval]:
        has_array = (
            a_eval.samples is not None
            or b_eval.samples is not None
            or (t_override is None and fac_eval is not None and fac_eval.samples is not None)
        )
        n = max(
            len(a_eval.samples) if a_eval.samples is not None else 1,
            len(b_eval.samples) if b_eval.samples is not None else 1,
            len(fac_eval.samples)
            if (t_override is None and fac_eval is not None and fac_eval.samples is not None)
            else 1,
        )

        a_arr = (
            _resample_array(a_eval.samples, n)
            if a_eval.samples is not None
            else np.broadcast_to(np.array(a_eval.avg_rgb, dtype=np.float64), (n, 3)).copy()
        )
        b_arr = (
            _resample_array(b_eval.samples, n)
            if b_eval.samples is not None
            else np.broadcast_to(np.array(b_eval.avg_rgb, dtype=np.float64), (n, 3)).copy()
        )
        if t_override is not None:
            t_arr = np.full((n, 1), float(t_override), dtype=np.float64)
        elif fac_eval is not None and fac_eval.samples is not None:
            t_arr = _resample_array(fac_eval.samples, n)[:, None]
        else:
            t_val = fac_eval.avg_val if fac_eval is not None else 0.5
            t_arr = np.full((n, 1), t_val, dtype=np.float64)

        if clamp_factor:
            t_arr = np.clip(t_arr, 0.0, 1.0)

        blended = _blend_rgb_np(blend_type, a_arr, b_arr)
        if blended is None:
            return None
        out_arr = (1.0 - t_arr) * a_arr + t_arr * blended
        if clamp_result:
            out_arr = np.clip(out_arr, 0.0, 1.0)

        if not has_array:
            return _eval_color_constant(
                (float(out_arr[0, 0]), float(out_arr[0, 1]), float(out_arr[0, 2])),
                top_linked,
            )

        alpha_out: Optional[np.ndarray] = None
        if a_eval.alpha is not None and b_eval.alpha is not None:
            alpha_out = np.maximum(_resample_array(a_eval.alpha, n), _resample_array(b_eval.alpha, n))
        elif a_eval.alpha is not None and np.all(t_arr <= 0.0):
            alpha_out = _resample_array(a_eval.alpha, n)
        elif b_eval.alpha is not None and np.all(t_arr >= 1.0):
            alpha_out = _resample_array(b_eval.alpha, n)

        return _eval_color_from_samples(out_arr, top_linked, alpha=alpha_out)

    if fac_eval is None:
        if not clamp_factor:
            return None
        ev0 = _clip_color_eval(a_eval, 0.0, 1.0, top_linked) if clamp_result else a_eval
        ev1 = _compute_mix(1.0)
        if ev1 is None:
            return None
        return _envelope_color_evals(ev0, ev1, top_linked)

    return _compute_mix(None)


def _eval_math_op_np(
    op: str,
    a: np.ndarray,
    b: Optional[np.ndarray],
    c: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if op == "ADD" and b is not None:
            return a + b
        if op == "SUBTRACT" and b is not None:
            return a - b
        if op == "MULTIPLY" and b is not None:
            return a * b
        if op == "DIVIDE" and b is not None:
            return np.where(b != 0.0, a / np.where(b != 0.0, b, 1.0), 0.0)
        if op == "MULTIPLY_ADD" and b is not None and c is not None:
            return a * b + c
        if op == "POWER" and b is not None:
            is_int_exp = np.isfinite(b) & (b == np.trunc(b))
            valid_neg_base = (a < 0.0) & is_int_exp & ~((a == 0.0) & (b < 0.0))
            valid_pos = (a > 0.0) | ((a == 0.0) & (b >= 0.0))
            safe_a = np.where(valid_pos | valid_neg_base, a, 1.0)
            safe_b = np.where(valid_pos | valid_neg_base, b, 1.0)
            out = np.power(safe_a, safe_b)
            out = np.where((a == 0.0) & (b == 0.0), 1.0, out)
            return np.where(valid_pos | valid_neg_base, np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        if op == "LOGARITHM" and b is not None:
            valid = (a > 0.0) & (b > 0.0) & (b != 1.0)
            safe_a = np.where(valid, a, 2.0)
            safe_b = np.where(valid, b, 2.0)
            return np.where(valid, np.log(safe_a) / np.log(safe_b), 0.0)
        if op == "SQRT":
            return np.where(a >= 0.0, np.sqrt(np.maximum(a, 0.0)), 0.0)
        if op == "INVERSE_SQRT":
            return np.where(a > 0.0, 1.0 / np.sqrt(np.maximum(a, 1e-12)), 0.0)
        if op == "ABSOLUTE":
            return np.abs(a)
        if op == "EXPONENT":
            return np.exp(np.clip(a, -80.0, 80.0))
        if op == "MINIMUM" and b is not None:
            return np.minimum(a, b)
        if op == "MAXIMUM" and b is not None:
            return np.maximum(a, b)
        if op == "LESS_THAN" and b is not None:
            return np.where(a < b, 1.0, 0.0)
        if op == "GREATER_THAN" and b is not None:
            return np.where(a > b, 1.0, 0.0)
        if op == "SIGN":
            return np.sign(a)
        if op == "COMPARE" and b is not None and c is not None:
            return np.where(np.abs(a - b) <= np.maximum(c, 1e-5), 1.0, 0.0)
        if op == "SMOOTH_MIN" and b is not None and c is not None:
            k = np.abs(c)
            h = np.where(k > 0.0, np.maximum(k - np.abs(a - b), 0.0) / np.where(k > 0.0, k, 1.0), 0.0)
            return np.where(k > 0.0, np.minimum(a, b) - (h * h * h) * k * (1.0 / 6.0), np.minimum(a, b))
        if op == "SMOOTH_MAX" and b is not None and c is not None:
            k = np.abs(c)
            h = np.where(k > 0.0, np.maximum(k - np.abs(a - b), 0.0) / np.where(k > 0.0, k, 1.0), 0.0)
            return np.where(k > 0.0, np.maximum(a, b) + (h * h * h) * k * (1.0 / 6.0), np.maximum(a, b))
        if op == "ROUND":
            return np.floor(a + 0.5)
        if op == "FLOOR":
            return np.floor(a)
        if op == "CEIL":
            return np.ceil(a)
        if op == "TRUNC":
            return np.trunc(a)
        if op == "FRACT":
            return a - np.floor(a)
        if op == "MODULO" and b is not None:
            return np.where(b != 0.0, np.fmod(a, np.where(b != 0.0, b, 1.0)), 0.0)
        if op == "FLOORED_MODULO" and b is not None:
            safe_b = np.where(b != 0.0, b, 1.0)
            return np.where(b != 0.0, a - np.floor(a / safe_b) * safe_b, 0.0)
        if op == "WRAP" and b is not None and c is not None:
            span = b - c
            safe_span = np.where(span != 0.0, span, 1.0)
            return np.where(span != 0.0, a - safe_span * np.floor((a - c) / safe_span), c)
        if op == "SNAP" and b is not None:
            safe_b = np.where(b != 0.0, b, 1.0)
            return np.where(b != 0.0, np.floor(a / safe_b) * safe_b, 0.0)
        if op == "PINGPONG" and b is not None:
            safe_b = np.where(b != 0.0, b, 1.0)
            return np.where(b != 0.0, np.abs(((a - safe_b) % (2.0 * safe_b)) - safe_b), 0.0)
        if op == "SINE":
            return np.sin(a)
        if op == "COSINE":
            return np.cos(a)
        if op == "TANGENT":
            return np.nan_to_num(np.tan(a), nan=0.0, posinf=0.0, neginf=0.0)
        if op == "ARCSINE":
            return np.where((a >= -1.0) & (a <= 1.0), np.arcsin(np.clip(a, -1.0, 1.0)), 0.0)
        if op == "ARCCOSINE":
            return np.where((a >= -1.0) & (a <= 1.0), np.arccos(np.clip(a, -1.0, 1.0)), 0.0)
        if op == "ARCTANGENT":
            return np.arctan(a)
        if op == "ARCTAN2" and b is not None:
            return np.arctan2(a, b)
        if op == "SINH":
            return np.sinh(np.clip(a, -80.0, 80.0))
        if op == "COSH":
            return np.cosh(np.clip(a, -80.0, 80.0))
        if op == "TANH":
            return np.tanh(a)
        if op == "RADIANS":
            return np.deg2rad(a)
        if op == "DEGREES":
            return np.rad2deg(a)
    return None


def _eval_math_node(
    node: bpy.types.Node,
    group_stack: Tuple[bpy.types.Node, ...],
    visited: Set[Tuple[int, Tuple[int, ...]]],
    pixel_cache: Dict[Any, np.ndarray],
    top_linked: bool,
) -> Optional[SocketEval]:
    op = getattr(node, "operation", "ADD")
    if op in _MATH_1_ARG:
        n_args = 1
    elif op in _MATH_2_ARG:
        n_args = 2
    elif op in _MATH_3_ARG:
        n_args = 3
    else:
        return None

    arg_evals: List[SocketEval] = []
    for idx in range(n_args):
        sock = get_socket(node.inputs, "Value", socket_type="VALUE", occurrence=idx)
        if sock is None:
            return None
        ev = _eval_scalar_recursive(sock, group_stack, visited.copy(), pixel_cache, top_linked)
        if ev is None:
            return None
        arg_evals.append(ev)

    is_const = all(e.samples is None for e in arg_evals)
    _n, arrays, alpha_out = _align_scalar_evals(*arg_evals)
    a_arr = arrays[0]
    b_arr = arrays[1] if len(arrays) > 1 else None
    c_arr = arrays[2] if len(arrays) > 2 else None

    out = _eval_math_op_np(op, a_arr, b_arr, c_arr)
    if out is None:
        return None
    if getattr(node, "use_clamp", False):
        out = np.clip(out, 0.0, 1.0)
    if is_const:
        return _eval_scalar_constant(float(out[0]), top_linked)
    return _eval_scalar_from_samples(out, top_linked, alpha=alpha_out)


def _eval_map_range_node(
    node: bpy.types.Node,
    group_stack: Tuple[bpy.types.Node, ...],
    visited: Set[Tuple[int, Tuple[int, ...]]],
    pixel_cache: Dict[Any, np.ndarray],
    top_linked: bool,
) -> Optional[SocketEval]:
    if getattr(node, "data_type", "FLOAT") != "FLOAT":
        return None
    interp = getattr(node, "interpolation_type", "LINEAR")
    if interp not in {"LINEAR", "STEPPED", "SMOOTHSTEP", "SMOOTHERSTEP"}:
        return None

    val_sock = get_socket(node.inputs, "Value", socket_type="VALUE")
    fmin_sock = get_socket(node.inputs, "From Min", socket_type="VALUE")
    fmax_sock = get_socket(node.inputs, "From Max", socket_type="VALUE")
    tmin_sock = get_socket(node.inputs, "To Min", socket_type="VALUE")
    tmax_sock = get_socket(node.inputs, "To Max", socket_type="VALUE")
    if None in (val_sock, fmin_sock, fmax_sock, tmin_sock, tmax_sock):
        return None

    evals: List[SocketEval] = []
    for sock in (val_sock, fmin_sock, fmax_sock, tmin_sock, tmax_sock):
        ev = _eval_scalar_recursive(sock, group_stack, visited.copy(), pixel_cache, top_linked)
        if ev is None:
            return None
        evals.append(ev)

    if interp == "STEPPED":
        steps_sock = get_socket(node.inputs, "Steps", socket_type="VALUE")
        steps_ev = (
            _eval_scalar_recursive(steps_sock, group_stack, visited.copy(), pixel_cache, top_linked)
            if steps_sock is not None
            else _eval_scalar_constant(4.0, top_linked)
        )
        if steps_ev is None:
            return None
        evals.append(steps_ev)

    is_const = all(e.samples is None for e in evals)
    _n, arrays, alpha_out = _align_scalar_evals(*evals)
    val_arr, fmin_arr, fmax_arr, tmin_arr, tmax_arr = arrays[:5]

    span = fmax_arr - fmin_arr
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(span != 0.0, (val_arr - fmin_arr) / np.where(span != 0.0, span, 1.0), 0.0)

    clamp = bool(getattr(node, "clamp", True))
    if clamp or interp in {"SMOOTHSTEP", "SMOOTHERSTEP"}:
        factor = np.clip(factor, 0.0, 1.0)

    if interp == "STEPPED":
        steps_arr = arrays[5]
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.where(
                steps_arr > 0.0,
                np.floor(factor * (steps_arr + 1.0)) / np.where(steps_arr > 0.0, steps_arr, 1.0),
                0.0,
            )
        if clamp:
            factor = np.clip(factor, 0.0, 1.0)
    elif interp == "SMOOTHSTEP":
        factor = factor * factor * (3.0 - 2.0 * factor)
    elif interp == "SMOOTHERSTEP":
        factor = factor * factor * factor * (factor * (factor * 6.0 - 15.0) + 10.0)

    out = tmin_arr + factor * (tmax_arr - tmin_arr)
    if is_const:
        return _eval_scalar_constant(float(out[0]), top_linked)
    return _eval_scalar_from_samples(out, top_linked, alpha=alpha_out)


def _eval_valtorgb_node(
    node: bpy.types.Node,
    from_socket_name: str,
    group_stack: Tuple[bpy.types.Node, ...],
    visited: Set[Tuple[int, Tuple[int, ...]]],
    pixel_cache: Dict[Any, np.ndarray],
    top_linked: bool,
) -> Optional[SocketEval]:
    ramp = getattr(node, "color_ramp", None)
    if ramp is None or len(getattr(ramp, "elements", ())) == 0:
        return None

    fac_sock = _get_factor_socket(node.inputs)
    fac_eval = (
        _eval_scalar_recursive(fac_sock, group_stack, visited.copy(), pixel_cache, top_linked)
        if fac_sock is not None
        else _eval_scalar_constant(0.5, top_linked)
    )

    if fac_eval is None:
        # Conservative min/max envelope across ramp stops (and spline samples if non-linear RGB)
        elements = ramp.elements
        sample_rgbas = [tuple(float(v) for v in el.color) for el in elements]
        if getattr(ramp, "color_mode", "RGB") != "RGB" or getattr(ramp, "interpolation", "LINEAR") in {
            "B_SPLINE",
            "CARDINAL",
        }:
            for t in np.linspace(0.0, 1.0, 33):
                sample_rgbas.append(tuple(float(v) for v in ramp.evaluate(float(t))))

        if from_socket_name == "Alpha":
            alphas = [c[3] for c in sample_rgbas]
            return SocketEval(
                min_val=min(alphas),
                max_val=max(alphas),
                avg_val=sum(alphas) / len(alphas),
                is_linked=top_linked,
                is_textured=False,
                p01=min(alphas),
                p99=max(alphas),
            )

        lumas = [constants.linear_luminance(c[:3]) for c in sample_rgbas]
        avg_rgb = (
            sum(c[0] for c in sample_rgbas) / len(sample_rgbas),
            sum(c[1] for c in sample_rgbas) / len(sample_rgbas),
            sum(c[2] for c in sample_rgbas) / len(sample_rgbas),
        )
        return SocketEval(
            min_val=min(lumas),
            max_val=max(lumas),
            avg_val=sum(lumas) / len(lumas),
            is_linked=top_linked,
            is_textured=False,
            avg_rgb=avg_rgb,
            samples=None,
            p01=min(lumas),
            p99=max(lumas),
        )

    if fac_eval.samples is None:
        rgba = ramp.evaluate(float(fac_eval.avg_val))
        if from_socket_name == "Alpha":
            return _eval_scalar_constant(float(rgba[3]), top_linked)
        return _eval_color_constant((float(rgba[0]), float(rgba[1]), float(rgba[2])), top_linked)

    fac_arr = fac_eval.samples
    color_mode = getattr(ramp, "color_mode", "RGB")
    interp = getattr(ramp, "interpolation", "LINEAR")

    if color_mode == "RGB" and interp in {"LINEAR", "CONSTANT"}:
        sorted_els = sorted(ramp.elements, key=lambda e: float(e.position))
        pos = np.array([float(e.position) for e in sorted_els], dtype=np.float64)
        cols = np.array([[float(c) for c in e.color] for e in sorted_els], dtype=np.float64)
        if interp == "LINEAR":
            out_rgba = np.column_stack([np.interp(fac_arr, pos, cols[:, ch]) for ch in range(4)])
        else:
            idx = np.clip(np.searchsorted(pos, fac_arr, side="right") - 1, 0, len(pos) - 1)
            out_rgba = cols[idx]
    else:
        out_rgba = np.array([ramp.evaluate(float(t)) for t in fac_arr], dtype=np.float64)

    if from_socket_name == "Alpha":
        return _eval_scalar_from_samples(out_rgba[:, 3], top_linked, alpha=fac_eval.alpha)
    return _eval_color_from_samples(out_rgba[:, :3], top_linked, alpha=fac_eval.alpha)


def _eval_invert_node(
    node: bpy.types.Node,
    group_stack: Tuple[bpy.types.Node, ...],
    visited: Set[Tuple[int, Tuple[int, ...]]],
    pixel_cache: Dict[Any, np.ndarray],
    top_linked: bool,
) -> Optional[SocketEval]:
    fac_sock = _get_factor_socket(node.inputs)
    col_sock = get_socket(node.inputs, "Color", socket_type="RGBA")
    if col_sock is None:
        return None
    col_eval = _eval_color_recursive(col_sock, group_stack, visited.copy(), pixel_cache, top_linked)
    fac_eval = (
        _eval_scalar_recursive(fac_sock, group_stack, visited.copy(), pixel_cache, top_linked)
        if fac_sock is not None
        else _eval_scalar_constant(1.0, top_linked)
    )
    if col_eval is None or fac_eval is None:
        return None

    has_array = col_eval.samples is not None or fac_eval.samples is not None
    if not has_array:
        f = fac_eval.avg_val
        r, g, b = col_eval.avg_rgb
        return _eval_color_constant(
            ((1.0 - f) * r + f * (1.0 - r), (1.0 - f) * g + f * (1.0 - g), (1.0 - f) * b + f * (1.0 - b)),
            top_linked,
        )

    n = max(
        len(col_eval.samples) if col_eval.samples is not None else 1,
        len(fac_eval.samples) if fac_eval.samples is not None else 1,
    )
    col_arr = (
        _resample_array(col_eval.samples, n)
        if col_eval.samples is not None
        else np.broadcast_to(np.array(col_eval.avg_rgb, dtype=np.float64), (n, 3))
    )
    fac_arr = (
        _resample_array(fac_eval.samples, n)[:, None]
        if fac_eval.samples is not None
        else np.full((n, 1), fac_eval.avg_val, dtype=np.float64)
    )
    out = (1.0 - fac_arr) * col_arr + fac_arr * (1.0 - col_arr)
    alpha_out = _resample_array(col_eval.alpha, n) if col_eval.alpha is not None else None
    return _eval_color_from_samples(out, top_linked, alpha=alpha_out)


def _eval_gamma_node(
    node: bpy.types.Node,
    group_stack: Tuple[bpy.types.Node, ...],
    visited: Set[Tuple[int, Tuple[int, ...]]],
    pixel_cache: Dict[Any, np.ndarray],
    top_linked: bool,
) -> Optional[SocketEval]:
    col_sock = get_socket(node.inputs, "Color", socket_type="RGBA")
    gam_sock = get_socket(node.inputs, "Gamma", socket_type="VALUE")
    if col_sock is None or gam_sock is None:
        return None
    col_eval = _eval_color_recursive(col_sock, group_stack, visited.copy(), pixel_cache, top_linked)
    gam_eval = _eval_scalar_recursive(gam_sock, group_stack, visited.copy(), pixel_cache, top_linked)
    if col_eval is None or gam_eval is None:
        return None

    has_array = col_eval.samples is not None or gam_eval.samples is not None
    n = max(
        len(col_eval.samples) if col_eval.samples is not None else 1,
        len(gam_eval.samples) if gam_eval.samples is not None else 1,
    )
    col_arr = (
        _resample_array(col_eval.samples, n)
        if col_eval.samples is not None
        else np.broadcast_to(np.array(col_eval.avg_rgb, dtype=np.float64), (n, 3))
    )
    gam_arr = (
        _resample_array(gam_eval.samples, n)[:, None]
        if gam_eval.samples is not None
        else np.full((n, 1), gam_eval.avg_val, dtype=np.float64)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        pos_col = np.maximum(col_arr, 1e-12)
        out = np.where(col_arr > 0.0, np.power(pos_col, gam_arr), np.where(gam_arr == 0.0, 1.0, 0.0))
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    if not has_array:
        return _eval_color_constant((float(out[0, 0]), float(out[0, 1]), float(out[0, 2])), top_linked)
    alpha_out = _resample_array(col_eval.alpha, n) if col_eval.alpha is not None else None
    return _eval_color_from_samples(out, top_linked, alpha=alpha_out)


def _eval_hue_sat_node(
    node: bpy.types.Node,
    group_stack: Tuple[bpy.types.Node, ...],
    visited: Set[Tuple[int, Tuple[int, ...]]],
    pixel_cache: Dict[Any, np.ndarray],
    top_linked: bool,
) -> Optional[SocketEval]:
    hue_sock = get_socket(node.inputs, "Hue", socket_type="VALUE")
    sat_sock = get_socket(node.inputs, "Saturation", socket_type="VALUE")
    val_sock = get_socket(node.inputs, "Value", socket_type="VALUE")
    fac_sock = _get_factor_socket(node.inputs)
    col_sock = get_socket(node.inputs, "Color", socket_type="RGBA")
    if None in (hue_sock, sat_sock, val_sock, col_sock):
        return None

    col_eval = _eval_color_recursive(col_sock, group_stack, visited.copy(), pixel_cache, top_linked)
    hue_eval = _eval_scalar_recursive(hue_sock, group_stack, visited.copy(), pixel_cache, top_linked)
    sat_eval = _eval_scalar_recursive(sat_sock, group_stack, visited.copy(), pixel_cache, top_linked)
    val_eval = _eval_scalar_recursive(val_sock, group_stack, visited.copy(), pixel_cache, top_linked)
    fac_eval = (
        _eval_scalar_recursive(fac_sock, group_stack, visited.copy(), pixel_cache, top_linked)
        if fac_sock is not None
        else _eval_scalar_constant(1.0, top_linked)
    )
    if None in (col_eval, hue_eval, sat_eval, val_eval, fac_eval):
        return None

    scalar_evals = (hue_eval, sat_eval, val_eval, fac_eval)
    has_array = col_eval.samples is not None or any(e.samples is not None for e in scalar_evals)
    n = max(
        len(col_eval.samples) if col_eval.samples is not None else 1,
        *(len(e.samples) if e.samples is not None else 1 for e in scalar_evals),
    )

    col_arr = (
        _resample_array(col_eval.samples, n)
        if col_eval.samples is not None
        else np.broadcast_to(np.array(col_eval.avg_rgb, dtype=np.float64), (n, 3))
    )
    hue_arr = (
        _resample_array(hue_eval.samples, n)
        if hue_eval.samples is not None
        else np.full(n, hue_eval.avg_val, dtype=np.float64)
    )
    sat_arr = (
        _resample_array(sat_eval.samples, n)
        if sat_eval.samples is not None
        else np.full(n, sat_eval.avg_val, dtype=np.float64)
    )
    val_arr = (
        _resample_array(val_eval.samples, n)
        if val_eval.samples is not None
        else np.full(n, val_eval.avg_val, dtype=np.float64)
    )
    fac_arr = (
        _resample_array(fac_eval.samples, n)[:, None]
        if fac_eval.samples is not None
        else np.full((n, 1), fac_eval.avg_val, dtype=np.float64)
    )

    hsv = _rgb_to_hsv_np(col_arr)
    hsv[:, 0] = (hsv[:, 0] + (hue_arr - 0.5)) % 1.0
    hsv[:, 1] = np.clip(hsv[:, 1] * sat_arr, 0.0, 1.0)
    hsv[:, 2] = np.maximum(hsv[:, 2] * val_arr, 0.0)
    shifted = _hsv_to_rgb_np(hsv)
    out = (1.0 - fac_arr) * col_arr + fac_arr * shifted

    if not has_array:
        return _eval_color_constant((float(out[0, 0]), float(out[0, 1]), float(out[0, 2])), top_linked)
    alpha_out = _resample_array(col_eval.alpha, n) if col_eval.alpha is not None else None
    return _eval_color_from_samples(out, top_linked, alpha=alpha_out)


def _eval_bright_contrast_node(
    node: bpy.types.Node,
    group_stack: Tuple[bpy.types.Node, ...],
    visited: Set[Tuple[int, Tuple[int, ...]]],
    pixel_cache: Dict[Any, np.ndarray],
    top_linked: bool,
) -> Optional[SocketEval]:
    col_sock = get_socket(node.inputs, "Color", socket_type="RGBA")
    bright_sock = (
        get_socket(node.inputs, "Brightness", socket_type="VALUE")
        or get_socket(node.inputs, "Bright", socket_type="VALUE")
    )
    contrast_sock = get_socket(node.inputs, "Contrast", socket_type="VALUE")
    if None in (col_sock, bright_sock, contrast_sock):
        return None

    col_eval = _eval_color_recursive(col_sock, group_stack, visited.copy(), pixel_cache, top_linked)
    bright_eval = _eval_scalar_recursive(bright_sock, group_stack, visited.copy(), pixel_cache, top_linked)
    contrast_eval = _eval_scalar_recursive(contrast_sock, group_stack, visited.copy(), pixel_cache, top_linked)
    if None in (col_eval, bright_eval, contrast_eval):
        return None

    has_array = (
        col_eval.samples is not None
        or bright_eval.samples is not None
        or contrast_eval.samples is not None
    )
    n = max(
        len(col_eval.samples) if col_eval.samples is not None else 1,
        len(bright_eval.samples) if bright_eval.samples is not None else 1,
        len(contrast_eval.samples) if contrast_eval.samples is not None else 1,
    )
    col_arr = (
        _resample_array(col_eval.samples, n)
        if col_eval.samples is not None
        else np.broadcast_to(np.array(col_eval.avg_rgb, dtype=np.float64), (n, 3))
    )
    bright_arr = (
        _resample_array(bright_eval.samples, n)[:, None]
        if bright_eval.samples is not None
        else np.full((n, 1), bright_eval.avg_val, dtype=np.float64)
    )
    contrast_arr = (
        _resample_array(contrast_eval.samples, n)[:, None]
        if contrast_eval.samples is not None
        else np.full((n, 1), contrast_eval.avg_val, dtype=np.float64)
    )
    a = 1.0 + contrast_arr
    b = bright_arr - contrast_arr * 0.5
    out = np.maximum(a * col_arr + b, 0.0)

    if not has_array:
        return _eval_color_constant((float(out[0, 0]), float(out[0, 1]), float(out[0, 2])), top_linked)
    alpha_out = _resample_array(col_eval.alpha, n) if col_eval.alpha is not None else None
    return _eval_color_from_samples(out, top_linked, alpha=alpha_out)


def _eval_clamp_node(
    node: bpy.types.Node,
    group_stack: Tuple[bpy.types.Node, ...],
    visited: Set[Tuple[int, Tuple[int, ...]]],
    pixel_cache: Dict[Any, np.ndarray],
    top_linked: bool,
) -> Optional[SocketEval]:
    val_sock = get_socket(node.inputs, "Value", socket_type="VALUE")
    min_sock = get_socket(node.inputs, "Min", socket_type="VALUE")
    max_sock = get_socket(node.inputs, "Max", socket_type="VALUE")
    if None in (val_sock, min_sock, max_sock):
        return None

    val_eval = _eval_scalar_recursive(val_sock, group_stack, visited.copy(), pixel_cache, top_linked)
    min_eval = _eval_scalar_recursive(min_sock, group_stack, visited.copy(), pixel_cache, top_linked)
    max_eval = _eval_scalar_recursive(max_sock, group_stack, visited.copy(), pixel_cache, top_linked)
    if None in (val_eval, min_eval, max_eval):
        return None

    is_const = all(e.samples is None for e in (val_eval, min_eval, max_eval))
    _n, (val_arr, min_arr, max_arr), alpha_out = _align_scalar_evals(val_eval, min_eval, max_eval)
    clamp_type = getattr(node, "clamp_type", "MINMAX")
    if clamp_type == "RANGE":
        lo = np.minimum(min_arr, max_arr)
        hi = np.maximum(min_arr, max_arr)
        out = np.clip(val_arr, lo, hi)
    else:
        out = np.minimum(np.maximum(val_arr, min_arr), max_arr)

    if is_const:
        return _eval_scalar_constant(float(out[0]), top_linked)
    return _eval_scalar_from_samples(out, top_linked, alpha=alpha_out)


def _eval_rgbtowb_node(
    node: bpy.types.Node,
    group_stack: Tuple[bpy.types.Node, ...],
    visited: Set[Tuple[int, Tuple[int, ...]]],
    pixel_cache: Dict[Any, np.ndarray],
    top_linked: bool,
) -> Optional[SocketEval]:
    col_sock = get_socket(node.inputs, "Color", socket_type="RGBA")
    if col_sock is None:
        return None
    col_eval = _eval_color_recursive(col_sock, group_stack, visited.copy(), pixel_cache, top_linked)
    if col_eval is None:
        return None
    return _color_eval_to_scalar(col_eval, top_linked)


def _eval_color_recursive(
    socket: Optional[bpy.types.NodeSocket],
    group_stack: Tuple[bpy.types.Node, ...],
    visited: Set[Tuple[int, Tuple[int, ...]]],
    pixel_cache: Dict[Any, np.ndarray],
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

    if getattr(from_node, "mute", False):
        pass_sock = _muted_passthrough_input(from_node, from_socket)
        if pass_sock is None:
            return None
        if pass_sock.type == "VALUE":
            s_eval = _eval_scalar_recursive(pass_sock, group_stack, visited, pixel_cache, top_linked)
            return _scalar_eval_to_color(s_eval, top_linked) if s_eval is not None else None
        return _eval_color_recursive(pass_sock, group_stack, visited, pixel_cache, top_linked)

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

    if from_node.type in {"MIX", "MIX_RGB"}:
        if from_node.type == "MIX" and getattr(from_node, "data_type", "FLOAT") != "RGBA":
            if getattr(from_node, "data_type", "FLOAT") == "FLOAT":
                s_eval = _eval_mix_float_node(
                    from_node, group_stack, visited, pixel_cache, top_linked
                )
                return _scalar_eval_to_color(s_eval, top_linked) if s_eval is not None else None
            return None
        return _eval_mix_rgba_node(from_node, group_stack, visited, pixel_cache, top_linked)

    if from_node.type == "VALTORGB":
        if from_socket.name == "Alpha":
            s_eval = _eval_valtorgb_node(
                from_node, "Alpha", group_stack, visited, pixel_cache, top_linked
            )
            return _scalar_eval_to_color(s_eval, top_linked) if s_eval is not None else None
        return _eval_valtorgb_node(from_node, "Color", group_stack, visited, pixel_cache, top_linked)

    if from_node.type == "INVERT":
        return _eval_invert_node(from_node, group_stack, visited, pixel_cache, top_linked)

    if from_node.type == "GAMMA":
        return _eval_gamma_node(from_node, group_stack, visited, pixel_cache, top_linked)

    if from_node.type == "HUE_SAT":
        return _eval_hue_sat_node(from_node, group_stack, visited, pixel_cache, top_linked)

    if from_node.type == "BRIGHTCONTRAST":
        return _eval_bright_contrast_node(from_node, group_stack, visited, pixel_cache, top_linked)

    if from_node.type in {"MATH", "MAP_RANGE", "CLAMP", "RGBTOBW", "SEPARATE_COLOR"}:
        s_eval = _eval_scalar_from_node(
            from_node, from_socket, group_stack, visited, pixel_cache, top_linked
        )
        return _scalar_eval_to_color(s_eval, top_linked) if s_eval is not None else None

    if from_node.type == "TEX_IMAGE":
        samples = _sample_image_rgba(getattr(from_node, "image", None), pixel_cache)
        if samples is None:
            return None
        if from_socket.name == "Alpha":
            alpha_rgb = np.repeat(samples[:, 3:4], 3, axis=1)
            return _eval_color_from_samples(alpha_rgb, top_linked, alpha=None)
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
        if sub_tree is not None and from_node not in group_stack:
            for go_sock in _match_group_output_inputs(sub_tree, from_socket):
                res = _eval_color_recursive(go_sock, group_stack + (from_node,), visited, pixel_cache, top_linked)
                if res is not None:
                    return res
        return None

    return None


def _eval_scalar_from_node(
    from_node: bpy.types.Node,
    from_socket: bpy.types.NodeSocket,
    group_stack: Tuple[bpy.types.Node, ...],
    visited: Set[Tuple[int, Tuple[int, ...]]],
    pixel_cache: Dict[Any, np.ndarray],
    top_linked: bool,
) -> Optional[SocketEval]:
    if from_node.type == "MATH":
        return _eval_math_node(from_node, group_stack, visited, pixel_cache, top_linked)
    if from_node.type == "MAP_RANGE":
        return _eval_map_range_node(from_node, group_stack, visited, pixel_cache, top_linked)
    if from_node.type == "CLAMP":
        return _eval_clamp_node(from_node, group_stack, visited, pixel_cache, top_linked)
    if from_node.type == "RGBTOBW":
        return _eval_rgbtowb_node(from_node, group_stack, visited, pixel_cache, top_linked)
    if from_node.type == "SEPARATE_COLOR":
        in_sock = get_socket(from_node.inputs, "Color") or _first_enabled_socket(from_node.inputs)
        c_eval = _eval_color_recursive(in_sock, group_stack, visited, pixel_cache, top_linked)
        if c_eval is None:
            return None
        ch_map = {"Red": 0, "Green": 1, "Blue": 2}
        ch = ch_map.get(from_socket.name, 0)
        mode = getattr(from_node, "mode", "RGB")
        if c_eval.samples is not None:
            arr = _rgb_to_hsv_np(c_eval.samples) if mode == "HSV" else c_eval.samples
            return _eval_scalar_from_samples(arr[:, ch], top_linked, alpha=c_eval.alpha)
        if mode == "HSV":
            hsv = _rgb_to_hsv_np(np.array([c_eval.avg_rgb], dtype=np.float64))
            return _eval_scalar_constant(float(hsv[0, ch]), top_linked)
        return _eval_scalar_constant(c_eval.avg_rgb[ch], top_linked)
    return None


def _eval_scalar_recursive(
    socket: Optional[bpy.types.NodeSocket],
    group_stack: Tuple[bpy.types.Node, ...],
    visited: Set[Tuple[int, Tuple[int, ...]]],
    pixel_cache: Dict[Any, np.ndarray],
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

    if getattr(from_node, "mute", False):
        pass_sock = _muted_passthrough_input(from_node, from_socket)
        if pass_sock is None:
            return None
        if pass_sock.type == "RGBA":
            c_eval = _eval_color_recursive(pass_sock, group_stack, visited, pixel_cache, top_linked)
            return _color_eval_to_scalar(c_eval, top_linked) if c_eval is not None else None
        return _eval_scalar_recursive(pass_sock, group_stack, visited, pixel_cache, top_linked)

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

    if from_node.type in {"MATH", "MAP_RANGE", "CLAMP", "RGBTOBW", "SEPARATE_COLOR"}:
        return _eval_scalar_from_node(
            from_node, from_socket, group_stack, visited, pixel_cache, top_linked
        )

    if from_node.type == "MIX":
        dt = getattr(from_node, "data_type", "FLOAT")
        if dt == "FLOAT":
            return _eval_mix_float_node(from_node, group_stack, visited, pixel_cache, top_linked)
        if dt == "RGBA":
            c_eval = _eval_mix_rgba_node(from_node, group_stack, visited, pixel_cache, top_linked)
            return _color_eval_to_scalar(c_eval, top_linked) if c_eval is not None else None
        return None

    if from_node.type == "MIX_RGB":
        c_eval = _eval_mix_rgba_node(from_node, group_stack, visited, pixel_cache, top_linked)
        return _color_eval_to_scalar(c_eval, top_linked) if c_eval is not None else None

    if from_node.type == "VALTORGB":
        if from_socket.name == "Alpha":
            return _eval_valtorgb_node(
                from_node, "Alpha", group_stack, visited, pixel_cache, top_linked
            )
        c_eval = _eval_valtorgb_node(
            from_node, "Color", group_stack, visited, pixel_cache, top_linked
        )
        return _color_eval_to_scalar(c_eval, top_linked) if c_eval is not None else None

    if from_node.type in {"COMBINE_COLOR", "INVERT", "GAMMA", "HUE_SAT", "BRIGHTCONTRAST"}:
        if from_node.type == "COMBINE_COLOR":
            c_eval = _combine_color_eval(from_node, group_stack, visited, pixel_cache, top_linked)
        elif from_node.type == "INVERT":
            c_eval = _eval_invert_node(from_node, group_stack, visited, pixel_cache, top_linked)
        elif from_node.type == "GAMMA":
            c_eval = _eval_gamma_node(from_node, group_stack, visited, pixel_cache, top_linked)
        elif from_node.type == "HUE_SAT":
            c_eval = _eval_hue_sat_node(from_node, group_stack, visited, pixel_cache, top_linked)
        else:
            c_eval = _eval_bright_contrast_node(
                from_node, group_stack, visited, pixel_cache, top_linked
            )
        return _color_eval_to_scalar(c_eval, top_linked) if c_eval is not None else None

    if from_node.type == "TEX_IMAGE":
        samples = _sample_image_rgba(getattr(from_node, "image", None), pixel_cache)
        if samples is None:
            return None
        if from_socket.name == "Alpha":
            return _eval_scalar_from_samples(samples[:, 3], top_linked, alpha=None)
        lumas = _luma_array(samples[:, :3])
        return _eval_scalar_from_samples(lumas, top_linked, alpha=samples[:, 3])

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
        if sub_tree is not None and from_node not in group_stack:
            for go_sock in _match_group_output_inputs(sub_tree, from_socket):
                res = _eval_scalar_recursive(go_sock, group_stack + (from_node,), visited, pixel_cache, top_linked)
                if res is not None:
                    return res
        return None

    return None


def evaluate_color_socket(
    socket: Optional[bpy.types.NodeSocket],
    pixel_cache: Optional[Dict[Any, np.ndarray]] = None,
    group_stack: Tuple[bpy.types.Node, ...] = (),
) -> Optional[SocketEval]:
    """Evaluate a color socket to min/max/avg/percentile linear Rec.709 luminance and RGB."""
    if socket is None:
        return None
    cache = pixel_cache if pixel_cache is not None else {}
    return _eval_color_recursive(socket, group_stack, set(), cache, socket.is_linked)


def evaluate_scalar_socket(
    socket: Optional[bpy.types.NodeSocket],
    pixel_cache: Optional[Dict[Any, np.ndarray]] = None,
    group_stack: Tuple[bpy.types.Node, ...] = (),
) -> Optional[SocketEval]:
    """Evaluate a float scalar socket to min/max/avg/percentile value and linkage state."""
    if socket is None:
        return None
    cache = pixel_cache if pixel_cache is not None else {}
    return _eval_scalar_recursive(socket, group_stack, set(), cache, socket.is_linked)


def _outlier_value(eval_res: SocketEval, low: float, high: float) -> float:
    low_val = float(eval_res.p01 if eval_res.p01 is not None else eval_res.min_val)
    high_val = float(eval_res.p99 if eval_res.p99 is not None else eval_res.max_val)
    low_dist = max(0.0, low - low_val)
    high_dist = max(0.0, high_val - high)
    return low_val if low_dist >= high_dist else high_val


def _check_albedo_range(
    low_val: float,
    high_val: float,
    is_conductor: bool,
) -> List[Tuple[str, float]]:
    """Return list of ``(issue_code, measured_value)`` for a luminance range."""
    hits: List[Tuple[str, float]] = []
    if is_conductor:
        if low_val < constants.CONDUCTOR_MIN_LUMA:
            hits.append(("DARK_METAL", float(low_val)))
        return hits

    if low_val < constants.DIELECTRIC_MIN_LUMA:
        hits.append(("DIELECTRIC_CRUSHED_BLACK", float(low_val)))
    if high_val > constants.DIELECTRIC_CRIT_LUMA:
        hits.append(("BLOWN_ALBEDO_CRIT", float(high_val)))
    elif high_val > constants.DIELECTRIC_WARN_LUMA:
        hits.append(("BLOWN_ALBEDO_WARN", float(high_val)))
    return hits


def _evaluate_bsdf_pbr(
    bsdf: bpy.types.Node,
    group_stack: Tuple[bpy.types.Node, ...],
    cache: Dict[Any, np.ndarray],
) -> Tuple[
    List[Tuple[str, str, Category]],
    List[Tuple[str, float]],
    Dict[str, Optional[SocketEval]],
]:
    """Evaluate BSDF sockets once and classify PBR violations and unevaluable linked sockets."""
    base_sock = get_socket(bsdf.inputs, "Base Color")
    met_sock = get_socket(bsdf.inputs, "Metallic")
    rough_sock = get_socket(bsdf.inputs, "Roughness")
    spec_sock = get_socket(bsdf.inputs, "Specular IOR Level")
    ior_sock = get_socket(bsdf.inputs, "IOR")
    trans_sock = get_socket(bsdf.inputs, "Transmission Weight")

    base_eval = evaluate_color_socket(base_sock, cache, group_stack=group_stack)
    met_eval = evaluate_scalar_socket(met_sock, cache, group_stack=group_stack)
    rough_eval = evaluate_scalar_socket(rough_sock, cache, group_stack=group_stack)
    spec_eval = evaluate_scalar_socket(spec_sock, cache, group_stack=group_stack)
    ior_eval = evaluate_scalar_socket(ior_sock, cache, group_stack=group_stack)
    trans_eval = evaluate_scalar_socket(trans_sock, cache, group_stack=group_stack)

    evals: Dict[str, Optional[SocketEval]] = {
        "Base Color": base_eval,
        "Metallic": met_eval,
        "Roughness": rough_eval,
        "Specular IOR Level": spec_eval,
        "IOR": ior_eval,
        "Transmission Weight": trans_eval,
    }

    unevaluable: List[Tuple[str, str, Category]] = []
    for sock_name, sock, ev, cat in (
        ("Base Color", base_sock, base_eval, Category.ALBEDO),
        ("Metallic", met_sock, met_eval, Category.ALBEDO),
        ("Roughness", rough_sock, rough_eval, Category.ROUGHNESS),
        ("Specular IOR Level", spec_sock, spec_eval, Category.ROUGHNESS),
        ("IOR", ior_sock, ior_eval, Category.ROUGHNESS),
    ):
        if sock is not None and sock.is_linked and ev is None:
            unevaluable.append((sock_name, sock_name, cat))

    violations: List[Tuple[str, float]] = []

    if met_eval is not None:
        if (not met_eval.is_linked) and (
            constants.SEMI_METAL_LOW < met_eval.avg_val < constants.SEMI_METAL_HIGH
        ):
            violations.append(("SEMI_METAL", float(met_eval.avg_val)))

    if base_eval is not None:
        if (
            base_eval.samples is not None
            and met_eval is not None
            and met_eval.samples is not None
        ):
            base_lumas = _luma_array(base_eval.samples)
            met_vals = met_eval.samples
            n = max(len(base_lumas), len(met_vals))
            if len(base_lumas) != n:
                base_lumas = _resample_array(base_lumas, n)
            if len(met_vals) != n:
                met_vals = _resample_array(met_vals, n)
            if base_eval.alpha is not None:
                alpha_arr = _resample_array(base_eval.alpha, n)
                if np.any(alpha_arr > 1e-3):
                    opaque = alpha_arr > 1e-3
                    base_lumas = base_lumas[opaque]
                    met_vals = met_vals[opaque]

            diel_mask = met_vals < constants.METALLIC_CONDUCTOR_THRESHOLD
            cond_mask = ~diel_mask
            if np.any(diel_mask):
                d_p01, d_p99 = np.percentile(base_lumas[diel_mask], [1.0, 99.0])
                violations.extend(_check_albedo_range(float(d_p01), float(d_p99), False))
            if np.any(cond_mask):
                c_p01, c_p99 = np.percentile(base_lumas[cond_mask], [1.0, 99.0])
                violations.extend(_check_albedo_range(float(c_p01), float(c_p99), True))
        elif base_eval.samples is None and met_eval is not None and met_eval.samples is not None:
            if np.any(met_eval.samples < constants.METALLIC_CONDUCTOR_THRESHOLD):
                violations.extend(
                    _check_albedo_range(float(base_eval.p01), float(base_eval.p99), False)
                )
            if np.any(met_eval.samples >= constants.METALLIC_CONDUCTOR_THRESHOLD):
                violations.extend(
                    _check_albedo_range(float(base_eval.p01), float(base_eval.p99), True)
                )
        else:
            metallic_val = met_eval.avg_val if met_eval is not None else 0.0
            violations.extend(
                _check_albedo_range(
                    float(base_eval.p01),
                    float(base_eval.p99),
                    metallic_val >= constants.METALLIC_CONDUCTOR_THRESHOLD,
                )
            )

    trans_val = trans_eval.avg_val if trans_eval is not None else 0.0
    metallic_avg = met_eval.avg_val if met_eval is not None else 0.0

    if rough_eval is not None and trans_val < constants.TRANSMISSION_THRESHOLD:
        r_low = float(rough_eval.p01)
        r_high = float(rough_eval.p99)
        if r_low < constants.ROUGHNESS_MIN:
            violations.append(("MIRROR_ROUGHNESS", r_low))
        if r_high > constants.ROUGHNESS_MAX:
            violations.append(("DEAD_ROUGHNESS", r_high))

    if spec_eval is not None:
        s_low = float(spec_eval.p01)
        s_high = float(spec_eval.p99)
        if s_low < constants.SPECULAR_MIN or s_high > constants.SPECULAR_MAX:
            violations.append(
                ("SPECULAR_OUTLIER", _outlier_value(spec_eval, constants.SPECULAR_MIN, constants.SPECULAR_MAX))
            )

    if (
        ior_eval is not None
        and metallic_avg < constants.METALLIC_CONDUCTOR_THRESHOLD
        and trans_val < constants.TRANSMISSION_THRESHOLD
    ):
        i_low = float(ior_eval.p01)
        i_high = float(ior_eval.p99)
        if i_low < constants.IOR_MIN or i_high > constants.IOR_MAX:
            violations.append(
                ("IOR_OUTLIER", _outlier_value(ior_eval, constants.IOR_MIN, constants.IOR_MAX))
            )

    return unevaluable, violations, evals


def bsdf_out_of_spec_sockets(
    mat: Any,
    bsdf: Optional[bpy.types.Node] = None,
    pixel_cache: Optional[Dict[Any, np.ndarray]] = None,
) -> Set[str]:
    """Return subset of Principled BSDF sockets that violate PBR thresholds."""
    if bsdf is None:
        target_bsdf = mat
        target_mat = None
    else:
        target_mat = mat
        target_bsdf = bsdf

    if target_bsdf is None:
        return set()

    cache = pixel_cache if pixel_cache is not None else {}
    group_stack: Tuple[bpy.types.Node, ...] = ()
    if target_mat is not None and getattr(target_mat, "node_tree", None) is not None:
        for node, g_stack in _find_principled_bsdfs_with_context(target_mat.node_tree):
            if node == target_bsdf:
                group_stack = g_stack
                break

    _uneval, violations, evals = _evaluate_bsdf_pbr(target_bsdf, group_stack, cache)
    codes = {code for code, _val in violations}

    out: Set[str] = set()
    if codes & {"DIELECTRIC_CRUSHED_BLACK", "BLOWN_ALBEDO_WARN", "BLOWN_ALBEDO_CRIT", "DARK_METAL"}:
        out.add("Base Color")

    met_eval = evals["Metallic"]
    if "SEMI_METAL" in codes or (
        met_eval is not None and (float(met_eval.p01) < 0.0 or float(met_eval.p99) > 1.0)
    ):
        out.add("Metallic")

    rough_eval = evals["Roughness"]
    trans_eval = evals["Transmission Weight"]
    trans_val = trans_eval.avg_val if trans_eval is not None else 0.0
    if codes & {"MIRROR_ROUGHNESS", "DEAD_ROUGHNESS"} or (
        rough_eval is not None
        and trans_val < constants.TRANSMISSION_THRESHOLD
        and (float(rough_eval.p01) < 0.0 or float(rough_eval.p99) > 1.0)
    ):
        out.add("Roughness")

    if "SPECULAR_OUTLIER" in codes:
        out.add("Specular IOR Level")
    if "IOR_OUTLIER" in codes:
        out.add("IOR")

    return out


def analyze_material(
    mat: Optional[bpy.types.Material],
    object_name: str = "",
    colorspace_map: Optional[Dict[bpy.types.Image, Dict[str, Set[str]]]] = None,
    pixel_cache: Optional[Dict[Any, np.ndarray]] = None,
    local_cs_map: Optional[Dict[bpy.types.Image, Dict[str, Set[str]]]] = None,
) -> List[DiagnosticIssue]:
    """Analyze a single material's active Principled BSDF(s) and colorspaces."""
    if mat is None or mat.node_tree is None:
        return []

    cache = pixel_cache if pixel_cache is not None else {}
    local_cs = local_cs_map if local_cs_map is not None else collect_colorspace_map([mat])
    global_cs = colorspace_map if colorspace_map is not None else local_cs

    issues: List[DiagnosticIssue] = []

    for bsdf, group_stack in _find_principled_bsdfs_with_context(mat.node_tree):
        unevaluable, violations, _evals = _evaluate_bsdf_pbr(bsdf, group_stack, cache)

        for _key, sock_name, cat in unevaluable:
            issues.append(
                DiagnosticIssue(
                    code="UNEVALUABLE_INPUT",
                    category=cat,
                    severity=Severity.INFO,
                    message=f"Material '{mat.name}' {sock_name} uses dynamic nodes; inspect visually",
                    object_name=object_name,
                    material_name=mat.name,
                    fixable=False,
                )
            )

        for code, val in violations:
            if code == "SEMI_METAL":
                issues.append(
                    DiagnosticIssue(
                        code=code,
                        category=Category.ALBEDO,
                        severity=Severity.CRITICAL,
                        message=f"Material '{mat.name}' has unlinked semi-metal Metallic={val:.3f}",
                        object_name=object_name,
                        material_name=mat.name,
                        measured_value=val,
                        expected_range="0.0 or 1.0 (unlinked)",
                        fixable=True,
                    )
                )
            elif code == "DIELECTRIC_CRUSHED_BLACK":
                issues.append(
                    DiagnosticIssue(
                        code=code,
                        category=Category.ALBEDO,
                        severity=Severity.CRITICAL,
                        message=(
                            f"Material '{mat.name}' dielectric Base Color luminance "
                            f"{val:.4f} < {constants.DIELECTRIC_MIN_LUMA}"
                        ),
                        object_name=object_name,
                        material_name=mat.name,
                        measured_value=val,
                        expected_range=f"[{constants.DIELECTRIC_MIN_LUMA}, {constants.DIELECTRIC_WARN_LUMA}]",
                        fixable=True,
                    )
                )
            elif code == "BLOWN_ALBEDO_CRIT":
                issues.append(
                    DiagnosticIssue(
                        code=code,
                        category=Category.ALBEDO,
                        severity=Severity.CRITICAL,
                        message=(
                            f"Material '{mat.name}' dielectric Base Color luminance "
                            f"{val:.4f} > {constants.DIELECTRIC_CRIT_LUMA}"
                        ),
                        object_name=object_name,
                        material_name=mat.name,
                        measured_value=val,
                        expected_range=f"<= {constants.DIELECTRIC_WARN_LUMA}",
                        fixable=True,
                    )
                )
            elif code == "BLOWN_ALBEDO_WARN":
                issues.append(
                    DiagnosticIssue(
                        code=code,
                        category=Category.ALBEDO,
                        severity=Severity.WARNING,
                        message=(
                            f"Material '{mat.name}' dielectric Base Color luminance "
                            f"{val:.4f} > {constants.DIELECTRIC_WARN_LUMA}"
                        ),
                        object_name=object_name,
                        material_name=mat.name,
                        measured_value=val,
                        expected_range=f"<= {constants.DIELECTRIC_WARN_LUMA}",
                        fixable=True,
                    )
                )
            elif code == "DARK_METAL":
                issues.append(
                    DiagnosticIssue(
                        code=code,
                        category=Category.ALBEDO,
                        severity=Severity.CRITICAL,
                        message=(
                            f"Material '{mat.name}' conductor Base Color luminance "
                            f"{val:.4f} < {constants.CONDUCTOR_MIN_LUMA}"
                        ),
                        object_name=object_name,
                        material_name=mat.name,
                        measured_value=val,
                        expected_range=f">= {constants.CONDUCTOR_MIN_LUMA}",
                        fixable=True,
                    )
                )
            elif code == "MIRROR_ROUGHNESS":
                issues.append(
                    DiagnosticIssue(
                        code=code,
                        category=Category.ROUGHNESS,
                        severity=Severity.WARNING,
                        message=f"Material '{mat.name}' roughness {val:.4f} < {constants.ROUGHNESS_MIN}",
                        object_name=object_name,
                        material_name=mat.name,
                        measured_value=val,
                        expected_range=f"[{constants.ROUGHNESS_MIN}, {constants.ROUGHNESS_MAX}]",
                        fixable=True,
                    )
                )
            elif code == "DEAD_ROUGHNESS":
                issues.append(
                    DiagnosticIssue(
                        code=code,
                        category=Category.ROUGHNESS,
                        severity=Severity.WARNING,
                        message=f"Material '{mat.name}' roughness {val:.4f} > {constants.ROUGHNESS_MAX}",
                        object_name=object_name,
                        material_name=mat.name,
                        measured_value=val,
                        expected_range=f"[{constants.ROUGHNESS_MIN}, {constants.ROUGHNESS_MAX}]",
                        fixable=True,
                    )
                )
            elif code == "SPECULAR_OUTLIER":
                issues.append(
                    DiagnosticIssue(
                        code=code,
                        category=Category.ROUGHNESS,
                        severity=Severity.WARNING,
                        message=(
                            f"Material '{mat.name}' Specular IOR Level {val:.3f} "
                            f"outside [{constants.SPECULAR_MIN}, {constants.SPECULAR_MAX}]"
                        ),
                        object_name=object_name,
                        material_name=mat.name,
                        measured_value=val,
                        expected_range=f"[{constants.SPECULAR_MIN}, {constants.SPECULAR_MAX}]",
                        fixable=True,
                    )
                )
            elif code == "IOR_OUTLIER":
                issues.append(
                    DiagnosticIssue(
                        code=code,
                        category=Category.ROUGHNESS,
                        severity=Severity.WARNING,
                        message=(
                            f"Material '{mat.name}' IOR {val:.3f} "
                            f"outside [{constants.IOR_MIN}, {constants.IOR_MAX}]"
                        ),
                        object_name=object_name,
                        material_name=mat.name,
                        measured_value=val,
                        expected_range=f"[{constants.IOR_MIN}, {constants.IOR_MAX}]",
                        fixable=True,
                    )
                )

    for img, local_entry in local_cs.items():
        if not local_entry["data_sockets"]:
            continue
        if not is_valid_data_colorspace(img):
            cs_name = getattr(getattr(img, "colorspace_settings", None), "name", "")
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


def scan_scene_materials_with_stats(
    scene: Optional[bpy.types.Scene],
    selected_only: bool = False,
) -> Tuple[List[DiagnosticIssue], int, int]:
    """Scan all scene materials once and return ``(issues, scanned_mats, scanned_imgs)``."""
    mat_entries = _iter_scene_materials(scene, selected_only=selected_only)
    if not mat_entries:
        return [], 0, 0

    materials = [mat for mat, _obj_name in mat_entries]
    global_cs, per_mat_cs = _collect_colorspace_map_per_material(materials)
    pixel_cache: Dict[Any, np.ndarray] = {}

    deduped: List[DiagnosticIssue] = []
    seen_keys: Set[Tuple[str, str, str, str]] = set()

    for mat, obj_name in mat_entries:
        mat_issues = analyze_material(
            mat,
            object_name=obj_name,
            colorspace_map=global_cs,
            pixel_cache=pixel_cache,
            local_cs_map=per_mat_cs.get(mat.as_pointer(), {}),
        )
        for issue in mat_issues:
            extra = issue.message if issue.code == "UNEVALUABLE_INPUT" else ""
            dedup_key = (issue.code, issue.material_name, issue.image_name, extra)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            deduped.append(issue)

    return deduped, len(mat_entries), len(global_cs)


def scan_scene_materials(
    scene: Optional[bpy.types.Scene],
    selected_only: bool = False,
) -> Tuple[List[DiagnosticIssue], int]:
    """Scan all scene materials for PBR shader and colorspace issues."""
    issues, mat_count, _img_count = scan_scene_materials_with_stats(
        scene, selected_only=selected_only
    )
    return issues, mat_count
