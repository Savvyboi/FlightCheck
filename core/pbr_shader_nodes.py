"""Dynamic Blender 5.0+ shader node group builders for FlightCheck diagnostics and clamping."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import bpy

from .. import constants
from ..constants import (
    soft_knee_clamp_color,
    soft_knee_clamp_ior,
    soft_knee_clamp_metallic,
    soft_knee_clamp_roughness,
    soft_knee_clamp_specular,
    soft_knee_lower,
    soft_knee_upper,
)

__all__ = [
    "soft_knee_upper",
    "soft_knee_lower",
    "soft_knee_clamp_metallic",
    "soft_knee_clamp_color",
    "soft_knee_clamp_roughness",
    "soft_knee_clamp_specular",
    "soft_knee_clamp_ior",
    "get_socket",
    "ensure_ctrl_group",
    "set_ctrl_mode",
    "get_ctrl_mode",
    "ensure_overlay_group",
    "ensure_clamp_group",
    "remove_flightcheck_node_groups",
]

_CTRL_VALUE_NODE_NAME = "ModeValue"

_OVERLAY_INPUT_SPECS = (
    ("Shader", "NodeSocketShader", None),
    ("Mode", "NodeSocketFloat", 0.0),
    ("Base Color", "NodeSocketColor", (0.8, 0.8, 0.8, 1.0)),
    ("Metallic", "NodeSocketFloat", 0.0),
    ("Roughness", "NodeSocketFloat", 0.5),
    ("Specular IOR Level", "NodeSocketFloat", 0.5),
    ("IOR", "NodeSocketFloat", 1.5),
    ("Transmission Weight", "NodeSocketFloat", 0.0),
    ("Metallic Linked", "NodeSocketFloat", 0.0),
)

_OVERLAY_OUTPUT_SPECS = (("Shader", "NodeSocketShader", None),)

_CLAMP_INPUT_SPECS = (
    ("Color", "NodeSocketColor", (0.8, 0.8, 0.8, 1.0)),
    ("Metallic", "NodeSocketFloat", 0.0),
    ("Roughness", "NodeSocketFloat", 0.5),
    ("Specular IOR Level", "NodeSocketFloat", 0.5),
    ("IOR", "NodeSocketFloat", 1.5),
    ("Metallic Linked", "NodeSocketFloat", 0.0),
    ("Transmission Weight", "NodeSocketFloat", 0.0),
    ("Dielectric Knee", "NodeSocketFloat", constants.CLAMP_DIELECTRIC_KNEE),
    ("Dielectric Max", "NodeSocketFloat", constants.CLAMP_DIELECTRIC_MAX),
)

_CLAMP_OUTPUT_SPECS = (
    ("Color", "NodeSocketColor", None),
    ("Metallic", "NodeSocketFloat", None),
    ("Roughness", "NodeSocketFloat", None),
    ("Specular IOR Level", "NodeSocketFloat", None),
    ("IOR", "NodeSocketFloat", None),
)


def get_socket(
    sockets: Any,
    name: str,
    socket_type: Optional[str] = None,
    occurrence: int = 0,
) -> Optional[bpy.types.NodeSocket]:
    """Return the ``occurrence``-th enabled socket matching ``name`` (and optional ``socket_type``)."""
    if sockets is None or occurrence < 0:
        return None
    matches = [
        sock
        for sock in sockets
        if sock.name == name
        and getattr(sock, "enabled", True)
        and (socket_type is None or sock.type == socket_type)
    ]
    if occurrence < len(matches):
        return matches[occurrence]
    return None


def _get_factor_socket(sockets: Any) -> Optional[bpy.types.NodeSocket]:
    """Return the Factor/Fac float socket (Blender 5.x uses 'Factor', 4.x uses 'Fac')."""
    return (
        get_socket(sockets, "Factor", socket_type="VALUE")
        or get_socket(sockets, "Fac", socket_type="VALUE")
    )


def _is_node_socket(value: Any) -> bool:
    return isinstance(value, bpy.types.NodeSocket) or hasattr(value, "is_output")


def _connect(
    nt: bpy.types.NodeTree,
    from_socket: Optional[bpy.types.NodeSocket],
    to_socket: Optional[bpy.types.NodeSocket],
) -> None:
    if from_socket is None or to_socket is None:
        raise RuntimeError("Cannot link missing shader socket.")
    nt.links.new(from_socket, to_socket)


def _set_input(
    nt: bpy.types.NodeTree,
    socket: Optional[bpy.types.NodeSocket],
    val_or_socket: Any,
) -> None:
    if socket is None:
        raise RuntimeError("Target input socket does not exist.")
    if val_or_socket is None:
        return
    if _is_node_socket(val_or_socket):
        _connect(nt, val_or_socket, socket)
    else:
        socket.default_value = val_or_socket


def _new_math(
    nt: bpy.types.NodeTree,
    operation: str,
    val0: Any = None,
    val1: Any = None,
    val2: Any = None,
    *,
    use_clamp: bool = False,
    loc: Tuple[float, float] = (0.0, 0.0),
) -> bpy.types.NodeSocket:
    node = nt.nodes.new("ShaderNodeMath")
    node.operation = operation
    node.use_clamp = use_clamp
    node.location = loc
    for idx, val in enumerate((val0, val1, val2)):
        if val is not None:
            sock = get_socket(node.inputs, "Value", socket_type="VALUE", occurrence=idx)
            _set_input(nt, sock, val)
    out = get_socket(node.outputs, "Value", socket_type="VALUE")
    if out is None:
        raise RuntimeError(f"ShaderNodeMath ({operation}) missing Value output.")
    return out


def _new_vec_math(
    nt: bpy.types.NodeTree,
    operation: str,
    vec0: Any = None,
    vec1: Any = None,
    scale: Any = None,
    *,
    loc: Tuple[float, float] = (0.0, 0.0),
) -> bpy.types.NodeSocket:
    node = nt.nodes.new("ShaderNodeVectorMath")
    node.operation = operation
    node.location = loc
    if vec0 is not None:
        _set_input(nt, get_socket(node.inputs, "Vector", socket_type="VECTOR", occurrence=0), vec0)
    if vec1 is not None:
        _set_input(nt, get_socket(node.inputs, "Vector", socket_type="VECTOR", occurrence=1), vec1)
    if scale is not None:
        _set_input(nt, get_socket(node.inputs, "Scale", socket_type="VALUE"), scale)
    out_name = "Value" if operation in {"DOT_PRODUCT", "LENGTH", "DISTANCE"} else "Vector"
    out_type = "VALUE" if out_name == "Value" else "VECTOR"
    out = get_socket(node.outputs, out_name, socket_type=out_type)
    if out is None:
        raise RuntimeError(f"ShaderNodeVectorMath ({operation}) missing {out_name} output.")
    return out


def _new_mix(
    nt: bpy.types.NodeTree,
    data_type: str,
    factor: Any = None,
    a: Any = None,
    b: Any = None,
    *,
    clamp_factor: bool = True,
    clamp_result: bool = False,
    loc: Tuple[float, float] = (0.0, 0.0),
) -> bpy.types.NodeSocket:
    node = nt.nodes.new("ShaderNodeMix")
    node.data_type = data_type
    node.clamp_factor = clamp_factor
    node.clamp_result = clamp_result
    node.location = loc
    sock_type = "RGBA" if data_type == "RGBA" else ("VECTOR" if data_type == "VECTOR" else "VALUE")
    _set_input(nt, _get_factor_socket(node.inputs), factor)
    _set_input(nt, get_socket(node.inputs, "A", socket_type=sock_type), a)
    _set_input(nt, get_socket(node.inputs, "B", socket_type=sock_type), b)
    out = get_socket(node.outputs, "Result", socket_type=sock_type)
    if out is None:
        raise RuntimeError(f"ShaderNodeMix ({data_type}) missing Result output.")
    return out


def _new_combine_color(
    nt: bpy.types.NodeTree,
    r: Any,
    g: Any,
    b: Any,
    *,
    loc: Tuple[float, float] = (0.0, 0.0),
) -> bpy.types.NodeSocket:
    node = nt.nodes.new("ShaderNodeCombineColor")
    node.mode = "RGB"
    node.location = loc
    _set_input(nt, get_socket(node.inputs, "Red"), r)
    _set_input(nt, get_socket(node.inputs, "Green"), g)
    _set_input(nt, get_socket(node.inputs, "Blue"), b)
    out = get_socket(node.outputs, "Color", socket_type="RGBA")
    if out is None:
        raise RuntimeError("ShaderNodeCombineColor missing Color output.")
    return out


def _mask_not(nt: bpy.types.NodeTree, mask: bpy.types.NodeSocket, loc=(0.0, 0.0)) -> bpy.types.NodeSocket:
    return _new_math(nt, "SUBTRACT", 1.0, mask, loc=loc)


def _mask_and(
    nt: bpy.types.NodeTree,
    a: bpy.types.NodeSocket,
    b: bpy.types.NodeSocket,
    loc=(0.0, 0.0),
) -> bpy.types.NodeSocket:
    return _new_math(nt, "MULTIPLY", a, b, loc=loc)


def _mask_or(
    nt: bpy.types.NodeTree,
    a: bpy.types.NodeSocket,
    b: bpy.types.NodeSocket,
    loc=(0.0, 0.0),
) -> bpy.types.NodeSocket:
    return _new_math(nt, "MAXIMUM", a, b, loc=loc)


def _get_or_create_shader_group(name: str) -> bpy.types.ShaderNodeTree:
    for existing in bpy.data.node_groups:
        if (
            existing.name == name
            and getattr(existing, "bl_idname", "") == "ShaderNodeTree"
            and constants.is_editable_id(existing)
        ):
            return existing
    existing = bpy.data.node_groups.get(name)
    if existing is not None and constants.is_editable_id(existing):
        if getattr(existing, "bl_idname", "") == "ShaderNodeTree":
            return existing
        bpy.data.node_groups.remove(existing, do_unlink=True)
    return bpy.data.node_groups.new(name=name, type="ShaderNodeTree")


def _interface_has_sockets(
    ng: bpy.types.NodeTree,
    input_specs: Tuple[Tuple[str, str, Any], ...],
    output_specs: Tuple[Tuple[str, str, Any], ...],
) -> bool:
    existing_in = {
        item.name: item.socket_type
        for item in ng.interface.items_tree
        if getattr(item, "item_type", "") == "SOCKET" and item.in_out == "INPUT"
    }
    existing_out = {
        item.name: item.socket_type
        for item in ng.interface.items_tree
        if getattr(item, "item_type", "") == "SOCKET" and item.in_out == "OUTPUT"
    }
    if len(existing_in) != len(input_specs) or len(existing_out) != len(output_specs):
        return False
    for s_name, s_type, _ in input_specs:
        if existing_in.get(s_name) != s_type:
            return False
    for s_name, s_type, _ in output_specs:
        if existing_out.get(s_name) != s_type:
            return False
    return True


def _populate_interface(
    ng: bpy.types.NodeTree,
    input_specs: Tuple[Tuple[str, str, Any], ...],
    output_specs: Tuple[Tuple[str, str, Any], ...],
) -> None:
    ng.interface.clear()
    for name, socket_type, default_val in input_specs:
        sock = ng.interface.new_socket(name=name, in_out="INPUT", socket_type=socket_type)
        if default_val is not None and hasattr(sock, "default_value"):
            sock.default_value = default_val
    for name, socket_type, default_val in output_specs:
        sock = ng.interface.new_socket(name=name, in_out="OUTPUT", socket_type=socket_type)
        if default_val is not None and hasattr(sock, "default_value"):
            sock.default_value = default_val


def _build_soft_knee_upper(
    nt: bpy.types.NodeTree,
    y_sock: bpy.types.NodeSocket,
    knee: Any,
    ceiling: Any,
    *,
    x: float = 0.0,
    y: float = 0.0,
) -> bpy.types.NodeSocket:
    """Build C^1-smooth tanh upper compression matching ``constants.soft_knee_upper``."""
    ceil_minus_eps = _new_math(nt, "SUBTRACT", ceiling, 1e-4, loc=(x - 240.0, y + 80.0))
    eff_knee = _new_math(nt, "MINIMUM", knee, ceil_minus_eps, loc=(x - 80.0, y + 80.0))
    raw_span = _new_math(nt, "SUBTRACT", ceiling, eff_knee, loc=(x, y))
    span = _new_math(nt, "MAXIMUM", raw_span, 1e-6, loc=(x + 160.0, y))
    delta = _new_math(nt, "SUBTRACT", y_sock, eff_knee, loc=(x, y - 140.0))
    arg = _new_math(nt, "DIVIDE", delta, span, loc=(x + 160.0, y - 140.0))
    tanh_val = _new_math(nt, "TANH", arg, loc=(x + 320.0, y - 140.0))
    compressed = _new_math(nt, "MULTIPLY_ADD", tanh_val, span, eff_knee, loc=(x + 480.0, y - 100.0))
    is_above = _new_math(nt, "GREATER_THAN", y_sock, eff_knee, loc=(x + 320.0, y + 120.0))
    return _new_mix(
        nt,
        "FLOAT",
        factor=is_above,
        a=y_sock,
        b=compressed,
        clamp_factor=True,
        clamp_result=False,
        loc=(x + 660.0, y),
    )


def _build_soft_knee_lower(
    nt: bpy.types.NodeTree,
    y_sock: bpy.types.NodeSocket,
    floor: Any,
    knee: Any,
    *,
    x: float = 0.0,
    y: float = 0.0,
) -> bpy.types.NodeSocket:
    """Build C^1-smooth tanh lower lift matching ``constants.soft_knee_lower``."""
    floor_plus_eps = _new_math(nt, "ADD", floor, 1e-4, loc=(x - 240.0, y + 80.0))
    eff_knee = _new_math(nt, "MAXIMUM", knee, floor_plus_eps, loc=(x - 80.0, y + 80.0))
    raw_span = _new_math(nt, "SUBTRACT", eff_knee, floor, loc=(x, y))
    span = _new_math(nt, "MAXIMUM", raw_span, 1e-6, loc=(x + 160.0, y))
    delta = _new_math(nt, "SUBTRACT", eff_knee, y_sock, loc=(x, y - 140.0))
    arg = _new_math(nt, "DIVIDE", delta, span, loc=(x + 160.0, y - 140.0))
    tanh_val = _new_math(nt, "TANH", arg, loc=(x + 320.0, y - 140.0))
    lift_amt = _new_math(nt, "MULTIPLY", span, tanh_val, loc=(x + 480.0, y - 140.0))
    lifted = _new_math(nt, "SUBTRACT", eff_knee, lift_amt, loc=(x + 640.0, y - 100.0))
    is_below = _new_math(nt, "LESS_THAN", y_sock, eff_knee, loc=(x + 320.0, y + 120.0))
    return _new_mix(
        nt,
        "FLOAT",
        factor=is_below,
        a=y_sock,
        b=lifted,
        clamp_factor=True,
        clamp_result=False,
        loc=(x + 820.0, y),
    )


def ensure_ctrl_group(force_rebuild: bool = False) -> bpy.types.ShaderNodeTree:
    """Create or reuse ``__FLIGHTCHECK_CTRL__`` holding the single O(1) mode switch value."""
    ng = _get_or_create_shader_group(constants.CTRL_GROUP_NAME)
    val_node = ng.nodes.get(_CTRL_VALUE_NODE_NAME)
    out_node = next((n for n in ng.nodes if n.type == "GROUP_OUTPUT"), None)
    if (
        not force_rebuild
        and ng.get("flightcheck_version") == constants.NODE_GROUP_VERSION
        and len(ng.nodes) > 0
        and _interface_has_sockets(ng, (), (("Mode", "NodeSocketFloat", 0.0),))
        and val_node is not None
        and val_node.type == "VALUE"
        and out_node is not None
        and (out_sock := get_socket(out_node.inputs, "Mode", socket_type="VALUE")) is not None
        and out_sock.is_linked
    ):
        return ng

    prev_mode = float(constants.MODE_OFF)
    if not force_rebuild and val_node is not None and val_node.type == "VALUE":
        prev_out = get_socket(val_node.outputs, "Value", socket_type="VALUE")
        if prev_out is not None:
            try:
                prev_mode = float(prev_out.default_value)
            except (TypeError, ValueError):
                prev_mode = float(constants.MODE_OFF)

    ng.nodes.clear()
    _populate_interface(ng, (), (("Mode", "NodeSocketFloat", 0.0),))

    val_node = ng.nodes.new("ShaderNodeValue")
    val_node.name = _CTRL_VALUE_NODE_NAME
    val_node.label = _CTRL_VALUE_NODE_NAME
    val_node.location = (-200.0, 0.0)
    val_out = get_socket(val_node.outputs, "Value", socket_type="VALUE")
    if val_out is not None:
        val_out.default_value = prev_mode

    out_node = ng.nodes.new("NodeGroupOutput")
    out_node.location = (100.0, 0.0)
    _connect(ng, val_out, get_socket(out_node.inputs, "Mode", socket_type="VALUE"))
    ng["flightcheck_version"] = constants.NODE_GROUP_VERSION
    return ng


def set_ctrl_mode(mode_int: int, *, create: Optional[bool] = None) -> None:
    """Update the shared ``__FLIGHTCHECK_CTRL__`` diagnostic mode in O(1) time."""
    mode_val = int(mode_int)
    if create is None:
        create = mode_val != constants.MODE_OFF

    if not create:
        ng = bpy.data.node_groups.get(constants.CTRL_GROUP_NAME)
        if (
            ng is None
            or getattr(ng, "bl_idname", "") != "ShaderNodeTree"
            or not constants.is_editable_id(ng)
        ):
            return
    ng = ensure_ctrl_group()

    val_node = ng.nodes.get(_CTRL_VALUE_NODE_NAME)
    if val_node is None:
        return
    val_out = get_socket(val_node.outputs, "Value", socket_type="VALUE")
    if val_out is not None:
        val_out.default_value = float(mode_val)
    ng.update_tag()


def get_ctrl_mode() -> int:
    """Return the active integer diagnostic mode stored in ``__FLIGHTCHECK_CTRL__``."""
    ng = bpy.data.node_groups.get(constants.CTRL_GROUP_NAME)
    if ng is None or getattr(ng, "bl_idname", "") != "ShaderNodeTree":
        return constants.MODE_OFF
    val_node = ng.nodes.get(_CTRL_VALUE_NODE_NAME)
    if val_node is None:
        return constants.MODE_OFF
    val_out = get_socket(val_node.outputs, "Value", socket_type="VALUE")
    if val_out is None:
        return constants.MODE_OFF
    return int(round(float(val_out.default_value)))


def _build_hatch_color(nt: bpy.types.NodeTree) -> bpy.types.NodeSocket:
    """Create object-space diagonal yellow/red hazard stripe pattern for Specular/IOR outliers."""
    tex_coord = nt.nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-1200.0, -900.0)

    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "DIAGONAL"
    wave.wave_profile = "SAW"
    wave.location = (-980.0, -900.0)
    _connect(nt, get_socket(tex_coord.outputs, "Object", socket_type="VECTOR"), get_socket(wave.inputs, "Vector"))
    _set_input(nt, get_socket(wave.inputs, "Scale"), 8.0)
    _set_input(nt, get_socket(wave.inputs, "Distortion"), 0.0)
    _set_input(nt, get_socket(wave.inputs, "Detail"), 0.0)

    stripe_mask = _new_math(
        nt,
        "GREATER_THAN",
        _get_factor_socket(wave.outputs),
        0.5,
        loc=(-760.0, -900.0),
    )
    return _new_mix(
        nt,
        "RGBA",
        factor=stripe_mask,
        a=constants.COLOR_HATCH_YELLOW,
        b=constants.COLOR_HATCH_RED,
        loc=(-560.0, -900.0),
    )


def _build_texel_ramp(nt: bpy.types.NodeTree) -> Tuple[bpy.types.NodeSocket, bpy.types.NodeSocket, bpy.types.NodeSocket]:
    """Read ``__flightcheck_td`` face-corner attribute and map via 5-stop diagnostic ColorRamp."""
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_type = "GEOMETRY"
    attr.attribute_name = constants.TEXEL_ATTR_NAME
    attr.location = (-1200.0, -1300.0)
    attr_fac = _get_factor_socket(attr.outputs)
    if attr_fac is None:
        raise RuntimeError("ShaderNodeAttribute missing Factor/Fac output.")

    attr_present = _new_math(
        nt,
        "GREATER_THAN",
        attr_fac,
        constants.TEXEL_ATTR_PRESENCE_EPS,
        loc=(-960.0, -1220.0),
    )

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (-960.0, -1420.0)
    ramp.color_ramp.color_mode = "RGB"
    ramp.color_ramp.interpolation = "LINEAR"
    elems = ramp.color_ramp.elements
    elems[0].position = 0.05
    elems[0].color = constants.COLOR_TEXEL_LOW
    elems[1].position = 1.0
    elems[1].color = constants.COLOR_TEXEL_HIGH
    stop_cyan = elems.new(0.25)
    stop_cyan.color = (0.05, 0.72, 0.90, 1.0)
    stop_ok = elems.new(0.50)
    stop_ok.color = constants.COLOR_TEXEL_OK
    stop_orange = elems.new(0.75)
    stop_orange.color = (1.0, 0.62, 0.05, 1.0)

    _connect(nt, attr_fac, _get_factor_socket(ramp.inputs))
    ramp_color = get_socket(ramp.outputs, "Color", socket_type="RGBA")
    if ramp_color is None:
        raise RuntimeError("ShaderNodeValToRGB missing Color output.")

    return attr_fac, attr_present, ramp_color


def ensure_overlay_group(force_rebuild: bool = False) -> bpy.types.ShaderNodeTree:
    """Create or reuse ``__FLIGHTCHECK_OVERLAY__`` false-color diagnostic shader group."""
    ng = _get_or_create_shader_group(constants.OVERLAY_GROUP_NAME)
    if (
        not force_rebuild
        and ng.get("flightcheck_version") == constants.NODE_GROUP_VERSION
        and len(ng.nodes) > 0
        and _interface_has_sockets(ng, _OVERLAY_INPUT_SPECS, _OVERLAY_OUTPUT_SPECS)
    ):
        return ng

    ng.nodes.clear()
    _populate_interface(ng, _OVERLAY_INPUT_SPECS, _OVERLAY_OUTPUT_SPECS)

    gi = ng.nodes.new("NodeGroupInput")
    gi.location = (-1600.0, 0.0)
    go = ng.nodes.new("NodeGroupOutput")
    go.location = (1300.0, 0.0)

    in_shader = get_socket(gi.outputs, "Shader", socket_type="SHADER")
    in_mode = get_socket(gi.outputs, "Mode", socket_type="VALUE")
    in_color = get_socket(gi.outputs, "Base Color", socket_type="RGBA")
    in_metallic = get_socket(gi.outputs, "Metallic", socket_type="VALUE")
    in_roughness = get_socket(gi.outputs, "Roughness", socket_type="VALUE")
    in_specular = get_socket(gi.outputs, "Specular IOR Level", socket_type="VALUE")
    in_ior = get_socket(gi.outputs, "IOR", socket_type="VALUE")
    in_transmission = get_socket(gi.outputs, "Transmission Weight", socket_type="VALUE")
    in_met_linked = get_socket(gi.outputs, "Metallic Linked", socket_type="VALUE")

    # Linear Rec.709 luminance Y and neutral pass-through gray C_pass = 0.25 * Y + 0.05
    luma = _new_vec_math(
        ng,
        "DOT_PRODUCT",
        vec0=in_color,
        vec1=constants.LUMA_WEIGHTS,
        loc=(-1250.0, 420.0),
    )
    gray_scalar = _new_math(ng, "MULTIPLY_ADD", luma, 0.25, 0.05, loc=(-1020.0, 420.0))
    c_pass = _new_combine_color(ng, gray_scalar, gray_scalar, gray_scalar, loc=(-800.0, 420.0))

    # Mode 1: Albedo & Metallic diagnostics
    is_dielectric = _new_math(
        ng, "LESS_THAN", in_metallic, constants.METALLIC_CONDUCTOR_THRESHOLD, loc=(-1250.0, 180.0)
    )
    is_conductor = _mask_not(ng, is_dielectric, loc=(-1060.0, 180.0))

    luma_lt_crushed = _new_math(ng, "LESS_THAN", luma, constants.DIELECTRIC_MIN_LUMA, loc=(-1250.0, 20.0))
    mask_crushed = _mask_and(ng, is_dielectric, luma_lt_crushed, loc=(-1020.0, 20.0))

    luma_gt_warn = _new_math(ng, "GREATER_THAN", luma, constants.DIELECTRIC_WARN_LUMA, loc=(-1250.0, -120.0))
    luma_gt_crit = _new_math(ng, "GREATER_THAN", luma, constants.DIELECTRIC_CRIT_LUMA, loc=(-1250.0, -260.0))
    luma_le_crit = _mask_not(ng, luma_gt_crit, loc=(-1060.0, -200.0))
    mask_blown_warn = _mask_and(
        ng, is_dielectric, _mask_and(ng, luma_gt_warn, luma_le_crit, loc=(-900.0, -140.0)), loc=(-740.0, -140.0)
    )
    mask_blown_crit = _mask_and(ng, is_dielectric, luma_gt_crit, loc=(-900.0, -280.0))

    luma_lt_dark_metal = _new_math(ng, "LESS_THAN", luma, constants.CONDUCTOR_MIN_LUMA, loc=(-1250.0, -400.0))
    mask_dark_metal = _mask_and(ng, is_conductor, luma_lt_dark_metal, loc=(-1020.0, -400.0))

    met_unlinked = _new_math(ng, "LESS_THAN", in_met_linked, 0.5, loc=(-1250.0, -540.0))
    met_gt_low = _new_math(ng, "GREATER_THAN", in_metallic, constants.SEMI_METAL_LOW, loc=(-1250.0, -660.0))
    met_lt_high = _new_math(ng, "LESS_THAN", in_metallic, constants.SEMI_METAL_HIGH, loc=(-1250.0, -780.0))
    mask_semi_metal = _mask_and(
        ng,
        met_unlinked,
        _mask_and(ng, met_gt_low, met_lt_high, loc=(-1020.0, -700.0)),
        loc=(-820.0, -640.0),
    )

    m1_col = _new_mix(ng, "RGBA", factor=mask_blown_warn, a=c_pass, b=constants.COLOR_BLOWN_WARN, loc=(-460.0, 260.0))
    m1_col = _new_mix(ng, "RGBA", factor=mask_crushed, a=m1_col, b=constants.COLOR_CRUSHED_BLACK, loc=(-280.0, 260.0))
    m1_col = _new_mix(ng, "RGBA", factor=mask_dark_metal, a=m1_col, b=constants.COLOR_DARK_METAL, loc=(-100.0, 260.0))
    m1_col = _new_mix(ng, "RGBA", factor=mask_blown_crit, a=m1_col, b=constants.COLOR_BLOWN_CRIT, loc=(80.0, 260.0))
    mode1_color = _new_mix(ng, "RGBA", factor=mask_semi_metal, a=m1_col, b=constants.COLOR_SEMI_METAL, loc=(260.0, 260.0))

    mask_m1_any = _mask_or(
        ng,
        _mask_or(ng, mask_crushed, mask_blown_warn, loc=(-560.0, 60.0)),
        _mask_or(
            ng,
            mask_blown_crit,
            _mask_or(ng, mask_dark_metal, mask_semi_metal, loc=(-560.0, -80.0)),
            loc=(-400.0, -20.0),
        ),
        loc=(-220.0, 20.0),
    )

    # Mode 2: Roughness, Specular IOR Level, and IOR diagnostics
    non_transmissive = _new_math(
        ng, "LESS_THAN", in_transmission, constants.TRANSMISSION_THRESHOLD, loc=(-1250.0, -1040.0)
    )
    rough_lt_min = _new_math(ng, "LESS_THAN", in_roughness, constants.ROUGHNESS_MIN, loc=(-1250.0, -1160.0))
    rough_gt_max = _new_math(ng, "GREATER_THAN", in_roughness, constants.ROUGHNESS_MAX, loc=(-1250.0, -1280.0))
    mask_mirror = _mask_and(ng, non_transmissive, rough_lt_min, loc=(-1020.0, -1120.0))
    mask_dead = _mask_and(ng, non_transmissive, rough_gt_max, loc=(-1020.0, -1260.0))

    spec_low = _new_math(ng, "LESS_THAN", in_specular, constants.SPECULAR_MIN, loc=(-1250.0, -1420.0))
    spec_high = _new_math(ng, "GREATER_THAN", in_specular, constants.SPECULAR_MAX, loc=(-1250.0, -1540.0))
    ior_low = _new_math(ng, "LESS_THAN", in_ior, constants.IOR_MIN, loc=(-1250.0, -1660.0))
    ior_high = _new_math(ng, "GREATER_THAN", in_ior, constants.IOR_MAX, loc=(-1250.0, -1780.0))
    mask_spec = _mask_or(ng, spec_low, spec_high, loc=(-1020.0, -1480.0))
    mask_ior = _mask_or(ng, ior_low, ior_high, loc=(-1020.0, -1720.0))
    mask_spec_ior = _mask_or(ng, mask_spec, mask_ior, loc=(-820.0, -1600.0))

    hatch_color = _build_hatch_color(ng)
    m2_col = _new_mix(ng, "RGBA", factor=mask_mirror, a=c_pass, b=constants.COLOR_MIRROR_ROUGH, loc=(-280.0, -900.0))
    m2_col = _new_mix(ng, "RGBA", factor=mask_dead, a=m2_col, b=constants.COLOR_DEAD_ROUGH, loc=(-100.0, -900.0))
    mode2_color = _new_mix(ng, "RGBA", factor=mask_spec_ior, a=m2_col, b=hatch_color, loc=(80.0, -900.0))

    mask_m2_any = _mask_or(
        ng,
        _mask_or(ng, mask_mirror, mask_dead, loc=(-560.0, -1180.0)),
        mask_spec_ior,
        loc=(-380.0, -1180.0),
    )

    # Mode 3: Texel Density attribute ramp
    attr_fac, attr_present, ramp_color = _build_texel_ramp(ng)
    mode3_color = _new_mix(ng, "RGBA", factor=attr_present, a=c_pass, b=ramp_color, loc=(-560.0, -1340.0))

    # Mode 4: Combined priority composite (Mode 1 > Mode 2 > Mode 3 out-of-spec)
    tex_lt_min = _new_math(ng, "LESS_THAN", attr_fac, 0.25, loc=(-760.0, -1920.0))
    tex_gt_max = _new_math(ng, "GREATER_THAN", attr_fac, 0.75, loc=(-760.0, -2040.0))
    tex_oos = _mask_and(
        ng,
        attr_present,
        _mask_or(ng, tex_lt_min, tex_gt_max, loc=(-560.0, -1980.0)),
        loc=(-380.0, -1980.0),
    )

    m4_col = _new_mix(ng, "RGBA", factor=tex_oos, a=c_pass, b=ramp_color, loc=(80.0, -1400.0))
    m4_col = _new_mix(ng, "RGBA", factor=mask_m2_any, a=m4_col, b=mode2_color, loc=(260.0, -1400.0))
    mode4_color = _new_mix(ng, "RGBA", factor=mask_m1_any, a=m4_col, b=mode1_color, loc=(440.0, -1400.0))

    # Mode selector (1..4) and Mode 0 bypass
    is_m2 = _new_math(ng, "GREATER_THAN", in_mode, 1.5, loc=(440.0, 80.0))
    is_m3 = _new_math(ng, "GREATER_THAN", in_mode, 2.5, loc=(440.0, -60.0))
    is_m4 = _new_math(ng, "GREATER_THAN", in_mode, 3.5, loc=(440.0, -200.0))

    diag_col = _new_mix(ng, "RGBA", factor=is_m2, a=mode1_color, b=mode2_color, loc=(640.0, 120.0))
    diag_col = _new_mix(ng, "RGBA", factor=is_m3, a=diag_col, b=mode3_color, loc=(820.0, 40.0))
    diag_col = _new_mix(ng, "RGBA", factor=is_m4, a=diag_col, b=mode4_color, loc=(1000.0, -40.0))

    emission = ng.nodes.new("ShaderNodeEmission")
    emission.location = (1000.0, -220.0)
    _set_input(ng, get_socket(emission.inputs, "Color", socket_type="RGBA"), diag_col)
    _set_input(ng, get_socket(emission.inputs, "Strength", socket_type="VALUE"), 1.0)

    is_active = _new_math(ng, "GREATER_THAN", in_mode, 0.5, loc=(820.0, 240.0))
    mix_shader = ng.nodes.new("ShaderNodeMixShader")
    mix_shader.location = (1160.0, 0.0)
    _set_input(ng, _get_factor_socket(mix_shader.inputs), is_active)
    _set_input(ng, get_socket(mix_shader.inputs, "Shader", socket_type="SHADER", occurrence=0), in_shader)
    _set_input(
        ng,
        get_socket(mix_shader.inputs, "Shader", socket_type="SHADER", occurrence=1),
        get_socket(emission.outputs, "Emission", socket_type="SHADER"),
    )
    _connect(
        ng,
        get_socket(mix_shader.outputs, "Shader", socket_type="SHADER"),
        get_socket(go.inputs, "Shader", socket_type="SHADER"),
    )
    ng["flightcheck_version"] = constants.NODE_GROUP_VERSION
    return ng


def ensure_clamp_group(force_rebuild: bool = False) -> bpy.types.ShaderNodeTree:
    """Create or reuse ``__FLIGHTCHECK_CLAMP__`` implementing soft-knee PBR remediation."""
    ng = _get_or_create_shader_group(constants.CLAMP_GROUP_NAME)
    if (
        not force_rebuild
        and ng.get("flightcheck_version") == constants.NODE_GROUP_VERSION
        and len(ng.nodes) > 0
        and _interface_has_sockets(ng, _CLAMP_INPUT_SPECS, _CLAMP_OUTPUT_SPECS)
    ):
        return ng

    ng.nodes.clear()
    _populate_interface(ng, _CLAMP_INPUT_SPECS, _CLAMP_OUTPUT_SPECS)

    gi = ng.nodes.new("NodeGroupInput")
    gi.location = (-1600.0, 0.0)
    go = ng.nodes.new("NodeGroupOutput")
    go.location = (1500.0, 0.0)

    in_color = get_socket(gi.outputs, "Color", socket_type="RGBA")
    in_metallic = get_socket(gi.outputs, "Metallic", socket_type="VALUE")
    in_roughness = get_socket(gi.outputs, "Roughness", socket_type="VALUE")
    in_specular = get_socket(gi.outputs, "Specular IOR Level", socket_type="VALUE")
    in_ior = get_socket(gi.outputs, "IOR", socket_type="VALUE")
    in_met_linked = get_socket(gi.outputs, "Metallic Linked", socket_type="VALUE")
    in_transmission = get_socket(gi.outputs, "Transmission Weight", socket_type="VALUE")
    in_diel_knee = get_socket(gi.outputs, "Dielectric Knee", socket_type="VALUE")
    in_diel_max = get_socket(gi.outputs, "Dielectric Max", socket_type="VALUE")

    # 1. Metallic: clamp [0, 1]; snap unlinked semi-metals to 0.0 or 1.0
    met_le1 = _new_math(ng, "MINIMUM", in_metallic, 1.0, loc=(-1300.0, -240.0))
    met_clamped = _new_math(ng, "MAXIMUM", met_le1, 0.0, loc=(-1140.0, -240.0))
    met_unlinked = _new_math(ng, "LESS_THAN", in_met_linked, 0.5, loc=(-1300.0, -380.0))
    met_gt_low = _new_math(ng, "GREATER_THAN", met_clamped, constants.SEMI_METAL_LOW, loc=(-1300.0, -500.0))
    met_lt_high = _new_math(ng, "LESS_THAN", met_clamped, constants.SEMI_METAL_HIGH, loc=(-1300.0, -620.0))
    is_semi = _mask_and(
        ng,
        met_unlinked,
        _mask_and(ng, met_gt_low, met_lt_high, loc=(-1100.0, -540.0)),
        loc=(-920.0, -460.0),
    )
    met_snap = _mask_not(
        ng,
        _new_math(ng, "LESS_THAN", met_clamped, constants.METALLIC_CONDUCTOR_THRESHOLD, loc=(-1100.0, -700.0)),
        loc=(-920.0, -700.0),
    )
    out_metallic = _new_mix(
        ng, "FLOAT", factor=is_semi, a=met_clamped, b=met_snap, clamp_factor=True, clamp_result=False, loc=(-720.0, -340.0)
    )

    # 2. Base Color: clamp channels >= 0, compute Rec.709 luminance Y, apply soft-knee target Y
    sep = ng.nodes.new("ShaderNodeSeparateColor")
    sep.mode = "RGB"
    sep.location = (-1350.0, 420.0)
    _connect(ng, in_color, get_socket(sep.inputs, "Color", socket_type="RGBA"))
    r_in = _new_math(ng, "MAXIMUM", get_socket(sep.outputs, "Red"), 0.0, loc=(-1160.0, 520.0))
    g_in = _new_math(ng, "MAXIMUM", get_socket(sep.outputs, "Green"), 0.0, loc=(-1160.0, 400.0))
    b_in = _new_math(ng, "MAXIMUM", get_socket(sep.outputs, "Blue"), 0.0, loc=(-1160.0, 280.0))
    rgb_clamped = _new_combine_color(ng, r_in, g_in, b_in, loc=(-980.0, 400.0))

    y_luma = _new_vec_math(
        ng, "DOT_PRODUCT", vec0=rgb_clamped, vec1=constants.LUMA_WEIGHTS, loc=(-780.0, 400.0)
    )

    y_cond = _build_soft_knee_lower(
        ng, y_luma, constants.CLAMP_CONDUCTOR_FLOOR, constants.CLAMP_CONDUCTOR_FLOOR_KNEE, x=-560.0, y=580.0
    )
    y_diel_up = _build_soft_knee_upper(ng, y_luma, in_diel_knee, in_diel_max, x=-560.0, y=240.0)
    y_diel = _build_soft_knee_lower(
        ng,
        y_diel_up,
        constants.CLAMP_DIELECTRIC_FLOOR,
        constants.CLAMP_DIELECTRIC_FLOOR_KNEE,
        x=180.0,
        y=240.0,
    )

    is_cond = _mask_not(
        ng,
        _new_math(ng, "LESS_THAN", out_metallic, constants.METALLIC_CONDUCTOR_THRESHOLD, loc=(220.0, 460.0)),
        loc=(420.0, 460.0),
    )
    y_target = _new_mix(
        ng, "FLOAT", factor=is_cond, a=y_diel, b=y_cond, clamp_factor=True, clamp_result=False, loc=(1050.0, 420.0)
    )

    # Hue-preserving scaling with smooth neutral fallback below Y = 1e-3
    y_safe = _new_math(ng, "MAXIMUM", y_luma, 1e-6, loc=(520.0, 680.0))
    scale = _new_math(ng, "DIVIDE", y_target, y_safe, loc=(720.0, 680.0))
    r_scaled = _new_math(ng, "MULTIPLY", r_in, scale, loc=(920.0, 760.0))
    g_scaled = _new_math(ng, "MULTIPLY", g_in, scale, loc=(920.0, 640.0))
    b_scaled = _new_math(ng, "MULTIPLY", b_in, scale, loc=(920.0, 520.0))
    rgb_scaled = _new_combine_color(ng, r_scaled, g_scaled, b_scaled, loc=(1100.0, 640.0))

    gray_target = _new_combine_color(ng, y_target, y_target, y_target, loc=(1100.0, 440.0))
    near_black_t = _new_math(ng, "DIVIDE", y_luma, 1e-3, use_clamp=True, loc=(920.0, 320.0))
    out_color = _new_mix(
        ng,
        "RGBA",
        factor=near_black_t,
        a=gray_target,
        b=rgb_scaled,
        clamp_factor=True,
        clamp_result=False,
        loc=(1300.0, 540.0),
    )

    # 3. Roughness: clamp [0, 1], lift floor & compress ceiling unless transmissive
    rgh_le1 = _new_math(ng, "MINIMUM", in_roughness, 1.0, loc=(-1300.0, -900.0))
    rgh_val = _new_math(ng, "MAXIMUM", rgh_le1, 0.0, loc=(-1140.0, -900.0))
    rgh_low = _build_soft_knee_lower(
        ng, rgh_val, constants.CLAMP_ROUGHNESS_FLOOR, constants.CLAMP_ROUGHNESS_FLOOR_KNEE, x=-940.0, y=-900.0
    )
    rgh_clamped = _build_soft_knee_upper(
        ng, rgh_low, constants.CLAMP_ROUGHNESS_CEIL_KNEE, constants.CLAMP_ROUGHNESS_CEIL, x=-40.0, y=-900.0
    )
    is_trans = _mask_not(
        ng,
        _new_math(ng, "LESS_THAN", in_transmission, constants.TRANSMISSION_THRESHOLD, loc=(420.0, -1060.0)),
        loc=(600.0, -1060.0),
    )
    out_roughness = _new_mix(
        ng, "FLOAT", factor=is_trans, a=rgh_clamped, b=rgh_val, clamp_factor=True, clamp_result=False, loc=(840.0, -920.0)
    )

    # 4. Specular IOR Level: soft-knee into [0.255, 0.745]
    spec_low = _build_soft_knee_lower(
        ng, in_specular, constants.CLAMP_SPECULAR_MIN, constants.CLAMP_SPECULAR_MIN_KNEE, x=-940.0, y=-1320.0
    )
    out_specular = _build_soft_knee_upper(
        ng, spec_low, constants.CLAMP_SPECULAR_MAX_KNEE, constants.CLAMP_SPECULAR_MAX, x=-40.0, y=-1320.0
    )

    # 5. IOR: soft-knee into [1.305, 1.795]
    ior_low = _build_soft_knee_lower(
        ng, in_ior, constants.CLAMP_IOR_MIN, constants.CLAMP_IOR_MIN_KNEE, x=-940.0, y=-1740.0
    )
    out_ior = _build_soft_knee_upper(
        ng, ior_low, constants.CLAMP_IOR_MAX_KNEE, constants.CLAMP_IOR_MAX, x=-40.0, y=-1740.0
    )

    _connect(ng, out_color, get_socket(go.inputs, "Color", socket_type="RGBA"))
    _connect(ng, out_metallic, get_socket(go.inputs, "Metallic", socket_type="VALUE"))
    _connect(ng, out_roughness, get_socket(go.inputs, "Roughness", socket_type="VALUE"))
    _connect(ng, out_specular, get_socket(go.inputs, "Specular IOR Level", socket_type="VALUE"))
    _connect(ng, out_ior, get_socket(go.inputs, "IOR", socket_type="VALUE"))
    ng["flightcheck_version"] = constants.NODE_GROUP_VERSION
    return ng


def remove_flightcheck_node_groups(force: bool = False) -> int:
    """Remove FlightCheck node groups with zero users (or all when ``force=True``)."""
    known_names = {
        constants.CTRL_GROUP_NAME,
        constants.OVERLAY_GROUP_NAME,
        constants.CLAMP_GROUP_NAME,
    }
    removed = 0
    for ng in list(bpy.data.node_groups):
        if not constants.is_editable_id(ng):
            continue
        if ng.name not in known_names and not ng.name.startswith(constants.PREFIX):
            continue
        if force:
            ng.use_fake_user = False
        if force or ng.users == 0:
            bpy.data.node_groups.remove(ng, do_unlink=True)
            removed += 1
    return removed
