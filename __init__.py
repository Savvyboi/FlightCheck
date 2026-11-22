"""FlightCheck: Real-Time PBR, Texel Density, and Scale Pre-Flight Suite for Blender 5.0+."""

from __future__ import annotations

import bpy
from bpy.app.handlers import persistent

from . import constants, core, fixers, hud, properties, state

_RELEVANT_UPDATE_TYPES = (
    bpy.types.Object,
    bpy.types.Mesh,
    bpy.types.Material,
    bpy.types.Image,
    bpy.types.NodeTree,
)

_IN_DEPSGRAPH_HANDLER = False


def _remove_handler(handler_list, callback) -> None:
    for existing in list(handler_list):
        if existing is callback:
            handler_list.remove(existing)


@persistent
def _on_load_post(_dummy=None) -> None:
    """Reset cached reports and synchronize viewport overlays after loading a blend file."""
    state.CACHE.clear()
    context = getattr(bpy, "context", None)
    scene = getattr(context, "scene", None) if context is not None else None
    fc = getattr(scene, "flightcheck", None) if scene is not None else None

    if fc is not None and fc.enabled:
        core.run_full_scan(scene, write_texel_attr=True)
        mode_int = constants.MODE_NAME_TO_INT.get(fc.diagnostic_mode, constants.MODE_OFF)
        if mode_int != constants.MODE_OFF:
            core.shader_splicer.splice_scene_materials(scene, selected_only=bool(fc.selected_only))
        core.pbr_shader_nodes.set_ctrl_mode(mode_int)
        hud.sync_draw_handlers(scene)
    else:
        core.pbr_shader_nodes.set_ctrl_mode(constants.MODE_OFF)
        hud.unregister_all_draw_handlers()

    hud.tag_view3d_redraw()


@persistent
def _on_depsgraph_update_post(scene, depsgraph=None) -> None:
    """Mark diagnostics dirty when relevant scene datablocks change while FlightCheck is active."""
    global _IN_DEPSGRAPH_HANDLER
    if _IN_DEPSGRAPH_HANDLER:
        return
    _IN_DEPSGRAPH_HANDLER = True
    try:
        target_scene = scene
        if target_scene is None:
            context = getattr(bpy, "context", None)
            target_scene = getattr(context, "scene", None) if context is not None else None
        if target_scene is None:
            return

        fc = getattr(target_scene, "flightcheck", None)
        if fc is None or not fc.enabled or not fc.auto_rescan:
            return

        if depsgraph is not None:
            relevant = any(
                isinstance(getattr(update, "id", None), _RELEVANT_UPDATE_TYPES)
                for update in depsgraph.updates
            )
            if not relevant:
                return

        state.CACHE.mark_dirty()
    finally:
        _IN_DEPSGRAPH_HANDLER = False


def register() -> None:
    """Register FlightCheck properties, operators, UI panels, and lifecycle handlers."""
    properties.register()
    fixers.register()
    hud.register()

    _remove_handler(bpy.app.handlers.load_post, _on_load_post)
    _remove_handler(bpy.app.handlers.depsgraph_update_post, _on_depsgraph_update_post)
    bpy.app.handlers.load_post.append(_on_load_post)
    bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update_post)


def unregister() -> None:
    """Unregister FlightCheck lifecycle handlers, overlays, UI, fixers, and properties."""
    _remove_handler(bpy.app.handlers.depsgraph_update_post, _on_depsgraph_update_post)
    _remove_handler(bpy.app.handlers.load_post, _on_load_post)

    hud.unregister_all_draw_handlers()
    state.CACHE.clear()

    hud.unregister()
    fixers.unregister()
    properties.unregister()


__all__ = [
    "constants",
    "core",
    "fixers",
    "hud",
    "properties",
    "state",
    "register",
    "unregister",
]
