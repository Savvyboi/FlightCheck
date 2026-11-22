"""FlightCheck: Real-Time PBR, Texel Density, and Scale Pre-Flight Suite for Blender 5.0+."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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
_IN_RENDER = False
_SAVED_RENDER_MODE: Optional[int] = None
_SAVED_SAVE_STATE: Optional[Dict[str, Any]] = None


def _remove_handler(handler_list, callback) -> None:
    for existing in list(handler_list):
        if existing is callback:
            handler_list.remove(existing)


def _sync_wm_issues(scene: Optional[bpy.types.Scene], report) -> None:
    context = getattr(bpy, "context", None)
    wm = getattr(context, "window_manager", None) if context is not None else None
    if wm is None:
        return
    wm_state = getattr(wm, "flightcheck_wm", None)
    filter_val = wm_state.active_issue_filter if wm_state is not None else "ALL"
    properties.sync_issue_collection(wm, report, filter_val=filter_val)


def _run_debounced_rescan(scene: Optional[bpy.types.Scene]) -> None:
    """Re-sync active shader overlays, run full scene scan, and redraw 3D Viewports."""
    if _IN_RENDER or state.is_internal_mutation():
        return
    if scene is None:
        context = getattr(bpy, "context", None)
        scene = getattr(context, "scene", None) if context is not None else None
    if scene is None:
        return

    fc = getattr(scene, "flightcheck", None)
    if fc is None or not fc.enabled:
        return

    with state.internal_mutation():
        mode_int = constants.MODE_NAME_TO_INT.get(fc.diagnostic_mode, constants.MODE_OFF)
        if mode_int != constants.MODE_OFF:
            core.shader_splicer.reconcile_scene_splices(
                scene, selected_only=bool(fc.selected_only)
            )
            core.pbr_shader_nodes.set_ctrl_mode(mode_int, create=True)
        else:
            core.pbr_shader_nodes.set_ctrl_mode(constants.MODE_OFF, create=False)
            core.shader_splicer.unsplice_scene_materials(None)

        report = core.run_full_scan(scene, write_texel_attr=True)
        _sync_wm_issues(scene, report)
        hud.sync_draw_handlers(scene)

    hud.tag_view3d_redraw()


@persistent
def _on_load_post(_filepath=None) -> None:
    """Reset caches and defer initial scan/splicing after loading a blend file."""
    global _IN_RENDER, _SAVED_RENDER_MODE, _SAVED_SAVE_STATE
    _IN_RENDER = False
    _SAVED_RENDER_MODE = None
    _SAVED_SAVE_STATE = None

    state.cancel_pending_rescan()
    state.CACHE.clear()
    core.graph_crawler.clear_image_sample_cache()

    context = getattr(bpy, "context", None)
    scene = getattr(context, "scene", None) if context is not None else None
    fc = getattr(scene, "flightcheck", None) if scene is not None else None

    if fc is not None and fc.enabled:
        hud.sync_draw_handlers(scene)
        state.schedule_rescan(scene, delay=0.1)
    else:
        core.pbr_shader_nodes.set_ctrl_mode(constants.MODE_OFF, create=False)
        hud.unregister_all_draw_handlers()

    hud.tag_view3d_redraw()


@persistent
def _on_depsgraph_update_post(scene=None, depsgraph=None) -> None:
    """Mark scene diagnostics dirty and schedule a debounced rescan after relevant edits."""
    global _IN_DEPSGRAPH_HANDLER
    if _IN_DEPSGRAPH_HANDLER or _IN_RENDER or state.is_internal_mutation():
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
        if fc is None or not fc.enabled:
            return

        if depsgraph is not None:
            relevant = any(
                isinstance(getattr(update, "id", None), _RELEVANT_UPDATE_TYPES)
                for update in depsgraph.updates
            )
            if not relevant:
                return

        state.CACHE.mark_dirty(target_scene)
        if fc.auto_rescan:
            state.schedule_rescan(target_scene, delay=0.35)
    finally:
        _IN_DEPSGRAPH_HANDLER = False


@persistent
def _on_render_pre(_scene=None, _depsgraph=None) -> None:
    """Temporarily disable FlightCheck false-color overlays before rendering."""
    global _IN_RENDER, _SAVED_RENDER_MODE
    if _IN_RENDER:
        return
    _IN_RENDER = True
    state.cancel_pending_rescan()
    _SAVED_RENDER_MODE = core.pbr_shader_nodes.get_ctrl_mode()
    with state.internal_mutation():
        core.pbr_shader_nodes.set_ctrl_mode(constants.MODE_OFF, create=False)


@persistent
def _on_render_post(_scene=None, _depsgraph=None) -> None:
    """Restore FlightCheck false-color overlay mode after rendering completes or cancels."""
    global _IN_RENDER, _SAVED_RENDER_MODE
    was_rendering = _IN_RENDER
    saved_mode = _SAVED_RENDER_MODE
    _IN_RENDER = False
    _SAVED_RENDER_MODE = None

    if was_rendering and saved_mode is not None and saved_mode != constants.MODE_OFF:
        with state.internal_mutation():
            core.pbr_shader_nodes.set_ctrl_mode(saved_mode, create=False)
        hud.tag_view3d_redraw()


@persistent
def _on_save_pre(_filepath=None) -> None:
    """Strip diagnostic overlay nodes and temporary texel attributes before saving."""
    global _SAVED_SAVE_STATE
    state.cancel_pending_rescan()
    mode_int = core.pbr_shader_nodes.get_ctrl_mode()
    spliced_mats: List[bpy.types.Material] = [
        mat for mat in bpy.data.materials if core.shader_splicer.is_material_spliced(mat)
    ]

    with state.internal_mutation():
        core.pbr_shader_nodes.set_ctrl_mode(constants.MODE_OFF, create=False)
        core.shader_splicer.unsplice_scene_materials(None)
        removed_attrs = core.texel_analyzer.remove_texel_attributes(None)
        core.pbr_shader_nodes.remove_flightcheck_node_groups(force=False)

    _SAVED_SAVE_STATE = {
        "mode": mode_int,
        "spliced_mats": spliced_mats,
        "had_texel_attrs": removed_attrs > 0,
    }


@persistent
def _on_save_post(_filepath=None) -> None:
    """Restore stripped FlightCheck viewport overlays and texel attributes after saving."""
    global _SAVED_SAVE_STATE
    saved = _SAVED_SAVE_STATE
    _SAVED_SAVE_STATE = None
    if saved is None:
        return

    with state.internal_mutation():
        for mat in saved["spliced_mats"]:
            try:
                core.shader_splicer.splice_material(mat)
            except ReferenceError:
                pass
        if saved["mode"] != constants.MODE_OFF:
            core.pbr_shader_nodes.set_ctrl_mode(saved["mode"], create=True)
        if saved["had_texel_attrs"]:
            context = getattr(bpy, "context", None)
            scene = getattr(context, "scene", None) if context is not None else None
            if scene is not None:
                report = core.run_full_scan(scene, write_texel_attr=True)
                _sync_wm_issues(scene, report)

    hud.tag_view3d_redraw()


@persistent
def _on_undo_redo_post(_scene=None) -> None:
    """Mark diagnostics dirty and schedule resync after undo or redo."""
    core.graph_crawler.clear_image_sample_cache()
    context = getattr(bpy, "context", None)
    scene = getattr(context, "scene", None) if context is not None else None
    if scene is None:
        state.CACHE.mark_dirty(None)
        return

    state.CACHE.mark_dirty(scene)
    fc = getattr(scene, "flightcheck", None)
    if fc is not None and fc.enabled:
        hud.sync_draw_handlers(scene)
        if fc.auto_rescan:
            state.schedule_rescan(scene, delay=0.15)
    else:
        hud.unregister_all_draw_handlers()

    hud.tag_view3d_redraw()


_HANDLER_REGISTRY = (
    ("load_post", _on_load_post),
    ("depsgraph_update_post", _on_depsgraph_update_post),
    ("render_init", _on_render_pre),
    ("render_pre", _on_render_pre),
    ("render_complete", _on_render_post),
    ("render_cancel", _on_render_post),
    ("save_pre", _on_save_pre),
    ("save_post", _on_save_post),
    ("undo_post", _on_undo_redo_post),
    ("redo_post", _on_undo_redo_post),
)


def register() -> None:
    """Register FlightCheck properties, operators, UI panels, and lifecycle handlers."""
    properties.register()
    fixers.register()
    hud.register()

    state.register_rescan_callback(_run_debounced_rescan)
    for attr_name, callback in _HANDLER_REGISTRY:
        handler_list = getattr(bpy.app.handlers, attr_name, None)
        if handler_list is not None:
            _remove_handler(handler_list, callback)
            handler_list.append(callback)


def unregister() -> None:
    """Unregister FlightCheck lifecycle handlers, overlays, UI, fixers, and properties."""
    global _IN_RENDER, _SAVED_RENDER_MODE, _SAVED_SAVE_STATE
    _IN_RENDER = False
    _SAVED_RENDER_MODE = None
    _SAVED_SAVE_STATE = None

    state.cancel_pending_rescan()
    state.register_rescan_callback(None)

    for attr_name, callback in reversed(_HANDLER_REGISTRY):
        handler_list = getattr(bpy.app.handlers, attr_name, None)
        if handler_list is not None:
            _remove_handler(handler_list, callback)

    try:
        with state.internal_mutation():
            core.pbr_shader_nodes.set_ctrl_mode(constants.MODE_OFF, create=False)
            core.shader_splicer.unsplice_scene_materials(None)
            core.pbr_shader_nodes.remove_flightcheck_node_groups(force=False)
    except Exception:
        pass

    hud.unregister_all_draw_handlers()
    core.graph_crawler.clear_image_sample_cache()
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
