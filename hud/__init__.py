"""HUD overlays, 3D bounding cages, header popover, pie menu, and N-Panel registration."""

from __future__ import annotations

from typing import Any, List, Tuple

import bpy

from . import overlay_2d, overlay_3d, ui_panel

CLASSES = ui_panel.CLASSES

_ADDON_KEYMAPS: List[Tuple[bpy.types.KeyMap, bpy.types.KeyMapItem]] = []


def tag_view3d_redraw() -> None:
    """Tag all 3D View areas across open windows for redraw."""
    context = getattr(bpy, "context", None)
    if context is None:
        return
    wm = getattr(context, "window_manager", None)
    if wm is None:
        return
    for window in wm.windows:
        screen = getattr(window, "screen", None)
        if screen is None:
            continue
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def sync_draw_handlers(scene: Any) -> None:
    """Synchronize 2D HUD and 3D cage draw handlers with scene settings."""
    settings = getattr(scene, "flightcheck", None) if scene is not None else None
    if settings is not None and settings.enabled:
        if settings.show_hud_2d:
            overlay_2d.register_draw_handler()
        else:
            overlay_2d.unregister_draw_handler()

        if settings.show_overlay_3d:
            overlay_3d.register_draw_handler()
        else:
            overlay_3d.unregister_draw_handler()
    else:
        overlay_2d.unregister_draw_handler()
        overlay_3d.unregister_draw_handler()


def unregister_all_draw_handlers() -> None:
    """Remove all FlightCheck 2D and 3D viewport draw handlers."""
    overlay_2d.unregister_draw_handler()
    overlay_3d.unregister_draw_handler()


def _register_keymaps() -> None:
    context = getattr(bpy, "context", None)
    wm = getattr(context, "window_manager", None) if context is not None else None
    keyconfigs = getattr(wm, "keyconfigs", None) if wm is not None else None
    kc_addon = getattr(keyconfigs, "addon", None) if keyconfigs is not None else None
    if kc_addon is None:
        return

    km = kc_addon.keymaps.new(name="3D View", space_type="VIEW_3D")
    kmi = km.keymap_items.new(
        "wm.call_menu_pie",
        type="F",
        value="PRESS",
        alt=True,
        shift=True,
    )
    kmi.properties.name = "FLIGHTCHECK_MT_pie_menu"
    _ADDON_KEYMAPS.append((km, kmi))


def _unregister_keymaps() -> None:
    for km, kmi in _ADDON_KEYMAPS:
        try:
            km.keymap_items.remove(kmi)
        except (RuntimeError, ReferenceError, ValueError):
            pass
    _ADDON_KEYMAPS.clear()


def register() -> None:
    """Register FlightCheck View3D panels, operators, header status, and pie keymap."""
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    try:
        bpy.types.VIEW3D_HT_header.remove(ui_panel.draw_view3d_header_status)
    except ValueError:
        pass
    bpy.types.VIEW3D_HT_header.append(ui_panel.draw_view3d_header_status)
    _register_keymaps()


def unregister() -> None:
    """Tear down keymaps, header draw hook, draw handlers, and UI classes."""
    ui_panel.cancel_deferred_issue_sync()
    _unregister_keymaps()
    try:
        bpy.types.VIEW3D_HT_header.remove(ui_panel.draw_view3d_header_status)
    except ValueError:
        pass
    unregister_all_draw_handlers()
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
