"""HUD overlays, 3D bounding cages, and View3D N-Panel registration."""

from __future__ import annotations

from typing import Any

import bpy

from . import overlay_2d, overlay_3d, ui_panel

CLASSES = ui_panel.CLASSES


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


def register() -> None:
    """Register FlightCheck View3D N-Panel UI classes."""
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    """Tear down draw handlers and unregister View3D N-Panel UI classes."""
    unregister_all_draw_handlers()
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
