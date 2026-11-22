"""Scene and WindowManager PropertyGroups for FlightCheck."""

from __future__ import annotations

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
)

from . import constants
from .state import CACHE

_UPDATING = False


def _on_enabled_update(self, context) -> None:
    global _UPDATING
    if _UPDATING:
        return
    _UPDATING = True
    try:
        scene = context.scene if context else getattr(self, "id_data", None)
        if scene is None:
            return
        from .core import pbr_shader_nodes, run_full_scan, shader_splicer
        from .hud import sync_draw_handlers, tag_view3d_redraw

        if self.enabled:
            run_full_scan(scene, write_texel_attr=True)
            mode_int = constants.MODE_NAME_TO_INT.get(self.diagnostic_mode, constants.MODE_OFF)
            if mode_int != constants.MODE_OFF:
                shader_splicer.splice_scene_materials(scene, selected_only=self.selected_only)
                pbr_shader_nodes.set_ctrl_mode(mode_int)
        else:
            pbr_shader_nodes.set_ctrl_mode(constants.MODE_OFF)
            shader_splicer.unsplice_scene_materials(scene)

        sync_draw_handlers(scene)
        tag_view3d_redraw()
    finally:
        _UPDATING = False


def _on_mode_update(self, context) -> None:
    global _UPDATING
    if _UPDATING:
        return
    _UPDATING = True
    try:
        scene = context.scene if context else getattr(self, "id_data", None)
        if scene is None:
            return
        from .core import pbr_shader_nodes, run_full_scan, shader_splicer
        from .hud import tag_view3d_redraw

        mode_int = constants.MODE_NAME_TO_INT.get(self.diagnostic_mode, constants.MODE_OFF)
        if mode_int != constants.MODE_OFF:
            if CACHE.get_report() is None or CACHE.is_dirty():
                run_full_scan(scene, write_texel_attr=True)
            shader_splicer.splice_scene_materials(scene, selected_only=self.selected_only)
        pbr_shader_nodes.set_ctrl_mode(mode_int)
        tag_view3d_redraw()
    finally:
        _UPDATING = False


def _on_overlay_toggle_update(self, context) -> None:
    scene = context.scene if context else getattr(self, "id_data", None)
    if scene is None:
        return
    from .hud import sync_draw_handlers, tag_view3d_redraw

    sync_draw_handlers(scene)
    tag_view3d_redraw()


def _on_scan_param_update(self, context) -> None:
    CACHE.mark_dirty()
    scene = context.scene if context else getattr(self, "id_data", None)
    if scene is None:
        return
    if self.enabled and self.auto_rescan:
        from .core import run_full_scan
        from .hud import tag_view3d_redraw

        run_full_scan(scene, write_texel_attr=True)
        tag_view3d_redraw()


def _on_redraw_update(_self, _context) -> None:
    from .hud import tag_view3d_redraw

    tag_view3d_redraw()


class FlightCheckSceneSettings(bpy.types.PropertyGroup):
    enabled: BoolProperty(
        name="Enable FlightCheck",
        description="Activate real-time PBR, texel density, and scale pre-flight diagnostics",
        default=False,
        update=_on_enabled_update,
    )
    diagnostic_mode: EnumProperty(
        name="Diagnostic Mode",
        description="Active false-color viewport diagnostic shader overlay",
        items=constants.MODE_ENUM_ITEMS,
        default="OFF",
        update=_on_mode_update,
    )
    show_hud_2d: BoolProperty(
        name="2D Flight HUD",
        description="Display aviation-style 2D telemetry HUD overlay in the 3D Viewport",
        default=True,
        update=_on_overlay_toggle_update,
    )
    show_overlay_3d: BoolProperty(
        name="3D Warning Cages",
        description="Display 3D wireframe bounding cages on scale and dimension anomalies",
        default=True,
        update=_on_overlay_toggle_update,
    )
    target_texel_density: FloatProperty(
        name="Target Texel Density (px/m)",
        description="Desired texel density in pixels per meter",
        default=constants.TEXEL_DEFAULT_TARGET,
        min=16.0,
        max=16384.0,
        step=100,
        precision=1,
        update=_on_scan_param_update,
    )
    default_texture_size: IntProperty(
        name="Default Texture Size",
        description="Fallback square texture resolution when no base image is connected",
        default=constants.TEXEL_DEFAULT_RES,
        min=64,
        max=16384,
        update=_on_scan_param_update,
    )
    auto_rescan: BoolProperty(
        name="Auto Rescan",
        description="Mark diagnostics dirty on depsgraph changes when FlightCheck is active",
        default=True,
    )
    selected_only: BoolProperty(
        name="Selected Only",
        description="Restrict scanning, splicing, and fixing to currently selected objects",
        default=False,
        update=_on_scan_param_update,
    )
    clamp_dielectric_knee: FloatProperty(
        name="Soft-Knee Start",
        description="Linear luminance where soft-knee dielectric compression begins",
        default=constants.CLAMP_DIELECTRIC_KNEE,
        min=0.40,
        max=0.75,
        precision=3,
    )
    clamp_dielectric_max: FloatProperty(
        name="Dielectric Ceiling",
        description="Asymptotic upper linear luminance for clamped dielectrics",
        default=constants.CLAMP_DIELECTRIC_MAX,
        min=0.50,
        max=0.80,
        precision=3,
    )


class FlightCheckWMState(bpy.types.PropertyGroup):
    active_issue_filter: EnumProperty(
        name="Filter",
        description="Filter diagnostic issues shown in the N-Panel checklist",
        items=[
            ("ALL", "All", "Show all diagnostic issues"),
            ("ALBEDO", "Albedo", "Show albedo and metallic luminance issues"),
            ("ROUGHNESS", "Roughness", "Show roughness, specular, and IOR issues"),
            ("COLORSPACE", "Color Space", "Show data-map sRGB configuration issues"),
            ("TEXEL", "Texel", "Show texel density variance and missing UV issues"),
            ("SCALE", "Scale", "Show transform, scale, and dimension anomalies"),
        ],
        default="ALL",
    )
    show_details: BoolProperty(
        name="Show Issue Details",
        description="Display detailed measured values and expected ranges per issue",
        default=True,
    )
    hud_position: EnumProperty(
        name="HUD Corner",
        description="Corner of the 3D Viewport where the 2D Flight HUD renders",
        items=[
            ("TOP_LEFT", "Top Left", "Render Flight HUD in top-left corner"),
            ("TOP_RIGHT", "Top Right", "Render Flight HUD in top-right corner"),
            ("BOTTOM_LEFT", "Bottom Left", "Render Flight HUD in bottom-left corner"),
            ("BOTTOM_RIGHT", "Bottom Right", "Render Flight HUD in bottom-right corner"),
        ],
        default="TOP_LEFT",
        update=_on_redraw_update,
    )


CLASSES = (
    FlightCheckSceneSettings,
    FlightCheckWMState,
)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.flightcheck = PointerProperty(type=FlightCheckSceneSettings)
    bpy.types.WindowManager.flightcheck_wm = PointerProperty(type=FlightCheckWMState)


def unregister() -> None:
    if hasattr(bpy.types.WindowManager, "flightcheck_wm"):
        del bpy.types.WindowManager.flightcheck_wm
    if hasattr(bpy.types.Scene, "flightcheck"):
        del bpy.types.Scene.flightcheck
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
