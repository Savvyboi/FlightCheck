"""Scene and WindowManager PropertyGroups for FlightCheck."""

from __future__ import annotations

from typing import Any, Optional

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)

from . import constants
from .state import (
    CACHE,
    DiagnosticReport,
    Severity,
    cancel_pending_rescan,
    internal_mutation,
    schedule_rescan,
)

_UPDATING = False


def _resolve_scene(self: Any, context: Optional[bpy.types.Context]) -> Optional[bpy.types.Scene]:
    id_data = getattr(self, "id_data", None)
    if isinstance(id_data, bpy.types.Scene):
        return id_data
    return getattr(context, "scene", None) if context is not None else None


def _on_enabled_update(self, context) -> None:
    global _UPDATING
    if _UPDATING:
        return
    _UPDATING = True
    try:
        scene = _resolve_scene(self, context)
        if scene is None:
            return
        from .core import pbr_shader_nodes, run_full_scan, shader_splicer
        from .hud import sync_draw_handlers, tag_view3d_redraw

        with internal_mutation():
            if self.enabled:
                run_full_scan(scene, write_texel_attr=True)
                mode_int = constants.MODE_NAME_TO_INT.get(
                    self.diagnostic_mode, constants.MODE_OFF
                )
                if mode_int != constants.MODE_OFF:
                    shader_splicer.reconcile_scene_splices(
                        scene, selected_only=bool(self.selected_only)
                    )
                    pbr_shader_nodes.set_ctrl_mode(mode_int, create=True)
                else:
                    pbr_shader_nodes.set_ctrl_mode(constants.MODE_OFF, create=False)
                    shader_splicer.unsplice_scene_materials(None)
            else:
                cancel_pending_rescan()
                pbr_shader_nodes.set_ctrl_mode(constants.MODE_OFF, create=False)
                shader_splicer.unsplice_scene_materials(None)

        sync_draw_handlers(scene)
        tag_view3d_redraw()
    finally:
        _UPDATING = False


def _on_mode_update(self, context) -> None:
    global _UPDATING
    if _UPDATING:
        return
    if not self.enabled:
        return
    _UPDATING = True
    try:
        scene = _resolve_scene(self, context)
        if scene is None:
            return
        from .core import pbr_shader_nodes, run_full_scan, shader_splicer
        from .hud import tag_view3d_redraw

        with internal_mutation():
            mode_int = constants.MODE_NAME_TO_INT.get(
                self.diagnostic_mode, constants.MODE_OFF
            )
            if mode_int == constants.MODE_OFF:
                pbr_shader_nodes.set_ctrl_mode(constants.MODE_OFF, create=False)
                shader_splicer.unsplice_scene_materials(None)
            else:
                if CACHE.get_report(scene) is None or CACHE.is_dirty(scene):
                    run_full_scan(scene, write_texel_attr=True)
                shader_splicer.reconcile_scene_splices(
                    scene, selected_only=bool(self.selected_only)
                )
                pbr_shader_nodes.set_ctrl_mode(mode_int, create=True)
        tag_view3d_redraw()
    finally:
        _UPDATING = False


def _on_overlay_toggle_update(self, context) -> None:
    scene = _resolve_scene(self, context)
    if scene is None:
        return
    from .hud import sync_draw_handlers, tag_view3d_redraw

    sync_draw_handlers(scene)
    tag_view3d_redraw()


def _on_scan_param_update(self, context) -> None:
    scene = _resolve_scene(self, context)
    if scene is None:
        return
    CACHE.mark_dirty(scene)
    if self.enabled:
        from .core import shader_splicer
        from .hud import tag_view3d_redraw

        if self.diagnostic_mode != "OFF":
            with internal_mutation():
                shader_splicer.reconcile_scene_splices(
                    scene, selected_only=bool(self.selected_only)
                )
        if self.auto_rescan:
            schedule_rescan(scene, delay=0.35)
        tag_view3d_redraw()


def _on_clamp_knee_update(self, context) -> None:
    global _UPDATING
    if _UPDATING:
        return
    _UPDATING = True
    try:
        if self.clamp_dielectric_knee > 0.79:
            self.clamp_dielectric_knee = 0.79
        if self.clamp_dielectric_max < self.clamp_dielectric_knee + 0.01:
            self.clamp_dielectric_max = min(0.80, self.clamp_dielectric_knee + 0.01)
    finally:
        _UPDATING = False


def _on_clamp_max_update(self, context) -> None:
    global _UPDATING
    if _UPDATING:
        return
    _UPDATING = True
    try:
        if self.clamp_dielectric_max < self.clamp_dielectric_knee + 0.01:
            self.clamp_dielectric_knee = max(0.40, self.clamp_dielectric_max - 0.01)
    finally:
        _UPDATING = False


def _on_filter_update(self, context) -> None:
    wm = getattr(self, "id_data", None)
    if not isinstance(wm, bpy.types.WindowManager):
        wm = getattr(context, "window_manager", None) if context else None
    scene = getattr(context, "scene", None) if context else None
    if wm is not None:
        sync_issue_collection(wm, CACHE.get_report(scene), filter_val=self.active_issue_filter)


def _on_redraw_update(_self, _context) -> None:
    from .hud import tag_view3d_redraw

    tag_view3d_redraw()


class FlightCheckIssueItem(bpy.types.PropertyGroup):
    code: StringProperty(name="Code")
    label: StringProperty(name="Label")
    category: StringProperty(name="Category")
    severity: StringProperty(name="Severity")
    message: StringProperty(name="Message")
    object_name: StringProperty(name="Object")
    material_name: StringProperty(name="Material")
    image_name: StringProperty(name="Image")
    measured_text: StringProperty(name="Measured")
    expected_range: StringProperty(name="Expected")
    fixable: BoolProperty(name="Fixable", default=True)
    is_shared_image: BoolProperty(name="Shared Image", default=False)


def sync_issue_collection(
    wm: Optional[bpy.types.WindowManager],
    report: Optional[DiagnosticReport],
    filter_val: str = "ALL",
) -> None:
    """Populate ``wm.flightcheck_issues`` from ``report`` for UIList display."""
    if wm is None or not hasattr(wm, "flightcheck_issues"):
        return
    col = wm.flightcheck_issues
    col.clear()
    if report is None:
        return

    if filter_val == "ALL":
        filtered = list(report.issues)
    else:
        filtered = [i for i in report.issues if i.category.value == filter_val]

    severity_rank = {
        Severity.CRITICAL: 0,
        Severity.WARNING: 1,
        Severity.INFO: 2,
        Severity.OK: 3,
    }
    ordered = sorted(
        filtered,
        key=lambda i: (
            severity_rank.get(i.severity, 4),
            i.category.value,
            i.code,
            i.object_name,
            i.material_name,
        ),
    )

    for issue in ordered:
        item = col.add()
        item.code = issue.code
        item.label = constants.ISSUE_LABELS.get(issue.code, issue.code)
        item.category = issue.category.value
        item.severity = issue.severity.value
        item.message = issue.message
        item.object_name = issue.object_name
        item.material_name = issue.material_name
        item.image_name = issue.image_name
        item.measured_text = (
            f"{issue.measured_value:.3f}" if issue.measured_value is not None else ""
        )
        item.expected_range = issue.expected_range
        item.fixable = bool(issue.fixable)
        item.is_shared_image = bool(issue.is_shared_image)

    wm_state = getattr(wm, "flightcheck_wm", None)
    if wm_state is not None:
        if len(col) == 0:
            wm_state.active_issue_index = 0
        elif wm_state.active_issue_index >= len(col):
            wm_state.active_issue_index = len(col) - 1


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
    equalize_mode: EnumProperty(
        name="Texel Equalize Mode",
        description="Choose between non-overlapping relative packing or absolute target px/m scaling",
        items=[
            (
                "RELATIVE",
                "Relative (Preserve 0-1 Layout)",
                "Equalize UV islands to object average density and repack cleanly inside [0, 1]",
            ),
            (
                "ABSOLUTE",
                "Absolute (Target px/m)",
                "Scale UV islands to exact Target Texel Density (may tile beyond [0, 1])",
            ),
        ],
        default="RELATIVE",
    )
    auto_rescan: BoolProperty(
        name="Auto Rescan",
        description="Automatically run debounced rescans and overlay resyncs after scene edits",
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
        max=0.79,
        precision=3,
        update=_on_clamp_knee_update,
    )
    clamp_dielectric_max: FloatProperty(
        name="Dielectric Ceiling",
        description="Asymptotic upper linear luminance for clamped dielectrics",
        default=constants.CLAMP_DIELECTRIC_MAX,
        min=0.50,
        max=0.80,
        precision=3,
        update=_on_clamp_max_update,
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
        update=_on_filter_update,
    )
    active_issue_index: IntProperty(
        name="Active Issue Index",
        description="Selected issue index in the Pre-Flight Checklist UIList",
        default=0,
    )
    show_details: BoolProperty(
        name="Show Issue Details",
        description="Display detailed measured values and expected ranges per issue",
        default=True,
    )
    hud_style: EnumProperty(
        name="HUD Style",
        description="Visual density of the 2D Viewport Flight HUD overlay",
        items=[
            ("COMPACT", "Compact Strip", "Single compact horizontal status and gauge bar"),
            ("FULL", "Full Aviation Card", "Full telemetry card with progress bars and scan stats"),
            ("MINIMAL", "Minimal Pill", "Minimal corner GO / CAUTION / NO-GO badge"),
        ],
        default="COMPACT",
        update=_on_redraw_update,
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
    FlightCheckIssueItem,
    FlightCheckSceneSettings,
    FlightCheckWMState,
)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.flightcheck = PointerProperty(type=FlightCheckSceneSettings)
    bpy.types.WindowManager.flightcheck_wm = PointerProperty(type=FlightCheckWMState)
    bpy.types.WindowManager.flightcheck_issues = CollectionProperty(type=FlightCheckIssueItem)


def unregister() -> None:
    if hasattr(bpy.types.WindowManager, "flightcheck_issues"):
        del bpy.types.WindowManager.flightcheck_issues
    if hasattr(bpy.types.WindowManager, "flightcheck_wm"):
        del bpy.types.WindowManager.flightcheck_wm
    if hasattr(bpy.types.Scene, "flightcheck"):
        del bpy.types.Scene.flightcheck
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
