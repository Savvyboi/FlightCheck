"""View3D N-Panel UI, header status popover, radial pie menu, and issue checklist."""

from __future__ import annotations

from typing import List, Optional

import bpy
from bpy.props import EnumProperty, IntProperty, StringProperty

from .. import constants
from ..properties import sync_issue_collection
from ..state import CACHE, DiagnosticIssue, FlightStatus, Severity

_MAX_DISPLAYED_ISSUES = 80
_SOLID_SHADING_TYPES = frozenset({"SOLID", "WIREFRAME"})
_DEFERRED_SYNC_REGISTERED = False


def _severity_icon(severity: Severity) -> str:
    """Map DiagnosticIssue severity to a standard Blender UI icon."""
    if severity == Severity.CRITICAL:
        return "ERROR"
    if severity == Severity.WARNING:
        return "INFO"
    if severity == Severity.INFO:
        return "QUESTION"
    return "CHECKMARK"


def _severity_icon_from_name(severity_name: str) -> str:
    """Map a serialized Severity string to a standard Blender UI icon."""
    if severity_name == Severity.CRITICAL.value:
        return "ERROR"
    if severity_name == Severity.WARNING.value:
        return "INFO"
    if severity_name == Severity.INFO.value:
        return "QUESTION"
    return "CHECKMARK"


def _status_icon(status: FlightStatus) -> str:
    """Map FlightStatus to a standard Blender UI icon."""
    if status == FlightStatus.GO:
        return "CHECKMARK"
    if status == FlightStatus.CAUTION:
        return "INFO"
    return "ERROR"


def _is_solid_shading(context: bpy.types.Context) -> bool:
    """Return True when the active 3D Viewport is in Solid or Wireframe shading."""
    space = getattr(context, "space_data", None)
    shading = getattr(space, "shading", None) if space is not None else None
    return getattr(shading, "type", "") in _SOLID_SHADING_TYPES


def _format_texel_preset_label(density: int) -> str:
    if density >= 1024 and density % 1024 == 0:
        return f"{density // 1024}K"
    return str(density)


def _tag_view3d_redraw_safe() -> None:
    context = getattr(bpy, "context", None)
    wm = getattr(context, "window_manager", None) if context is not None else None
    if wm is None:
        return
    for window in wm.windows:
        screen = getattr(window, "screen", None)
        if screen is None:
            continue
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _sync_issues_from_context(context: Optional[bpy.types.Context]) -> None:
    if context is None:
        return
    wm = getattr(context, "window_manager", None)
    scene = getattr(context, "scene", None)
    wm_state = getattr(wm, "flightcheck_wm", None) if wm is not None else None
    filter_val = wm_state.active_issue_filter if wm_state is not None else "ALL"
    if wm is not None:
        sync_issue_collection(wm, CACHE.get_report(scene), filter_val=filter_val)


def _deferred_issue_sync_timer() -> Optional[float]:
    global _DEFERRED_SYNC_REGISTERED
    _DEFERRED_SYNC_REGISTERED = False
    context = getattr(bpy, "context", None)
    _sync_issues_from_context(context)
    _tag_view3d_redraw_safe()
    return None


def _schedule_deferred_issue_sync() -> None:
    """Schedule a one-shot outside-draw sync of ``wm.flightcheck_issues``."""
    global _DEFERRED_SYNC_REGISTERED
    if bpy.app.background or _DEFERRED_SYNC_REGISTERED:
        return
    timers = getattr(bpy.app, "timers", None)
    if timers is None:
        return
    _DEFERRED_SYNC_REGISTERED = True
    try:
        if timers.is_registered(_deferred_issue_sync_timer):
            timers.unregister(_deferred_issue_sync_timer)
        timers.register(_deferred_issue_sync_timer, first_interval=0.01, persistent=False)
    except Exception:
        _DEFERRED_SYNC_REGISTERED = False


def cancel_deferred_issue_sync() -> None:
    """Cancel any pending UI checklist sync timer on teardown."""
    global _DEFERRED_SYNC_REGISTERED
    _DEFERRED_SYNC_REGISTERED = False
    try:
        timers = getattr(bpy.app, "timers", None)
        if timers is not None and timers.is_registered(_deferred_issue_sync_timer):
            timers.unregister(_deferred_issue_sync_timer)
    except Exception:
        pass


def _order_filtered_issues(
    issues: List[DiagnosticIssue],
    filter_val: str,
) -> List[DiagnosticIssue]:
    if filter_val == "ALL":
        filtered = list(issues)
    else:
        filtered = [issue for issue in issues if issue.category.value == filter_val]

    severity_rank = {
        Severity.CRITICAL: 0,
        Severity.WARNING: 1,
        Severity.INFO: 2,
        Severity.OK: 3,
    }
    return sorted(
        filtered,
        key=lambda i: (
            severity_rank.get(i.severity, 4),
            i.category.value,
            i.code,
            i.object_name,
            i.material_name,
        ),
    )


def _is_collection_synced(
    col: bpy.types.bpy_prop_collection,
    ordered: List[DiagnosticIssue],
) -> bool:
    if len(col) != len(ordered):
        return False
    for item, issue in zip(col, ordered):
        if (
            item.code != issue.code
            or item.severity != issue.severity.value
            or item.object_name != issue.object_name
            or item.material_name != issue.material_name
            or item.image_name != issue.image_name
            or item.message != issue.message
        ):
            return False
    return True


def _resolve_target_object(
    scene: Optional[bpy.types.Scene],
    object_name: str,
    material_name: str,
) -> Optional[bpy.types.Object]:
    if scene is None:
        return None
    if object_name:
        obj = scene.objects.get(object_name)
        if obj is not None:
            return obj
    if material_name:
        for obj in scene.objects:
            if not hasattr(obj, "material_slots"):
                continue
            for slot in obj.material_slots:
                mat = slot.material
                if mat is not None and mat.name == material_name:
                    return obj
    return None


class FLIGHTCHECK_OT_select_issue_target(bpy.types.Operator):
    """Select the target object for a diagnostic issue and frame it in the viewport."""

    bl_idname = "flightcheck.select_issue_target"
    bl_label = "Select Issue Target"
    bl_description = "Select the affected object in Object Mode and frame it in the 3D Viewport"
    bl_options = {"REGISTER", "UNDO"}

    object_name: StringProperty(name="Object Name", default="")
    material_name: StringProperty(name="Material Name", default="")

    def execute(self, context: bpy.types.Context) -> set[str]:
        scene = getattr(context, "scene", None)
        target = _resolve_target_object(scene, self.object_name, self.material_name)
        if target is None:
            if self.object_name:
                self.report({"INFO"}, f"Object '{self.object_name}' was not found in active scene.")
            elif self.material_name:
                self.report({"INFO"}, f"No scene object currently uses material '{self.material_name}'.")
            else:
                self.report({"INFO"}, "This diagnostic issue has no selectable object target.")
            return {"CANCELLED"}

        if context.mode != "OBJECT":
            if bpy.ops.object.mode_set.poll():
                try:
                    bpy.ops.object.mode_set(mode="OBJECT")
                except RuntimeError:
                    self.report({"WARNING"}, "Switch to Object Mode to select target object.")
                    return {"CANCELLED"}
            else:
                self.report({"WARNING"}, "Switch to Object Mode to select target object.")
                return {"CANCELLED"}

        view_layer = getattr(context, "view_layer", None)
        if view_layer is None or target.name not in view_layer.objects:
            self.report({"INFO"}, f"Object '{target.name}' is not in the active view layer.")
            return {"CANCELLED"}

        for obj in view_layer.objects:
            try:
                obj.select_set(False)
            except RuntimeError:
                pass

        try:
            if target.hide_get(view_layer=view_layer):
                target.hide_set(False, view_layer=view_layer)
            target.select_set(True)
            view_layer.objects.active = target
        except RuntimeError as exc:
            self.report({"WARNING"}, f"Could not select '{target.name}': {exc}")
            return {"CANCELLED"}

        # N-Panel invocations run with region='UI', so override with region='WINDOW' to frame view.
        area = getattr(context, "area", None)
        if area is None or area.type != "VIEW_3D":
            screen = getattr(context, "screen", None)
            if screen is not None:
                area = next((a for a in screen.areas if a.type == "VIEW_3D"), None)

        if area is not None and area.type == "VIEW_3D":
            win_region = next((r for r in area.regions if r.type == "WINDOW"), None)
            if win_region is not None:
                try:
                    with context.temp_override(
                        window=getattr(context, "window", None),
                        area=area,
                        region=win_region,
                    ):
                        if bpy.ops.view3d.view_selected.poll():
                            bpy.ops.view3d.view_selected(use_all_regions=False)
                except RuntimeError:
                    pass

        _tag_view3d_redraw_safe()
        return {"FINISHED"}


class FLIGHTCHECK_OT_set_texel_preset(bpy.types.Operator):
    """Apply a preset Target Texel Density value (px/m)."""

    bl_idname = "flightcheck.set_texel_preset"
    bl_label = "Set Texel Density Preset"
    bl_description = "Set Target Texel Density to a standard power-of-two px/m preset"
    bl_options = {"REGISTER", "UNDO"}

    density: IntProperty(name="Texel Density", default=1024, min=16, max=16384)

    def execute(self, context: bpy.types.Context) -> set[str]:
        settings = getattr(getattr(context, "scene", None), "flightcheck", None)
        if settings is None:
            return {"CANCELLED"}
        settings.target_texel_density = float(self.density)
        _tag_view3d_redraw_safe()
        return {"FINISHED"}


class FLIGHTCHECK_OT_set_mode(bpy.types.Operator):
    """Switch the active FlightCheck diagnostic viewport shader mode."""

    bl_idname = "flightcheck.set_mode"
    bl_label = "Set Diagnostic Mode"
    bl_description = "Switch FlightCheck false-color viewport overlay mode"
    bl_options = {"REGISTER", "UNDO"}

    mode: EnumProperty(
        name="Diagnostic Mode",
        items=constants.MODE_ENUM_ITEMS,
        default="OFF",
    )

    def execute(self, context: bpy.types.Context) -> set[str]:
        settings = getattr(getattr(context, "scene", None), "flightcheck", None)
        if settings is None:
            return {"CANCELLED"}
        if self.mode != "OFF" and not settings.enabled:
            settings.enabled = True
        settings.diagnostic_mode = self.mode
        _sync_issues_from_context(context)
        _tag_view3d_redraw_safe()
        return {"FINISHED"}


class FLIGHTCHECK_UL_issues(bpy.types.UIList):
    """Scrollable checklist rows for FlightCheck diagnostic issues."""

    bl_idname = "FLIGHTCHECK_UL_issues"

    def draw_item(
        self,
        context: bpy.types.Context,
        layout: bpy.types.UILayout,
        data: bpy.types.AnyType,
        item: bpy.types.AnyType,
        icon: int,
        active_data: bpy.types.AnyType,
        active_propname: str,
        index: int = 0,
    ) -> None:
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row(align=True)
            row.alert = item.severity == Severity.CRITICAL.value

            targets = [t for t in (item.object_name, item.material_name, item.image_name) if t]
            target_str = " / ".join(targets)
            label_text = item.label or item.code
            display_text = f"{label_text}  ({target_str})" if target_str else label_text
            row.label(text=display_text, icon=_severity_icon_from_name(item.severity))

            if item.is_shared_image:
                row.label(text="", icon="LOCKED")

            sel_row = row.row(align=True)
            sel_row.enabled = bool(item.object_name or item.material_name)
            op = sel_row.operator(
                "flightcheck.select_issue_target",
                text="",
                icon="RESTRICT_SELECT_OFF",
            )
            op.object_name = item.object_name
            op.material_name = item.material_name
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text="", icon=_severity_icon_from_name(item.severity))

    def filter_items(self, context: bpy.types.Context, data: bpy.types.AnyType, propname: str):
        items = getattr(data, propname)
        helper_funcs = bpy.types.UI_UL_list
        flt_flags: List[int] = []
        flt_neworder: List[int] = []

        if self.filter_name:
            flt_flags = [0] * len(items)
            for field_name in ("label", "object_name", "material_name", "code", "image_name"):
                helper_funcs.filter_items_by_name(
                    self.filter_name,
                    self.bitflag_filter_item,
                    items,
                    propname=field_name,
                    flags=flt_flags,
                )

        if self.use_filter_sort_alpha:
            flt_neworder = helper_funcs.sort_items_by_name(items, "label")

        return flt_flags, flt_neworder


class _FlightCheckPanelBase(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "FlightCheck"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return context.scene is not None and hasattr(context.scene, "flightcheck")


def _draw_status_banner(
    layout: bpy.types.UILayout,
    context: bpy.types.Context,
    *,
    compact: bool = False,
) -> None:
    scene = context.scene
    report = CACHE.get_report(scene)
    is_stale = bool(report is not None and CACHE.is_dirty(scene))
    draw_ms = CACHE.get_draw_time_ms()

    banner = layout.box()
    if report is None:
        banner.label(text="STATUS: READY (Not Scanned)", icon="INFO")
        if not compact:
            banner.label(text=f"HUD Draw: {draw_ms:.2f} ms")
        return

    status = report.status
    row = banner.row()
    row.alert = status == FlightStatus.NO_GO
    row.label(text=f"STATUS: {status.value}", icon=_status_icon(status))

    if is_stale:
        stale_row = banner.row()
        stale_row.alert = True
        stale_row.label(text="STALE: Scene changed since last scan", icon="FILE_REFRESH")

    banner.label(text=f"{report.critical_count} Critical  |  {report.warning_count} Warning")
    if not compact:
        banner.label(text=f"Scan: {report.scan_duration_ms:.1f} ms  |  HUD: {draw_ms:.2f} ms")
        scores = report.gauge_scores()
        g_row = banner.row(align=True)
        for key in ("ALB", "RGH", "TXL", "SCL"):
            sev, val = scores[key]
            g_row.label(text=f"{key}:{int(round(val))}%", icon=_severity_icon(sev))


def _draw_solid_shading_hint(layout: bpy.types.UILayout) -> None:
    hint = layout.box()
    hint.label(
        text="Viewport is in Solid shading — switch to Material Preview or Rendered for shader overlays",
        icon="INFO",
    )


class FLIGHTCHECK_PT_header_popover(bpy.types.Panel):
    """Quick-access FlightCheck popover launched from the 3D Viewport header."""

    bl_space_type = "VIEW_3D"
    bl_region_type = "HEADER"
    bl_label = "FlightCheck"
    bl_idname = "FLIGHTCHECK_PT_header_popover"
    bl_ui_units_x = 14

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return context.scene is not None and hasattr(context.scene, "flightcheck")

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        settings = getattr(context.scene, "flightcheck", None)
        if settings is None:
            return
        wm_state = getattr(context.window_manager, "flightcheck_wm", None)

        _draw_status_banner(layout, context, compact=True)

        scan_col = layout.column(align=True)
        scan_col.scale_y = 1.15
        scan_col.operator("flightcheck.scan", text="Run Pre-Flight Scan", icon="FILE_REFRESH")

        layout.separator()
        ctrl = layout.column(align=True)
        ctrl.prop(
            settings,
            "enabled",
            toggle=True,
            icon="PAUSE" if settings.enabled else "PLAY",
        )
        mode_row = ctrl.row(align=True)
        mode_row.enabled = bool(settings.enabled)
        mode_row.prop(settings, "diagnostic_mode", text="Mode")

        if settings.enabled and settings.diagnostic_mode != "OFF" and _is_solid_shading(context):
            _draw_solid_shading_hint(layout)

        hud_row = ctrl.row(align=True)
        hud_row.prop(settings, "show_hud_2d")
        hud_row.prop(settings, "show_overlay_3d")

        if wm_state is not None:
            ctrl.prop(wm_state, "hud_style")
            ctrl.prop(wm_state, "hud_position")

        layout.separator()
        if context.mode != "OBJECT":
            warn_box = layout.box()
            warn_box.alert = True
            warn_box.label(text="Switch to Object Mode for geometry/UV fixers", icon="ERROR")

        actions = layout.row(align=True)
        fix_op = actions.operator("flightcheck.fix_all", text="Fix Safe Issues", icon="CHECKMARK")
        if hasattr(fix_op, "include_texels"):
            fix_op.include_texels = False
        actions.operator("flightcheck.cleanup", text="Clean Up All", icon="TRASH")


def draw_view3d_header_status(self, context: bpy.types.Context) -> None:
    """Draw compact clickable FlightCheck status pill in the 3D Viewport header."""
    scene = getattr(context, "scene", None)
    if scene is None or not hasattr(scene, "flightcheck"):
        return

    report = CACHE.get_report(scene)
    is_stale = bool(report is not None and CACHE.is_dirty(scene))

    if report is None:
        text = "FC: READY"
        icon = "INFO"
        is_alert = False
    elif report.status == FlightStatus.GO:
        text = "FC: GO"
        icon = "CHECKMARK"
        is_alert = False
    elif report.status == FlightStatus.CAUTION:
        text = f"FC: CAUTION ({report.warning_count})"
        icon = "INFO"
        is_alert = False
    else:
        text = f"FC: NO-GO ({report.critical_count})"
        icon = "ERROR"
        is_alert = True

    if is_stale:
        text = f"{text} *"

    row = self.layout.row(align=True)
    row.alert = is_alert
    row.popover(
        panel="FLIGHTCHECK_PT_header_popover",
        text=text,
        icon=icon,
    )


class FLIGHTCHECK_MT_pie_menu(bpy.types.Menu):
    """8-direction radial pie menu for rapid diagnostic mode switching and scans."""

    bl_label = "FlightCheck Diagnostics"
    bl_idname = "FLIGHTCHECK_MT_pie_menu"

    def draw(self, context: bpy.types.Context) -> None:
        pie = self.layout.menu_pie()

        # 1. West: Off
        op = pie.operator("flightcheck.set_mode", text="Off", icon="X")
        op.mode = "OFF"
        # 2. East: Combined
        op = pie.operator("flightcheck.set_mode", text="Combined", icon="SHADING_TEXTURE")
        op.mode = "COMBINED"
        # 3. South: Run Pre-Flight Scan
        pie.operator("flightcheck.scan", text="Run Pre-Flight Scan", icon="FILE_REFRESH")
        # 4. North: Albedo
        op = pie.operator("flightcheck.set_mode", text="Albedo", icon="COLOR")
        op.mode = "ALBEDO"
        # 5. North-West: Roughness
        op = pie.operator("flightcheck.set_mode", text="Roughness", icon="NODE_MATERIAL")
        op.mode = "ROUGHNESS"
        # 6. North-East: Texel Density
        op = pie.operator("flightcheck.set_mode", text="Texel Density", icon="UV")
        op.mode = "TEXEL_DENSITY"
        # 7. South-West: Toggle 2D HUD
        op = pie.operator("wm.context_toggle", text="Toggle 2D HUD", icon="OVERLAY")
        op.data_path = "scene.flightcheck.show_hud_2d"
        # 8. South-East: Fix Safe Issues
        op = pie.operator("flightcheck.fix_all", text="Fix Safe Issues", icon="CHECKMARK")
        if hasattr(op, "include_texels"):
            op.include_texels = False


class FLIGHTCHECK_PT_main(_FlightCheckPanelBase):
    """Primary FlightCheck status banner and overlay controls."""

    bl_label = "FlightCheck"
    bl_idname = "FLIGHTCHECK_PT_main"

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        settings = getattr(context.scene, "flightcheck", None)
        if settings is None:
            return
        wm_state = getattr(context.window_manager, "flightcheck_wm", None)

        _draw_status_banner(layout, context, compact=False)

        col = layout.column(align=True)
        col.scale_y = 1.25
        col.operator("flightcheck.scan", text="Run Pre-Flight Scan", icon="FILE_REFRESH")

        layout.separator()
        ctrl = layout.column(align=True)
        ctrl.prop(
            settings,
            "enabled",
            toggle=True,
            icon="PAUSE" if settings.enabled else "PLAY",
        )
        mode_row = ctrl.row(align=True)
        mode_row.enabled = bool(settings.enabled)
        mode_row.prop(settings, "diagnostic_mode")

        if settings.enabled and settings.diagnostic_mode != "OFF" and _is_solid_shading(context):
            _draw_solid_shading_hint(layout)

        row = ctrl.row(align=True)
        row.prop(settings, "show_hud_2d")
        row.prop(settings, "show_overlay_3d")

        if wm_state is not None:
            ctrl.prop(wm_state, "hud_style")
            ctrl.prop(wm_state, "hud_position")


class FLIGHTCHECK_PT_settings(_FlightCheckPanelBase):
    """Collapsible subpanel for texel density, clamping, and scan scope."""

    bl_label = "Diagnostic Settings"
    bl_idname = "FLIGHTCHECK_PT_settings"
    bl_parent_id = "FLIGHTCHECK_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        settings = getattr(context.scene, "flightcheck", None)
        if settings is None:
            return

        tex_col = layout.column(align=True)
        preset_row = tex_col.row(align=True)
        for preset in constants.TEXEL_PRESETS:
            op = preset_row.operator(
                "flightcheck.set_texel_preset",
                text=_format_texel_preset_label(preset),
                depress=abs(float(settings.target_texel_density) - float(preset)) < 0.5,
            )
            op.density = preset

        tex_col.prop(settings, "target_texel_density")
        tex_col.prop(settings, "default_texture_size")

        layout.separator()
        clamp_col = layout.column(align=True)
        clamp_col.prop(settings, "clamp_dielectric_knee")
        clamp_col.prop(settings, "clamp_dielectric_max")

        if float(settings.clamp_dielectric_max) > constants.DIELECTRIC_WARN_LUMA:
            warn_box = clamp_col.box()
            warn_box.alert = True
            warn_box.label(
                text=f"Dielectric ceiling > {constants.DIELECTRIC_WARN_LUMA:.2f} triggers warn",
                icon="ERROR",
            )

        layout.separator()
        scope_col = layout.column(align=True)
        scope_col.prop(settings, "selected_only")
        scope_col.prop(settings, "auto_rescan")


class FLIGHTCHECK_PT_fixers(_FlightCheckPanelBase):
    """Subpanel exposing non-destructive automated remediation operators."""

    bl_label = "Automated Remediation"
    bl_idname = "FLIGHTCHECK_PT_fixers"
    bl_parent_id = "FLIGHTCHECK_PT_main"

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        settings = getattr(context.scene, "flightcheck", None)

        if context.mode != "OBJECT":
            warn_box = layout.box()
            warn_box.alert = True
            warn_box.label(
                text="Switch to Object Mode for geometry/UV fixers",
                icon="ERROR",
            )

        col = layout.column(align=True)
        col.scale_y = 1.2
        safe_op = col.operator("flightcheck.fix_all", text="Fix Safe Issues", icon="CHECKMARK")
        if hasattr(safe_op, "include_texels"):
            safe_op.include_texels = False

        layout.separator()
        sub = layout.column(align=True)
        sub.operator("flightcheck.clamp_pbr", text="Clamp PBR (Soft-Knee)", icon="SHADING_RENDERED")
        sub.operator("flightcheck.fix_colorspace", text="Fix Data Color Spaces", icon="IMAGE_DATA")
        sub.operator("flightcheck.fix_transforms", text="Fix Object Transforms", icon="OBJECT_ORIGIN")

        layout.separator()
        tex_box = layout.box()
        tex_box.label(text="Texel Density Equalization", icon="UV")
        if settings is not None:
            tex_box.prop(settings, "equalize_mode", text="Mode")
        eq_op = tex_box.operator("flightcheck.equalize_texels", text="Equalize Texel Density", icon="UV")
        if settings is not None and hasattr(eq_op, "mode"):
            eq_op.mode = settings.equalize_mode

        layout.separator()
        layout.operator("flightcheck.cleanup", text="Clean Up All", icon="TRASH")


class FLIGHTCHECK_PT_issues(_FlightCheckPanelBase):
    """Pre-flight checklist displaying filtered DiagnosticIssue items."""

    bl_label = "Pre-Flight Checklist"
    bl_idname = "FLIGHTCHECK_PT_issues"
    bl_parent_id = "FLIGHTCHECK_PT_main"

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        wm = getattr(context, "window_manager", None)
        wm_state = getattr(wm, "flightcheck_wm", None) if wm is not None else None

        filter_val = "ALL"
        show_details = True
        if wm_state is not None:
            row = layout.row(align=True)
            row.prop(wm_state, "active_issue_filter", text="")
            row.prop(wm_state, "show_details", text="Details")
            filter_val = wm_state.active_issue_filter
            show_details = wm_state.show_details

        report = CACHE.get_report(context.scene)
        if report is None:
            layout.label(text="Run Pre-Flight Scan to populate issues.", icon="INFO")
            return

        ordered = _order_filtered_issues(report.issues, filter_val)
        layout.label(text=f"Showing {len(ordered)} of {len(report.issues)} issue(s)")

        if not ordered:
            ok_box = layout.box()
            ok_box.label(text="All checks passed for current filter.", icon="CHECKMARK")
            return

        issue_col = getattr(wm, "flightcheck_issues", None) if wm is not None else None
        if issue_col is not None and wm_state is not None and _is_collection_synced(issue_col, ordered):
            layout.template_list(
                "FLIGHTCHECK_UL_issues",
                "issues",
                wm,
                "flightcheck_issues",
                wm_state,
                "active_issue_index",
                rows=5,
                maxrows=8,
            )
            if show_details and 0 <= wm_state.active_issue_index < len(issue_col):
                active = issue_col[wm_state.active_issue_index]
                detail_box = layout.box()
                detail_box.label(text=active.message)
                meta_parts: List[str] = []
                if active.measured_text:
                    meta_parts.append(f"Measured: {active.measured_text}")
                if active.expected_range:
                    meta_parts.append(f"Expected: {active.expected_range}")
                if active.is_shared_image:
                    meta_parts.append("Shared image (protected)")
                elif not active.fixable:
                    meta_parts.append("Manual review")
                if meta_parts:
                    detail_box.label(text=" | ".join(meta_parts))
                if active.object_name or active.material_name:
                    sel_op = detail_box.operator(
                        "flightcheck.select_issue_target",
                        text="Select & Frame Target",
                        icon="RESTRICT_SELECT_OFF",
                    )
                    sel_op.object_name = active.object_name
                    sel_op.material_name = active.material_name
            return

        # CollectionProperty cannot be mutated inside draw(); schedule deferred sync and show fallback rows.
        _schedule_deferred_issue_sync()

        for issue in ordered[:_MAX_DISPLAYED_ISSUES]:
            box = layout.box()
            header = box.row(align=True)
            header.alert = issue.severity == Severity.CRITICAL

            targets = [t for t in (issue.object_name, issue.material_name, issue.image_name) if t]
            target_label = " / ".join(targets) if targets else issue.category.value
            label_name = constants.ISSUE_LABELS.get(issue.code, issue.code)
            header.label(text=f"{label_name} ({target_label})", icon=_severity_icon(issue.severity))

            if issue.is_shared_image:
                header.label(text="", icon="LOCKED")

            sel_row = header.row(align=True)
            sel_row.enabled = bool(issue.object_name or issue.material_name)
            sel_op = sel_row.operator(
                "flightcheck.select_issue_target",
                text="",
                icon="RESTRICT_SELECT_OFF",
            )
            sel_op.object_name = issue.object_name
            sel_op.material_name = issue.material_name

            if show_details:
                box.label(text=issue.message)
                meta_parts = []
                if issue.measured_value is not None:
                    meta_parts.append(f"Measured: {issue.measured_value:.3f}")
                if issue.expected_range:
                    meta_parts.append(f"Expected: {issue.expected_range}")
                if issue.is_shared_image:
                    meta_parts.append("Shared image (protected)")
                elif not issue.fixable:
                    meta_parts.append("Manual review")
                if meta_parts:
                    box.label(text=" | ".join(meta_parts))

        if len(ordered) > _MAX_DISPLAYED_ISSUES:
            layout.label(text=f"... {len(ordered) - _MAX_DISPLAYED_ISSUES} more issue(s) omitted", icon="INFO")


CLASSES = (
    FLIGHTCHECK_OT_select_issue_target,
    FLIGHTCHECK_OT_set_texel_preset,
    FLIGHTCHECK_OT_set_mode,
    FLIGHTCHECK_UL_issues,
    FLIGHTCHECK_PT_header_popover,
    FLIGHTCHECK_MT_pie_menu,
    FLIGHTCHECK_PT_main,
    FLIGHTCHECK_PT_settings,
    FLIGHTCHECK_PT_fixers,
    FLIGHTCHECK_PT_issues,
)
