"""View3D N-Panel UI for FlightCheck diagnostics and remediation."""

from __future__ import annotations

from typing import List

import bpy

from ..state import CACHE, DiagnosticIssue, FlightStatus, Severity

_MAX_DISPLAYED_ISSUES = 80


def _severity_icon(severity: Severity) -> str:
    """Map DiagnosticIssue severity to a standard Blender UI icon."""
    if severity == Severity.CRITICAL:
        return "ERROR"
    if severity == Severity.WARNING:
        return "INFO"
    return "CHECKMARK"


def _status_icon(status: FlightStatus) -> str:
    """Map FlightStatus to a standard Blender UI icon."""
    if status == FlightStatus.GO:
        return "CHECKMARK"
    if status == FlightStatus.CAUTION:
        return "INFO"
    return "ERROR"


class _FlightCheckPanelBase(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "FlightCheck"

    @classmethod
    def poll(cls, context) -> bool:
        return context.scene is not None and hasattr(context.scene, "flightcheck")


class FLIGHTCHECK_PT_main(_FlightCheckPanelBase):
    """Primary FlightCheck status banner and overlay controls."""

    bl_label = "FlightCheck"
    bl_idname = "FLIGHTCHECK_PT_main"

    def draw(self, context) -> None:
        layout = self.layout
        settings = getattr(context.scene, "flightcheck", None)
        if settings is None:
            return
        wm_state = getattr(context.window_manager, "flightcheck_wm", None)

        report = CACHE.get_report()
        draw_ms = CACHE.get_draw_time_ms()

        banner = layout.box()
        if report is None:
            banner.label(text="STATUS: READY (Not Scanned)", icon="INFO")
            banner.label(text=f"HUD Draw: {draw_ms:.2f} ms")
        else:
            status = report.status
            row = banner.row()
            row.alert = status == FlightStatus.NO_GO
            row.label(text=f"STATUS: {status.value}", icon=_status_icon(status))
            banner.label(text=f"{report.critical_count} Critical  |  {report.warning_count} Warning")
            banner.label(text=f"Scan: {report.scan_duration_ms:.1f} ms  |  HUD: {draw_ms:.2f} ms")

            scores = report.gauge_scores()
            g_row = banner.row(align=True)
            for key in ("ALB", "RGH", "TXL", "SCL"):
                sev, val = scores[key]
                g_row.label(text=f"{key}:{int(round(val))}%", icon=_severity_icon(sev))

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
        ctrl.prop(settings, "diagnostic_mode")

        row = ctrl.row(align=True)
        row.prop(settings, "show_hud_2d")
        row.prop(settings, "show_overlay_3d")

        if wm_state is not None:
            ctrl.prop(wm_state, "hud_position")


class FLIGHTCHECK_PT_settings(_FlightCheckPanelBase):
    """Collapsible subpanel for texel density, clamping, and scan scope."""

    bl_label = "Diagnostic Settings"
    bl_idname = "FLIGHTCHECK_PT_settings"
    bl_parent_id = "FLIGHTCHECK_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context) -> None:
        layout = self.layout
        settings = getattr(context.scene, "flightcheck", None)
        if settings is None:
            return

        col = layout.column(align=True)
        col.prop(settings, "target_texel_density")
        col.prop(settings, "default_texture_size")

        layout.separator()
        clamp_col = layout.column(align=True)
        clamp_col.prop(settings, "clamp_dielectric_knee")
        clamp_col.prop(settings, "clamp_dielectric_max")

        layout.separator()
        scope_col = layout.column(align=True)
        scope_col.prop(settings, "selected_only")
        scope_col.prop(settings, "auto_rescan")


class FLIGHTCHECK_PT_fixers(_FlightCheckPanelBase):
    """Subpanel exposing non-destructive automated remediation operators."""

    bl_label = "Automated Remediation"
    bl_idname = "FLIGHTCHECK_PT_fixers"
    bl_parent_id = "FLIGHTCHECK_PT_main"

    def draw(self, context) -> None:
        layout = self.layout
        col = layout.column(align=True)
        col.scale_y = 1.2
        col.operator("flightcheck.fix_all", text="Fix All Remediable", icon="CHECKMARK")

        layout.separator()
        sub = layout.column(align=True)
        sub.operator("flightcheck.clamp_pbr", text="Clamp PBR (Soft-Knee)", icon="SHADING_RENDERED")
        sub.operator("flightcheck.fix_colorspace", text="Fix Data Color Spaces", icon="IMAGE_DATA")
        sub.operator("flightcheck.fix_transforms", text="Fix Object Transforms", icon="OBJECT_ORIGIN")
        sub.operator("flightcheck.equalize_texels", text="Equalize Texel Density", icon="UV")

        layout.separator()
        layout.operator("flightcheck.cleanup", text="Clean Up All", icon="TRASH")


class FLIGHTCHECK_PT_issues(_FlightCheckPanelBase):
    """Pre-flight checklist displaying filtered DiagnosticIssue items."""

    bl_label = "Pre-Flight Checklist"
    bl_idname = "FLIGHTCHECK_PT_issues"
    bl_parent_id = "FLIGHTCHECK_PT_main"

    def draw(self, context) -> None:
        layout = self.layout
        wm_state = getattr(context.window_manager, "flightcheck_wm", None)

        filter_val = "ALL"
        show_details = True
        if wm_state is not None:
            row = layout.row(align=True)
            row.prop(wm_state, "active_issue_filter", text="")
            row.prop(wm_state, "show_details", text="Details")
            filter_val = wm_state.active_issue_filter
            show_details = wm_state.show_details

        report = CACHE.get_report()
        if report is None:
            layout.label(text="Run Pre-Flight Scan to populate issues.", icon="INFO")
            return

        if filter_val == "ALL":
            filtered: List[DiagnosticIssue] = list(report.issues)
        else:
            filtered = [issue for issue in report.issues if issue.category.value == filter_val]

        layout.label(text=f"Showing {len(filtered)} of {len(report.issues)} issue(s)")

        if not filtered:
            ok_box = layout.box()
            ok_box.label(text="All checks passed for current filter.", icon="CHECKMARK")
            return

        severity_rank = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.OK: 2}
        ordered = sorted(
            filtered,
            key=lambda i: (severity_rank.get(i.severity, 3), i.category.value, i.code, i.object_name, i.material_name),
        )

        for issue in ordered[:_MAX_DISPLAYED_ISSUES]:
            box = layout.box()
            header = box.row(align=True)
            header.alert = issue.severity == Severity.CRITICAL

            targets = [t for t in (issue.object_name, issue.material_name, issue.image_name) if t]
            target_label = " / ".join(targets) if targets else issue.category.value
            header.label(text=f"[{issue.code}] {target_label}", icon=_severity_icon(issue.severity))

            if issue.is_shared_image:
                header.label(text="", icon="LOCKED")

            if show_details:
                box.label(text=issue.message)
                meta_parts: List[str] = []
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
    FLIGHTCHECK_PT_main,
    FLIGHTCHECK_PT_settings,
    FLIGHTCHECK_PT_fixers,
    FLIGHTCHECK_PT_issues,
)
