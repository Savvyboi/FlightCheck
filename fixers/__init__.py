"""Non-destructive remediation operators and orchestration for FlightCheck."""

from __future__ import annotations

from typing import List, Optional

import bpy
from bpy.props import BoolProperty

from .. import constants
from ..state import CACHE, DiagnosticReport, cancel_pending_rescan, internal_mutation


def iter_scene_mesh_objects(
    scene: Optional[bpy.types.Scene],
    selected_only: bool = False,
) -> List[bpy.types.Object]:
    """Return mesh objects in ``scene``, optionally restricted to selected objects."""
    if scene is None:
        return []

    view_layer = None
    if selected_only:
        ctx = getattr(bpy, "context", None)
        if (
            ctx is not None
            and getattr(ctx, "scene", None) == scene
            and getattr(ctx, "view_layer", None) is not None
        ):
            view_layer = ctx.view_layer
        elif len(scene.view_layers) > 0:
            view_layer = scene.view_layers[0]

    objects: List[bpy.types.Object] = []
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        if selected_only:
            try:
                is_selected = (
                    obj.select_get(view_layer=view_layer)
                    if view_layer is not None
                    else obj.select_get()
                )
            except RuntimeError:
                is_selected = False
            if not is_selected:
                continue
        objects.append(obj)
    return objects


def update_view_layer(scene: Optional[bpy.types.Scene] = None) -> None:
    """Flush pending RNA object transform edits into ``matrix_world`` and ``dimensions``."""
    ctx = getattr(bpy, "context", None)
    if scene is not None and len(getattr(scene, "view_layers", ())) > 0:
        try:
            scene.view_layers[0].update()
            return
        except RuntimeError:
            pass
    if ctx is not None and getattr(ctx, "view_layer", None) is not None:
        try:
            ctx.view_layer.update()
        except RuntimeError:
            pass


def collect_scene_materials(
    scene: Optional[bpy.types.Scene],
    selected_only: bool = False,
) -> List[bpy.types.Material]:
    """Collect unique valid materials with node trees from scene mesh objects."""
    materials: List[bpy.types.Material] = []
    seen_ptrs: set[int] = set()
    for obj in iter_scene_mesh_objects(scene, selected_only=selected_only):
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None or getattr(mat, "node_tree", None) is None:
                continue
            ptr = mat.as_pointer()
            if ptr not in seen_ptrs:
                seen_ptrs.add(ptr)
                materials.append(mat)
    return materials


def sync_wm_issues(
    context: Optional[bpy.types.Context],
    report: Optional[DiagnosticReport],
) -> None:
    """Synchronize the WindowManager UIList issue collection when registered."""
    if context is None:
        return
    wm = getattr(context, "window_manager", None)
    wm_state = getattr(wm, "flightcheck_wm", None) if wm is not None else None
    if wm is None or wm_state is None or not hasattr(wm, "flightcheck_issues"):
        return
    from ..properties import sync_issue_collection

    sync_issue_collection(wm, report, filter_val=wm_state.active_issue_filter)


from . import clamp_pbr, equalize_texels, fix_colorspace, fix_transforms  # noqa: E402
from .clamp_pbr import (  # noqa: E402
    FLIGHTCHECK_OT_clamp_pbr,
    clamp_material_pbr,
    clamp_scene_pbr,
)
from .equalize_texels import (  # noqa: E402
    FLIGHTCHECK_OT_equalize_texels,
    check_uvs_exceed_unit_square,
    equalize_object_texels,
    equalize_scene_texels,
)
from .fix_colorspace import (  # noqa: E402
    FLIGHTCHECK_OT_fix_colorspace,
    fix_scene_colorspaces,
)
from .fix_transforms import (  # noqa: E402
    FLIGHTCHECK_OT_fix_transforms,
    fix_object_transform,
    fix_scene_transforms,
)


def _sync_spliced_overlays(scene: Optional[bpy.types.Scene]) -> None:
    """Keep active diagnostic overlays synchronized after scans or fixes."""
    from ..core import pbr_shader_nodes, shader_splicer
    from ..hud import sync_draw_handlers

    fc = getattr(scene, "flightcheck", None) if scene is not None else None
    mode_name = fc.diagnostic_mode if fc is not None else "OFF"
    mode_int = constants.MODE_NAME_TO_INT.get(mode_name, constants.MODE_OFF)

    if fc is not None and fc.enabled and mode_int != constants.MODE_OFF:
        reconcile = getattr(shader_splicer, "reconcile_scene_splices", None)
        if callable(reconcile):
            reconcile(scene, selected_only=bool(fc.selected_only))
        else:
            shader_splicer.splice_scene_materials(scene, selected_only=bool(fc.selected_only))
        pbr_shader_nodes.set_ctrl_mode(mode_int)
    else:
        for mat in bpy.data.materials:
            if constants.is_editable_id(mat) and shader_splicer.is_material_spliced(mat):
                shader_splicer.resync_material(mat)

    if scene is not None:
        sync_draw_handlers(scene)


class FLIGHTCHECK_OT_scan(bpy.types.Operator):
    """Run a full FlightCheck diagnostic scan on the active scene."""

    bl_idname = "flightcheck.scan"
    bl_label = "Run Pre-Flight Scan"
    bl_description = "Scan scene PBR values, color spaces, texel density, and object transforms"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return context.scene is not None

    def execute(self, context: bpy.types.Context) -> set[str]:
        from ..core import run_full_scan
        from ..hud import tag_view3d_redraw

        scene = context.scene
        with internal_mutation():
            report = run_full_scan(scene, write_texel_attr=True)
            sync_wm_issues(context, report)
            _sync_spliced_overlays(scene)
            tag_view3d_redraw()

        self.report(
            {"INFO"},
            (
                f"FlightCheck [{report.status.value}]: "
                f"{report.critical_count} critical, {report.warning_count} warning(s) "
                f"in {report.scan_duration_ms:.1f} ms"
            ),
        )
        return {"FINISHED"}


class FLIGHTCHECK_OT_fix_all(bpy.types.Operator):
    """Apply automated non-destructive FlightCheck remediation fixers."""

    bl_idname = "flightcheck.fix_all"
    bl_label = "Fix Safe Issues"
    bl_description = (
        "Run transform, color space, and soft-knee PBR clamping fixers (optionally equalize UVs)"
    )
    bl_options = {"REGISTER", "UNDO"}

    include_texels: BoolProperty(
        name="Equalize Texel Density",
        description="Also equalize UV island texel density (modifies UVs)",
        default=False,
        options={"SKIP_SAVE"},
    )

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return context.scene is not None and context.mode == "OBJECT"

    def execute(self, context: bpy.types.Context) -> set[str]:
        from ..core import run_full_scan
        from ..hud import tag_view3d_redraw

        scene = context.scene
        fc = getattr(scene, "flightcheck", None)
        selected_only = bool(fc.selected_only) if fc is not None else False
        target_td = (
            float(fc.target_texel_density)
            if fc is not None
            else constants.TEXEL_DEFAULT_TARGET
        )
        default_tex = (
            int(fc.default_texture_size)
            if fc is not None
            else constants.TEXEL_DEFAULT_RES
        )
        eq_mode = getattr(fc, "equalize_mode", "RELATIVE") if fc is not None else "RELATIVE"

        with internal_mutation():
            tr_res = fix_scene_transforms(scene, selected_only=selected_only)
            cs_res = fix_scene_colorspaces(scene, selected_only=selected_only)
            pbr_res = clamp_scene_pbr(scene, selected_only=selected_only)
            tx_res = None
            if self.include_texels:
                tx_res = equalize_scene_texels(
                    scene,
                    target_px_per_m=target_td,
                    default_tex_size=default_tex,
                    selected_only=selected_only,
                    mode=eq_mode,
                    pack_islands=True,
                )

            report = run_full_scan(scene, write_texel_attr=True)
            sync_wm_issues(context, report)
            _sync_spliced_overlays(scene)
            tag_view3d_redraw()

        parts = [
            f"Transforms fixed: {tr_res['objects_fixed']} (split {tr_res['meshes_split']})",
            f"Colorspaces fixed: {len(cs_res['fixed'])} (protected {len(cs_res['protected_shared'])})",
            f"PBR clamped: {pbr_res['materials_clamped']}",
        ]
        warn = False
        if tx_res is not None:
            parts.append(
                f"UV islands scaled ({eq_mode}): {tx_res['islands_scaled']} "
                f"(meshes_split={tx_res['meshes_split']}, "
                f"uvs_exceed_unit_square={tx_res['uvs_exceed_unit_square']})"
            )
            if tx_res["meshes_split"] > 0 or tx_res["uvs_exceed_unit_square"] > 0:
                warn = True

        summary = ", ".join(parts) + f" -> {report.status.value}"
        self.report({"WARNING"} if warn else {"INFO"}, f"FlightCheck Fix All: {summary}")
        return {"FINISHED"}


class FLIGHTCHECK_OT_cleanup(bpy.types.Operator):
    """Remove all FlightCheck shader overlays, attributes, HUD handlers, and optional clamps."""

    bl_idname = "flightcheck.cleanup"
    bl_label = "Clean Up All"
    bl_description = (
        "Restore original material links, remove temporary attributes/groups, and disable HUD"
    )
    bl_options = {"REGISTER", "UNDO"}

    remove_clamps: BoolProperty(
        name="Remove PBR Clamp Nodes",
        description="Restore original shader links and remove __FLIGHTCHECK_CLAMP_NODE__ nodes",
        default=True,
    )

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return context.scene is not None

    def execute(self, context: bpy.types.Context) -> set[str]:
        from ..core import graph_crawler, shader_splicer, texel_analyzer
        from ..hud import tag_view3d_redraw, unregister_all_draw_handlers

        scene = context.scene
        fc = getattr(scene, "flightcheck", None) if scene is not None else None

        with internal_mutation():
            cancel_pending_rescan()
            if fc is not None:
                fc.enabled = False
                fc.diagnostic_mode = "OFF"

            shader_res = shader_splicer.cleanup_all(remove_clamps=bool(self.remove_clamps))
            attrs_removed = texel_analyzer.remove_texel_attributes(scene=None)
            unregister_all_draw_handlers()

            clear_img_cache = getattr(graph_crawler, "clear_image_sample_cache", None)
            if callable(clear_img_cache):
                clear_img_cache()
            CACHE.clear()
            sync_wm_issues(context, None)
            tag_view3d_redraw()

        self.report(
            {"INFO"},
            (
                f"FlightCheck cleaned up: {shader_res} | "
                f"texel attributes removed: {attrs_removed}"
            ),
        )
        return {"FINISHED"}


CLASSES = (
    FLIGHTCHECK_OT_clamp_pbr,
    FLIGHTCHECK_OT_fix_colorspace,
    FLIGHTCHECK_OT_fix_transforms,
    FLIGHTCHECK_OT_equalize_texels,
    FLIGHTCHECK_OT_scan,
    FLIGHTCHECK_OT_fix_all,
    FLIGHTCHECK_OT_cleanup,
)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


__all__ = [
    "CLASSES",
    "FLIGHTCHECK_OT_clamp_pbr",
    "FLIGHTCHECK_OT_cleanup",
    "FLIGHTCHECK_OT_equalize_texels",
    "FLIGHTCHECK_OT_fix_all",
    "FLIGHTCHECK_OT_fix_colorspace",
    "FLIGHTCHECK_OT_fix_transforms",
    "FLIGHTCHECK_OT_scan",
    "check_uvs_exceed_unit_square",
    "clamp_material_pbr",
    "clamp_pbr",
    "clamp_scene_pbr",
    "collect_scene_materials",
    "equalize_object_texels",
    "equalize_scene_texels",
    "equalize_texels",
    "fix_colorspace",
    "fix_object_transform",
    "fix_scene_colorspaces",
    "fix_scene_transforms",
    "fix_transforms",
    "iter_scene_mesh_objects",
    "register",
    "sync_wm_issues",
    "unregister",
    "update_view_layer",
]
