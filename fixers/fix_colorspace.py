"""Color space remediation for data-only image textures with shared-image protection."""

from __future__ import annotations

from typing import Dict, List, Optional

import bpy

from . import collect_scene_materials


def fix_scene_colorspaces(
    scene: Optional[bpy.types.Scene],
    selected_only: bool = False,
) -> Dict[str, List[str]]:
    """Switch data-only textures to 'Non-Color', protecting images shared with color sockets."""
    if scene is None:
        return {"fixed": [], "protected_shared": []}

    from ..core import graph_crawler

    materials = collect_scene_materials(scene, selected_only=selected_only)
    if not materials:
        return {"fixed": [], "protected_shared": []}

    cs_map = graph_crawler.collect_colorspace_map(materials)

    fixed: List[str] = []
    protected_shared: List[str] = []

    for image, info in cs_map.items():
        if image is None or not hasattr(image, "colorspace_settings"):
            continue
        if not info.get("data_sockets"):
            continue
        if image.colorspace_settings.name == "Non-Color":
            continue

        if info.get("color_sockets"):
            protected_shared.append(image.name)
        else:
            image.colorspace_settings.name = "Non-Color"
            fixed.append(image.name)

    return {
        "fixed": sorted(fixed),
        "protected_shared": sorted(protected_shared),
    }


class FLIGHTCHECK_OT_fix_colorspace(bpy.types.Operator):
    """Set data-only image textures to Non-Color while protecting shared color maps."""

    bl_idname = "flightcheck.fix_colorspace"
    bl_label = "Fix Data Color Spaces"
    bl_description = (
        "Set data-only image textures to Non-Color; protect images shared with color sockets"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        from ..core import run_full_scan
        from ..hud import tag_view3d_redraw

        scene = context.scene
        fc = getattr(scene, "flightcheck", None)
        selected_only = bool(fc.selected_only) if fc is not None else False

        result = fix_scene_colorspaces(scene, selected_only=selected_only)
        run_full_scan(scene, write_texel_attr=True)
        tag_view3d_redraw()

        fixed_count = len(result["fixed"])
        protected_count = len(result["protected_shared"])
        level = {"WARNING"} if (protected_count > 0 and fixed_count == 0) else {"INFO"}
        self.report(
            level,
            (
                f"FlightCheck: Fixed {fixed_count} data map(s); "
                f"protected {protected_count} shared image(s)."
            ),
        )
        return {"FINISHED"}
