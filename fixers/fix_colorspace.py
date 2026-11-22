"""Color space remediation for data-only image textures with shared-image protection."""

from __future__ import annotations

from typing import Dict, List, Optional

import bpy

from .. import constants
from ..state import internal_mutation
from . import collect_scene_materials, sync_wm_issues

_NON_COLOR_PREFERENCE: tuple[str, ...] = (
    "Non-Color",
    "Raw",
    "Utility - Raw",
    "Generic Data",
)


def _is_valid_data_colorspace(image: bpy.types.Image, graph_crawler) -> bool:
    """Delegate to ``graph_crawler.is_valid_data_colorspace`` with a local fallback."""
    checker = getattr(graph_crawler, "is_valid_data_colorspace", None)
    if callable(checker):
        return bool(checker(image))

    cs_settings = getattr(image, "colorspace_settings", None)
    if cs_settings is None:
        return True
    if bool(getattr(cs_settings, "is_data", False)):
        return True
    cs_name = str(getattr(cs_settings, "name", "") or "")
    if cs_name == "Non-Color":
        return True
    if getattr(image, "is_float", False):
        low = cs_name.lower()
        if any(tok in low for tok in ("linear", "raw", "aces", "non-color")):
            return True
    return False


def _resolve_non_color_name(image: bpy.types.Image) -> Optional[str]:
    """Resolve the active OCIO config's non-color space identifier for ``image``."""
    cs_settings = getattr(image, "colorspace_settings", None)
    if cs_settings is None:
        return None
    prop = cs_settings.bl_rna.properties.get("name")
    if prop is None:
        return "Non-Color"

    identifiers = [item.identifier for item in getattr(prop, "enum_items", ())]
    if not identifiers:
        return None

    for preferred in _NON_COLOR_PREFERENCE:
        if preferred in identifiers:
            return preferred

    for ident in identifiers:
        low = ident.lower()
        if "non-color" in low or "raw" in low or "data" in low:
            return ident

    return None


def fix_scene_colorspaces(
    scene: Optional[bpy.types.Scene],
    selected_only: bool = False,
) -> Dict[str, List[str]]:
    """Switch data-only textures to Non-Color, protecting shared and linked images."""
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
        if not constants.is_editable_id(image):
            continue
        if not info.get("data_sockets"):
            continue
        if _is_valid_data_colorspace(image, graph_crawler):
            continue

        if info.get("color_sockets"):
            protected_shared.append(image.name)
            continue

        target_space = _resolve_non_color_name(image)
        if not target_space:
            continue

        try:
            image.colorspace_settings.name = target_space
        except (TypeError, AttributeError, RuntimeError):
            continue
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

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return context.scene is not None

    def execute(self, context: bpy.types.Context) -> set[str]:
        from ..core import run_full_scan
        from ..hud import tag_view3d_redraw

        scene = context.scene
        fc = getattr(scene, "flightcheck", None)
        selected_only = bool(fc.selected_only) if fc is not None else False

        with internal_mutation():
            result = fix_scene_colorspaces(scene, selected_only=selected_only)
            report = run_full_scan(scene, write_texel_attr=True)
            sync_wm_issues(context, report)
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
