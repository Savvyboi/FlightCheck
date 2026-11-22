"""Headless CLI verification harness for FlightCheck (Blender 5.0+ extension).

Runs 7 diagnostic test suites covering Blender 5.0+ API compliance, extension
lifecycle, static detection, shader splicing, texel density, HUD/3D overlays,
automated fixers, and full non-destructive cleanup.
"""

from __future__ import annotations

import ast
import sys
import time
import tomllib
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import bpy

_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

import flightcheck  # noqa: E402
from flightcheck import constants  # noqa: E402
from flightcheck.state import CACHE, FlightStatus, Severity  # noqa: E402
from flightcheck.tests import generate_test_scene  # noqa: E402


def _ensure_material_nodes(mat: bpy.types.Material):
    """Ensure material node tree exists across Blender 4.2 LTS and Blender 5.0+."""
    if mat.node_tree is None and bpy.app.version < (5, 0, 0):
        mat.use_nodes = True
    return mat.node_tree


def _ensure_registered() -> None:
    if not hasattr(bpy.types.Scene, "flightcheck"):
        flightcheck.register()


def _reset_scan_settings(scene: bpy.types.Scene) -> None:
    fc = scene.flightcheck
    fc.enabled = False
    fc.diagnostic_mode = "OFF"
    fc.selected_only = False
    fc.target_texel_density = constants.TEXEL_DEFAULT_TARGET
    fc.default_texture_size = constants.TEXEL_DEFAULT_RES
    fc.clamp_dielectric_knee = constants.CLAMP_DIELECTRIC_KNEE
    fc.clamp_dielectric_max = constants.CLAMP_DIELECTRIC_MAX


def _find_active_output_node(nt: bpy.types.ShaderNodeTree) -> bpy.types.Node | None:
    fallback = None
    for node in nt.nodes:
        if node.type == "OUTPUT_MATERIAL":
            if getattr(node, "is_active_output", False):
                return node
            if fallback is None:
                fallback = node
    return fallback


def _find_principled_node(nt: bpy.types.ShaderNodeTree) -> bpy.types.Node | None:
    for node in nt.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    return None


def _snapshot_materials() -> Dict[str, Dict[str, Any]]:
    snapshot: Dict[str, Dict[str, Any]] = {}
    tracked_inputs = ("Base Color", "Metallic", "Roughness", "Specular IOR Level", "IOR")
    for mat in bpy.data.materials:
        nt = _ensure_material_nodes(mat)
        if nt is None:
            continue
        out_node = _find_active_output_node(nt)
        surf_links: List[Tuple[str, str]] = []
        if out_node is not None:
            surf = out_node.inputs.get("Surface")
            if surf is not None:
                surf_links = [(lnk.from_node.name, lnk.from_socket.name) for lnk in surf.links]

        bsdf = _find_principled_node(nt)
        bsdf_inputs: Dict[str, Any] = {}
        if bsdf is not None:
            for sock_name in tracked_inputs:
                sock = bsdf.inputs.get(sock_name)
                if sock is None:
                    continue
                if sock.is_linked:
                    lnk = sock.links[0]
                    bsdf_inputs[sock_name] = ("LINKED", lnk.from_node.name, lnk.from_socket.name)
                else:
                    val = sock.default_value
                    if hasattr(val, "__len__"):
                        bsdf_inputs[sock_name] = ("VALUE", tuple(float(v) for v in val))
                    else:
                        bsdf_inputs[sock_name] = ("VALUE", (float(val),))

        snapshot[mat.name] = {
            "surface": surf_links,
            "bsdf_inputs": bsdf_inputs,
        }
    return snapshot


def _assert_material_snapshots_equal(
    before: Dict[str, Dict[str, Any]],
    after: Dict[str, Dict[str, Any]],
) -> None:
    assert set(before.keys()) == set(after.keys()), (
        f"Material set changed: before={sorted(before.keys())}, after={sorted(after.keys())}"
    )
    for mat_name, pre in before.items():
        post = after[mat_name]
        assert pre["surface"] == post["surface"], (
            f"Material {mat_name} Surface links mismatch: before={pre['surface']}, after={post['surface']}"
        )
        for sock_name, pre_val in pre["bsdf_inputs"].items():
            post_val = post["bsdf_inputs"].get(sock_name)
            assert post_val is not None and post_val[0] == pre_val[0], (
                f"Material {mat_name}.{sock_name} state mismatch: before={pre_val}, after={post_val}"
            )
            if pre_val[0] == "LINKED":
                assert pre_val == post_val, (
                    f"Material {mat_name}.{sock_name} link mismatch: before={pre_val}, after={post_val}"
                )
            else:
                diffs = [abs(a - b) for a, b in zip(pre_val[1], post_val[1])]
                assert len(pre_val[1]) == len(post_val[1]) and max(diffs, default=0.0) <= 1e-5, (
                    f"Material {mat_name}.{sock_name} value mismatch: before={pre_val[1]}, after={post_val[1]}"
                )


def _collect_docstring_ids(tree: ast.AST) -> set[int]:
    doc_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                val = node.body[0].value
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    doc_ids.add(id(val))
    return doc_ids


def test_blender5_compliance() -> None:
    """Suite 1: Verify manifest metadata and Blender 5.0+ strict API constraints."""
    pkg_dir = Path(__file__).resolve().parents[1]
    manifest_path = pkg_dir / "blender_manifest.toml"
    assert manifest_path.is_file(), f"Missing manifest at {manifest_path}"

    with manifest_path.open("rb") as fh:
        manifest = tomllib.load(fh)

    assert manifest.get("schema_version") == "1.0.0", f"Bad schema_version: {manifest.get('schema_version')}"
    assert manifest.get("id") == "flightcheck", f"Bad id: {manifest.get('id')}"
    assert manifest.get("blender_version_min") == "5.0.0", (
        f"Bad blender_version_min: {manifest.get('blender_version_min')}"
    )

    forbidden_builtins = ("3D_UNIFORM_COLOR", "2D_UNIFORM_COLOR")
    violations: List[str] = []
    checked_files = 0

    for py_file in sorted(pkg_dir.rglob("*.py")):
        rel_parts = py_file.relative_to(pkg_dir).parts
        if "tests" in rel_parts or "__pycache__" in rel_parts:
            continue
        checked_files += 1
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))
        doc_ids = _collect_docstring_ids(tree)
        rel_str = str(py_file.relative_to(pkg_dir))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "bgl" or alias.name.startswith("bgl."):
                        violations.append(f"{rel_str}:{node.lineno}: forbidden import '{alias.name}'")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod == "bgl" or mod.startswith("bgl."):
                    violations.append(f"{rel_str}:{node.lineno}: forbidden from-import '{mod}'")
            elif isinstance(node, ast.Attribute):
                if node.attr == "use_nodes":
                    violations.append(f"{rel_str}:{node.lineno}: forbidden '.use_nodes' attribute")
            elif isinstance(node, ast.Name):
                if node.id in {"bl_info", "bgl"}:
                    violations.append(f"{rel_str}:{node.lineno}: forbidden identifier '{node.id}'")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if id(node) in doc_ids:
                    continue
                if node.value == "use_nodes":
                    violations.append(f"{rel_str}:{node.lineno}: forbidden string 'use_nodes'")
                for bad_shader in forbidden_builtins:
                    if bad_shader in node.value:
                        violations.append(f"{rel_str}:{node.lineno}: forbidden shader '{bad_shader}'")

    assert checked_files >= 10, f"Unexpectedly few production files checked: {checked_files}"
    assert not violations, "Blender 5.0+ API compliance violations:\n  " + "\n  ".join(violations)


def test_extension_lifecycle() -> None:
    """Suite 2: Verify register / unregister / re-register lifecycle cleanly."""
    if hasattr(bpy.types.Scene, "flightcheck"):
        flightcheck.unregister()

    expected_ops = (
        "scan",
        "fix_all",
        "cleanup",
        "clamp_pbr",
        "fix_colorspace",
        "fix_transforms",
        "equalize_texels",
    )
    expected_panels = (
        "FLIGHTCHECK_PT_main",
        "FLIGHTCHECK_PT_settings",
        "FLIGHTCHECK_PT_fixers",
        "FLIGHTCHECK_PT_issues",
    )

    def _assert_registered() -> None:
        assert hasattr(bpy.types.Scene, "flightcheck"), "Missing Scene.flightcheck"
        assert hasattr(bpy.types.WindowManager, "flightcheck_wm"), "Missing WindowManager.flightcheck_wm"
        assert hasattr(bpy.context.scene, "flightcheck"), "Scene instance missing flightcheck"
        assert hasattr(bpy.context.window_manager, "flightcheck_wm"), "WindowManager instance missing flightcheck_wm"
        for op_name in expected_ops:
            op = getattr(bpy.ops.flightcheck, op_name)
            rna = op.get_rna_type()
            assert rna is not None, f"Operator flightcheck.{op_name} not registered"
        for pt_name in expected_panels:
            assert hasattr(bpy.types, pt_name), f"Panel {pt_name} not registered"

    def _assert_unregistered() -> None:
        assert not hasattr(bpy.types.Scene, "flightcheck"), "Scene.flightcheck still registered"
        assert not hasattr(bpy.types.WindowManager, "flightcheck_wm"), "WindowManager.flightcheck_wm still registered"
        for pt_name in expected_panels:
            assert not hasattr(bpy.types, pt_name), f"Panel {pt_name} still registered"
        for op_name in expected_ops:
            assert not hasattr(bpy.types, f"FLIGHTCHECK_OT_{op_name}"), (
                f"Operator class FLIGHTCHECK_OT_{op_name} still registered"
            )

    flightcheck.register()
    _assert_registered()

    flightcheck.unregister()
    _assert_unregistered()

    flightcheck.register()
    _assert_registered()


def test_phase1_static_detection() -> None:
    """Suite 3: Verify static detection of all PBR, colorspace, scale, and texel anomalies."""
    _ensure_registered()
    scene = bpy.context.scene
    generate_test_scene.build_full_violation_scene(scene)
    _reset_scan_settings(scene)

    from flightcheck.core import run_full_scan

    report = run_full_scan(scene, write_texel_attr=True)
    assert report.status == FlightStatus.NO_GO, f"Expected NO-GO, got {report.status}"
    assert report.critical_count > 0, "Expected critical_count > 0"
    assert report.warning_count > 0, "Expected warning_count > 0"

    mat_issues = {(issue.code, issue.material_name): issue for issue in report.issues if issue.material_name}
    expected_material_issues = [
        ("DIELECTRIC_CRUSHED_BLACK", "Mat_CrushedBlack"),
        ("BLOWN_ALBEDO_WARN", "Mat_BlownAlbedoWarn"),
        ("BLOWN_ALBEDO_CRIT", "Mat_BlownAlbedoCrit"),
        ("DARK_METAL", "Mat_DarkMetal"),
        ("SEMI_METAL", "Mat_SemiMetal"),
        ("MIRROR_ROUGHNESS", "Mat_MirrorRough"),
        ("DEAD_ROUGHNESS", "Mat_DeadRough"),
        ("SPECULAR_OUTLIER", "Mat_SpecIOROutlier"),
        ("IOR_OUTLIER", "Mat_SpecIOROutlier"),
        ("DATA_MAP_SRGB", "Mat_BadColorspaceDataOnly"),
        ("DATA_MAP_SRGB", "Mat_BadColorspaceShared"),
    ]
    for code, mat_name in expected_material_issues:
        assert (code, mat_name) in mat_issues, f"Missing expected material issue ({code}, {mat_name})"

    cs_data_only = mat_issues[("DATA_MAP_SRGB", "Mat_BadColorspaceDataOnly")]
    assert cs_data_only.image_name == "Img_Rough_sRGB"
    assert cs_data_only.is_shared_image is False

    cs_shared = mat_issues[("DATA_MAP_SRGB", "Mat_BadColorspaceShared")]
    assert cs_shared.image_name == "Img_Shared_sRGB"
    assert cs_shared.is_shared_image is True

    obj_issues = {(issue.code, issue.object_name) for issue in report.issues if issue.object_name}
    expected_object_issues = [
        ("SCALE_NON_UNIFORM", "Obj_NonUniformScale"),
        ("SCALE_INVERTED", "Obj_InvertedScale"),
        ("SCALE_UNAPPLIED", "Obj_MultiUserA"),
        ("SCALE_NON_UNIFORM", "Obj_MultiUserB"),
        ("DIMENSION_TOO_SMALL", "Obj_DimTiny"),
        ("DIMENSION_TOO_LARGE", "Obj_DimHuge"),
        ("TEXEL_LOW", "Obj_TexelLow"),
        ("TEXEL_HIGH", "Obj_TexelHigh"),
    ]
    for code, obj_name in expected_object_issues:
        assert (code, obj_name) in obj_issues, f"Missing expected object issue ({code}, {obj_name})"

    healthy_obj_issues = [i for i in report.issues if i.object_name == "Obj_Healthy"]
    assert not healthy_obj_issues, f"Obj_Healthy reported unexpected issues: {healthy_obj_issues}"
    healthy_mat_issues = [i for i in report.issues if i.material_name == "Mat_Healthy"]
    assert not healthy_mat_issues, f"Mat_Healthy reported unexpected issues: {healthy_mat_issues}"
    assert report.scale_info["Obj_Healthy"].severity == Severity.OK
    assert report.texel_stats["Obj_Healthy"].severity == Severity.OK
    assert report.texel_stats["Obj_TexelRef1m"].severity == Severity.OK


def test_phase2_shader_splicing() -> None:
    """Suite 4: Verify non-destructive shader splicing, O(1) mode switching, and clean unsplice."""
    _ensure_registered()
    scene = bpy.context.scene
    generate_test_scene.build_full_violation_scene(scene)
    _reset_scan_settings(scene)

    from flightcheck.core import pbr_shader_nodes, shader_splicer

    pre_snapshot = _snapshot_materials()
    assert len(pre_snapshot) > 0, "Expected materials in test scene"

    spliced_count = shader_splicer.splice_scene_materials(scene, selected_only=False)
    assert spliced_count >= len(pre_snapshot), (
        f"Expected at least {len(pre_snapshot)} spliced materials, got {spliced_count}"
    )

    pbr_shader_nodes.set_ctrl_mode(constants.MODE_ALBEDO)
    assert constants.CTRL_GROUP_NAME in bpy.data.node_groups, "Missing __FLIGHTCHECK_CTRL__ node group"
    assert constants.OVERLAY_GROUP_NAME in bpy.data.node_groups, "Missing __FLIGHTCHECK_OVERLAY__ node group"

    for mat in bpy.data.materials:
        nt = _ensure_material_nodes(mat)
        assert nt is not None
        assert shader_splicer.is_material_spliced(mat), f"Material {mat.name} not marked spliced"
        assert constants.CTRL_NODE_NAME in nt.nodes, f"Missing CTRL node in {mat.name}"
        assert constants.OVERLAY_NODE_NAME in nt.nodes, f"Missing OVERLAY node in {mat.name}"

        out_node = _find_active_output_node(nt)
        assert out_node is not None
        surf = out_node.inputs.get("Surface")
        assert surf is not None and surf.is_linked
        assert surf.links[0].from_node.name == constants.OVERLAY_NODE_NAME

    ctrl_ng = bpy.data.node_groups[constants.CTRL_GROUP_NAME]
    mode_val_node = ctrl_ng.nodes.get("ModeValue")
    assert mode_val_node is not None, "Missing ModeValue node inside __FLIGHTCHECK_CTRL__"

    for mode_int in (0, 1, 2, 3, 4):
        pbr_shader_nodes.set_ctrl_mode(mode_int)
        assert pbr_shader_nodes.get_ctrl_mode() == mode_int
        assert abs(float(mode_val_node.outputs[0].default_value) - float(mode_int)) <= 1e-6

    unspliced_count = shader_splicer.unsplice_scene_materials(scene)
    assert unspliced_count >= len(pre_snapshot)

    for mat in bpy.data.materials:
        nt = _ensure_material_nodes(mat)
        assert not shader_splicer.is_material_spliced(mat), f"Material {mat.name} still spliced"
        leftover = [n.name for n in nt.nodes if n.name.startswith(constants.PREFIX)]
        assert not leftover, f"Leftover FlightCheck nodes in {mat.name}: {leftover}"

    post_snapshot = _snapshot_materials()
    _assert_material_snapshots_equal(pre_snapshot, post_snapshot)
    pbr_shader_nodes.remove_flightcheck_node_groups()


def test_phase3_texel_density() -> None:
    """Suite 5: Verify exact texel density computation, attribute writing, and cleanup."""
    _ensure_registered()
    scene = bpy.context.scene
    generate_test_scene.build_full_violation_scene(scene)
    _reset_scan_settings(scene)

    from flightcheck.core import texel_analyzer

    stats_map, _issues = texel_analyzer.scan_scene_texels(
        scene,
        target_px_per_m=1024.0,
        default_tex_size=1024,
        selected_only=False,
        write_attribute=True,
    )

    ref_stats = stats_map["Obj_TexelRef1m"]
    assert abs(ref_stats.avg_density_px_per_m - 1024.0) <= 1.0, (
        f"Expected Obj_TexelRef1m density ~1024.0, got {ref_stats.avg_density_px_per_m}"
    )

    ref_mesh = bpy.data.objects["Obj_TexelRef1m"].data
    ref_attr = ref_mesh.attributes.get(constants.TEXEL_ATTR_NAME)
    assert ref_attr is not None, "Missing __flightcheck_td attribute on Obj_TexelRef1m"
    assert ref_attr.domain == "CORNER", f"Unexpected attribute domain: {ref_attr.domain}"
    assert ref_attr.data_type == "FLOAT", f"Unexpected attribute data_type: {ref_attr.data_type}"
    ref_vals = [0.0] * len(ref_attr.data)
    ref_attr.data.foreach_get("value", ref_vals)
    assert len(ref_vals) == 24
    assert all(abs(v - 0.50) <= 1e-3 for v in ref_vals), f"Expected ~0.50 face-corner values, got {ref_vals}"

    low_stats = stats_map["Obj_TexelLow"]
    assert abs(low_stats.avg_density_px_per_m - 256.0) <= 1.0, (
        f"Expected Obj_TexelLow density ~256.0, got {low_stats.avg_density_px_per_m}"
    )
    low_attr = bpy.data.objects["Obj_TexelLow"].data.attributes.get(constants.TEXEL_ATTR_NAME)
    assert low_attr is not None
    low_vals = [0.0] * len(low_attr.data)
    low_attr.data.foreach_get("value", low_vals)
    expected_low_encoded = constants.normalize_texel_ratio(256.0 / 1024.0)
    assert all(abs(v - expected_low_encoded) <= 1e-3 for v in low_vals)

    high_stats = stats_map["Obj_TexelHigh"]
    assert abs(high_stats.avg_density_px_per_m - 3072.0) <= 1.0, (
        f"Expected Obj_TexelHigh density ~3072.0, got {high_stats.avg_density_px_per_m}"
    )
    high_attr = bpy.data.objects["Obj_TexelHigh"].data.attributes.get(constants.TEXEL_ATTR_NAME)
    assert high_attr is not None
    high_vals = [0.0] * len(high_attr.data)
    high_attr.data.foreach_get("value", high_vals)
    expected_high_encoded = constants.normalize_texel_ratio(3072.0 / 1024.0)
    assert all(abs(v - expected_high_encoded) <= 1e-3 for v in high_vals)

    removed = texel_analyzer.remove_texel_attributes(scene)
    assert removed > 0
    for obj in scene.objects:
        if obj.type == "MESH":
            assert constants.TEXEL_ATTR_NAME not in obj.data.attributes, (
                f"Attribute {constants.TEXEL_ATTR_NAME} remained on {obj.name}"
            )


def test_phase4_hud_and_3d_overlays() -> None:
    """Suite 6: Verify HUD layout scaling, 3D cage geometry, <1.5ms budget, and handler idempotency."""
    _ensure_registered()
    scene = bpy.context.scene
    generate_test_scene.build_full_violation_scene(scene)
    _reset_scan_settings(scene)

    from flightcheck.core import run_full_scan
    from flightcheck.hud import overlay_2d, overlay_3d, unregister_all_draw_handlers

    report = run_full_scan(scene, write_texel_attr=True)
    corners = ("TOP_LEFT", "TOP_RIGHT", "BOTTOM_LEFT", "BOTTOM_RIGHT")

    for ui_scale in (0.0, 1.0, 2.0):
        for corner in corners:
            for mode_int in (0, 1, 2, 3, 4):
                layout = overlay_2d.compute_hud_layout(1920, 1080, ui_scale, corner, report, mode_int)
                assert layout["w"] > 0 and layout["h"] > 0
                assert layout["x"] >= 0.0 and layout["x"] + layout["w"] <= 1920.0
                assert layout["y"] >= 0.0 and layout["y"] + layout["h"] <= 1080.0

    l_zero = overlay_2d.compute_hud_layout(1920, 1080, 0.0, "TOP_LEFT", report, constants.MODE_ALBEDO)
    l_one = overlay_2d.compute_hud_layout(1920, 1080, 1.0, "TOP_LEFT", report, constants.MODE_ALBEDO)
    l_two = overlay_2d.compute_hud_layout(1920, 1080, 2.0, "TOP_LEFT", report, constants.MODE_ALBEDO)
    assert abs(l_zero["w"] - l_one["w"]) <= 1e-5 and abs(l_zero["h"] - l_one["h"]) <= 1e-5
    assert l_two["w"] > l_one["w"] and l_two["h"] > l_one["h"]

    segments = overlay_3d.build_cage_segments(scene, report)
    for key in ("crit", "warn", "dim"):
        assert key in segments, f"Missing '{key}' cage segment key"
        assert len(segments[key]) >= 24, f"Expected non-empty '{key}' cage vertices, got {len(segments[key])}"
        assert len(segments[key]) % 2 == 0, f"Odd vertex count in '{key}' line segments"

    iterations = 50
    t0 = time.perf_counter()
    for _ in range(iterations):
        overlay_2d.compute_hud_layout(1920, 1080, 1.0, "TOP_LEFT", report, constants.MODE_COMBINED)
        overlay_3d.build_cage_segments(scene, report)
    elapsed_ms = ((time.perf_counter() - t0) * 1000.0) / iterations
    assert elapsed_ms < 1.5, f"HUD + 3D cage geometry exceeded 1.5ms budget: {elapsed_ms:.3f}ms"

    try:
        overlay_2d.unregister_draw_handler()
        overlay_2d.register_draw_handler()
        overlay_2d.register_draw_handler()
        assert overlay_2d.is_handler_registered() is True
        overlay_2d.unregister_draw_handler()
        overlay_2d.unregister_draw_handler()
        assert overlay_2d.is_handler_registered() is False

        overlay_3d.unregister_draw_handler()
        overlay_3d.register_draw_handler()
        overlay_3d.register_draw_handler()
        assert overlay_3d.is_handler_registered() is True
        overlay_3d.unregister_draw_handler()
        overlay_3d.unregister_draw_handler()
        assert overlay_3d.is_handler_registered() is False
    finally:
        unregister_all_draw_handlers()


def _assert_outward_normals(obj: bpy.types.Object) -> None:
    mesh = obj.data
    assert len(mesh.polygons) > 0
    for poly in mesh.polygons:
        dot = poly.normal.dot(poly.center)
        assert dot > 0.0, f"Inward face normal on {obj.name} poly {poly.index}: dot={dot}"


def test_phase5_fixers_and_cleanup() -> None:
    """Suite 7: Verify shared image protection, NO-GO -> GO remediation, and lossless cleanup."""
    _ensure_registered()
    scene = bpy.context.scene

    from flightcheck.core import pbr_shader_nodes, run_full_scan, shader_splicer, texel_analyzer
    from flightcheck.fixers import clamp_pbr, equalize_texels, fix_colorspace, fix_transforms
    from flightcheck.hud import unregister_all_draw_handlers

    # 1. Shared image protection check on full violation scene
    generate_test_scene.build_full_violation_scene(scene)
    _reset_scan_settings(scene)
    cs_res = fix_colorspace.fix_scene_colorspaces(scene, selected_only=False)
    assert bpy.data.images["Img_Rough_sRGB"].colorspace_settings.name == "Non-Color"
    assert "Img_Rough_sRGB" in cs_res["fixed"]
    assert bpy.data.images["Img_Shared_sRGB"].colorspace_settings.name == "sRGB"
    assert "Img_Shared_sRGB" in cs_res["protected_shared"]

    # 2. Direct fixer pipeline on remediable scene: NO-GO -> GO
    generate_test_scene.build_remediable_scene(scene)
    _reset_scan_settings(scene)
    report_before = run_full_scan(scene, write_texel_attr=True)
    assert report_before.status == FlightStatus.NO_GO, f"Expected NO-GO before fixers, got {report_before.status}"

    fix_transforms.fix_scene_transforms(scene, selected_only=False)
    fix_colorspace.fix_scene_colorspaces(scene, selected_only=False)
    clamp_pbr.clamp_scene_pbr(scene, selected_only=False)
    equalize_texels.equalize_scene_texels(
        scene,
        target_px_per_m=scene.flightcheck.target_texel_density,
        default_tex_size=scene.flightcheck.default_texture_size,
        selected_only=False,
    )

    report_after = run_full_scan(scene, write_texel_attr=True)
    assert report_after.status == FlightStatus.GO, (
        f"Expected GO after fixers, got {report_after.status}; issues={report_after.issues}"
    )
    assert report_after.critical_count == 0, f"Expected 0 critical issues, got {report_after.issues}"
    assert report_after.warning_count == 0, f"Expected 0 warning issues, got {report_after.issues}"

    obj_a = bpy.data.objects["Obj_MultiUserA"]
    obj_b = bpy.data.objects["Obj_MultiUserB"]
    obj_inv = bpy.data.objects["Obj_InvertedScale"]
    obj_nonuni = bpy.data.objects["Obj_NonUniformScale"]
    assert obj_a.data != obj_b.data, "Expected MultiUserA and MultiUserB meshes to be split"

    for obj in (obj_a, obj_b, obj_inv, obj_nonuni):
        assert max(abs(float(s) - 1.0) for s in obj.scale) <= 1e-4, f"{obj.name} scale not 1.0: {tuple(obj.scale)}"
        _assert_outward_normals(obj)

    # 3. Operator fix_all() + splice + cleanup_all(remove_clamps=True) + attribute cleanup
    generate_test_scene.build_remediable_scene(scene)
    _reset_scan_settings(scene)
    pre_snapshot = _snapshot_materials()

    op_res = bpy.ops.flightcheck.fix_all()
    assert op_res == {"FINISHED"}, f"fix_all operator failed: {op_res}"

    op_report = run_full_scan(scene, write_texel_attr=True)
    assert op_report.status == FlightStatus.GO, (
        f"Expected GO after fix_all(), got {op_report.status}; issues={op_report.issues}"
    )
    assert op_report.critical_count == 0
    assert op_report.warning_count == 0

    shader_splicer.splice_scene_materials(scene, selected_only=False)
    pbr_shader_nodes.set_ctrl_mode(constants.MODE_COMBINED)

    shader_splicer.cleanup_all(remove_clamps=True)
    texel_analyzer.remove_texel_attributes(scene)
    unregister_all_draw_handlers()

    leftover_groups = [ng.name for ng in bpy.data.node_groups if ng.name.startswith(constants.PREFIX)]
    assert not leftover_groups, f"Leftover FlightCheck node groups: {leftover_groups}"

    for mat in bpy.data.materials:
        nt = _ensure_material_nodes(mat)
        if nt is None:
            continue
        leftover_nodes = [n.name for n in nt.nodes if n.name.startswith(constants.PREFIX)]
        assert not leftover_nodes, f"Leftover FlightCheck nodes in {mat.name}: {leftover_nodes}"

    for obj in scene.objects:
        if obj.type == "MESH":
            assert constants.TEXEL_ATTR_NAME not in obj.data.attributes, (
                f"Leftover {constants.TEXEL_ATTR_NAME} attribute on {obj.name}"
            )

    post_snapshot = _snapshot_materials()
    _assert_material_snapshots_equal(pre_snapshot, post_snapshot)

    cleanup_op_res = bpy.ops.flightcheck.cleanup(remove_clamps=True)
    assert cleanup_op_res == {"FINISHED"}
    CACHE.clear()


TESTS: List[Tuple[str, Callable[[], None]]] = [
    ("1. Blender 5.0+ Compliance", test_blender5_compliance),
    ("2. Extension Lifecycle", test_extension_lifecycle),
    ("3. Phase 1 Static Detection", test_phase1_static_detection),
    ("4. Phase 2 Shader Splicing", test_phase2_shader_splicing),
    ("5. Phase 3 Texel Density", test_phase3_texel_density),
    ("6. Phase 4 HUD & 3D Overlays", test_phase4_hud_and_3d_overlays),
    ("7. Phase 5 Fixers & Cleanup", test_phase5_fixers_and_cleanup),
]


def main() -> int:
    passed = 0
    failed = 0
    print("=" * 72)
    print(f"FlightCheck Verification Suite (Blender {bpy.app.version_string})")
    print("=" * 72)

    for label, fn in TESTS:
        t0 = time.perf_counter()
        try:
            fn()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[PASS] {label} ({dt_ms:.1f} ms)")
            passed += 1
        except Exception:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[FAIL] {label} ({dt_ms:.1f} ms)")
            traceback.print_exc()
            failed += 1

    print("=" * 72)
    print(f"Summary: {passed} passed, {failed} failed (total {len(TESTS)})")
    print("=" * 72)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
