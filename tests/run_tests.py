"""Headless CLI verification harness for FlightCheck (Blender 5.0+ extension).

Runs 8 diagnostic test suites covering Blender 5.0+ API compliance, extension
lifecycle, static detection, shader splicing, texel density, HUD/3D overlays,
automated fixers, non-destructive cleanup, and P0-P3 bug regressions.
"""

from __future__ import annotations

import ast
import inspect
import sys
import time
import tomllib
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import bmesh
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
    fc.equalize_mode = "RELATIVE"
    fc.auto_rescan = True
    fc.clamp_dielectric_knee = constants.CLAMP_DIELECTRIC_KNEE
    fc.clamp_dielectric_max = constants.CLAMP_DIELECTRIC_MAX
    wm = getattr(bpy.context, "window_manager", None)
    if wm is not None and hasattr(wm, "flightcheck_wm"):
        wm.flightcheck_wm.hud_style = "COMPACT"
        wm.flightcheck_wm.hud_position = "TOP_LEFT"
        wm.flightcheck_wm.active_issue_filter = "ALL"


def _get_enabled_socket(sockets, name: str, socket_type: str | None = None):
    for sock in sockets:
        if sock.name == name and getattr(sock, "enabled", True):
            if socket_type is None or sock.type == socket_type:
                return sock
    return sockets.get(name)


def _snapshot_uvs(objects) -> Dict[str, Tuple[int, Dict[Tuple[int, int], Tuple[float, float]]]]:
    snap: Dict[str, Tuple[int, Dict[Tuple[int, int], Tuple[float, float]]]] = {}
    for obj in objects:
        if obj.type != "MESH" or obj.data is None:
            continue
        mesh = obj.data
        uv_layer = mesh.uv_layers.active
        if uv_layer is None:
            continue
        per_corner = {
            (poly.index, mesh.loops[li].vertex_index): (
                float(uv_layer.data[li].uv[0]),
                float(uv_layer.data[li].uv[1]),
            )
            for poly in mesh.polygons
            for li in poly.loop_indices
        }
        snap[obj.name] = (len(uv_layer.data), per_corner)
    return snap


def _assert_uv_snapshots_equal(
    before: Dict[str, Tuple[int, Dict[Tuple[int, int], Tuple[float, float]]]],
    after: Dict[str, Tuple[int, Dict[Tuple[int, int], Tuple[float, float]]]],
) -> None:
    assert set(before.keys()) == set(after.keys()), (
        f"UV object keys changed: before={sorted(before.keys())}, after={sorted(after.keys())}"
    )
    for obj_name, (pre_len, pre_map) in before.items():
        post_len, post_map = after[obj_name]
        assert pre_len == post_len, (
            f"UV loop count changed on {obj_name}: {pre_len} vs {post_len}"
        )
        assert set(pre_map.keys()) == set(post_map.keys()), (
            f"UV topology keys changed on {obj_name}"
        )
        max_diff = max(
            (
                max(abs(pre_map[k][0] - post_map[k][0]), abs(pre_map[k][1] - post_map[k][1]))
                for k in pre_map
            ),
            default=0.0,
        )
        assert max_diff <= 1e-6, f"UVs unexpectedly mutated on {obj_name}: max_diff={max_diff}"


def _invoke_handler(fn: Callable[..., Any], *args: Any) -> None:
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    if has_varargs:
        fn(*args[:1])
        return
    pos_params = [
        p for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if not pos_params:
        fn()
    else:
        fn(*args[: len(pos_params)])


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
    exclude_patterns = manifest.get("build", {}).get("paths_exclude_pattern", [])
    assert any("tests/" in str(pat) for pat in exclude_patterns), (
        f"Missing 'tests/' in [build].paths_exclude_pattern: {exclude_patterns}"
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
    hud_styles = ("COMPACT", "FULL", "MINIMAL")

    for ui_scale in (0.0, 0.8, 1.0, 2.0):
        for corner in corners:
            for hud_style in hud_styles:
                for mode_int in (0, 1, 2, 3, 4):
                    layout = overlay_2d.compute_hud_layout(
                        1920,
                        1080,
                        ui_scale,
                        corner,
                        report,
                        mode_int,
                        hud_style=hud_style,
                    )
                    assert layout["w"] > 0 and layout["h"] > 0
                    assert layout["x"] >= 0.0 and layout["x"] + layout["w"] <= 1920.0
                    assert layout["y"] >= 0.0 and layout["y"] + layout["h"] <= 1080.0
                    assert layout.get("style") == hud_style

    l_zero = overlay_2d.compute_hud_layout(1920, 1080, 0.0, "TOP_LEFT", report, constants.MODE_ALBEDO)
    l_small = overlay_2d.compute_hud_layout(1920, 1080, 0.8, "TOP_LEFT", report, constants.MODE_ALBEDO)
    l_one = overlay_2d.compute_hud_layout(1920, 1080, 1.0, "TOP_LEFT", report, constants.MODE_ALBEDO)
    l_two = overlay_2d.compute_hud_layout(1920, 1080, 2.0, "TOP_LEFT", report, constants.MODE_ALBEDO)
    assert abs(l_zero["w"] - l_one["w"]) <= 1e-5 and abs(l_zero["h"] - l_one["h"]) <= 1e-5
    assert abs(l_small["scale"] - 0.8) <= 1e-5
    assert l_small["w"] < l_one["w"] and l_small["h"] < l_one["h"]
    assert l_two["w"] > l_one["w"] and l_two["h"] > l_one["h"]

    l_min = overlay_2d.compute_hud_layout(
        1920, 1080, 1.0, "TOP_LEFT", report, constants.MODE_ALBEDO, hud_style="MINIMAL"
    )
    l_full = overlay_2d.compute_hud_layout(
        1920, 1080, 1.0, "TOP_LEFT", report, constants.MODE_ALBEDO, hud_style="FULL"
    )
    assert l_min["h"] < l_full["h"]

    l_stale = overlay_2d.compute_hud_layout(
        1920,
        1080,
        1.0,
        "TOP_RIGHT",
        report,
        constants.MODE_ALBEDO,
        is_stale=True,
        solid_mode_hint=True,
        hud_style="FULL",
        margin_right_extra=120.0,
    )
    l_no_extra = overlay_2d.compute_hud_layout(
        1920,
        1080,
        1.0,
        "TOP_RIGHT",
        report,
        constants.MODE_ALBEDO,
        is_stale=True,
        solid_mode_hint=True,
        hud_style="FULL",
        margin_right_extra=0.0,
    )
    assert l_stale.get("is_stale") is True
    assert l_stale.get("stale_rect") is not None
    assert l_stale.get("solid_hint_rect") is not None
    assert l_stale["x"] < l_no_extra["x"]

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
        mode="ABSOLUTE",
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

    # 3. Operator fix_all() preserves UVs by default; fix_all(include_texels=True) transitions NO-GO -> GO
    generate_test_scene.build_remediable_scene(scene)
    _reset_scan_settings(scene)
    pre_snapshot = _snapshot_materials()
    uv_before = _snapshot_uvs(scene.objects)

    op_res = bpy.ops.flightcheck.fix_all()
    assert op_res == {"FINISHED"}, f"fix_all operator failed: {op_res}"
    _assert_uv_snapshots_equal(uv_before, _snapshot_uvs(scene.objects))

    scene.flightcheck.equalize_mode = "ABSOLUTE"
    op_res_tex = bpy.ops.flightcheck.fix_all(include_texels=True)
    assert op_res_tex == {"FINISHED"}, f"fix_all(include_texels=True) failed: {op_res_tex}"

    op_report = run_full_scan(scene, write_texel_attr=True)
    assert op_report.status == FlightStatus.GO, (
        f"Expected GO after fix_all(include_texels=True), got {op_report.status}; issues={op_report.issues}"
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


def test_bug_regressions() -> None:
    """Suite 8: Verify all P0-P3 bug-review regressions end-to-end."""
    _ensure_registered()
    scene = bpy.context.scene

    from flightcheck import state
    from flightcheck.core import (
        graph_crawler,
        pbr_shader_nodes,
        run_full_scan,
        scale_analyzer,
        shader_splicer,
        texel_analyzer,
    )
    from flightcheck.fixers import clamp_pbr, equalize_texels, fix_colorspace, fix_transforms
    from flightcheck.hud import unregister_all_draw_handlers

    def _fresh_scene() -> None:
        _ensure_registered()
        generate_test_scene.clear_scene(scene)
        _reset_scan_settings(scene)
        pbr_shader_nodes.remove_flightcheck_node_groups()
        unregister_all_draw_handlers()
        CACHE.clear()

    extra_scene: bpy.types.Scene | None = None
    try:
        # P0-1: render_pre/render_complete toggle CTRL mode; save_pre/save_post strip & restore overlays + __flightcheck_td
        _fresh_scene()
        mat_p01, _, _, _ = generate_test_scene._create_principled_material("Mat_P01")
        obj_p01 = generate_test_scene._link_cube_object(
            scene,
            "Obj_P01",
            generate_test_scene._create_cube_mesh("Mesh_P01", size=1.0, uv_scale=1.0),
            material=mat_p01,
        )
        scene.flightcheck.enabled = True
        scene.flightcheck.diagnostic_mode = "ALBEDO"
        assert shader_splicer.is_material_spliced(mat_p01), "[P0-1] Expected material spliced"
        assert pbr_shader_nodes.get_ctrl_mode() == constants.MODE_ALBEDO, "[P0-1] Expected MODE_ALBEDO"
        assert constants.TEXEL_ATTR_NAME in obj_p01.data.attributes, "[P0-1] Expected __flightcheck_td attribute"

        render_pre_cbs = [fn for fn in bpy.app.handlers.render_pre if getattr(fn, "__module__", "").startswith("flightcheck")]
        render_complete_cbs = [fn for fn in bpy.app.handlers.render_complete if getattr(fn, "__module__", "").startswith("flightcheck")]
        save_pre_cbs = [fn for fn in bpy.app.handlers.save_pre if getattr(fn, "__module__", "").startswith("flightcheck")]
        save_post_cbs = [fn for fn in bpy.app.handlers.save_post if getattr(fn, "__module__", "").startswith("flightcheck")]
        assert render_pre_cbs and render_complete_cbs, "[P0-1] Missing render_pre/render_complete handlers"
        assert save_pre_cbs and save_post_cbs, "[P0-1] Missing save_pre/save_post handlers"

        for cb in render_pre_cbs:
            _invoke_handler(cb, scene)
        assert pbr_shader_nodes.get_ctrl_mode() == constants.MODE_OFF, "[P0-1] CTRL mode not reset to 0 on render_pre"
        for cb in render_complete_cbs:
            _invoke_handler(cb, scene)
        assert pbr_shader_nodes.get_ctrl_mode() == constants.MODE_ALBEDO, "[P0-1] CTRL mode not restored on render_complete"

        for cb in save_pre_cbs:
            _invoke_handler(cb, "")
        assert not shader_splicer.is_material_spliced(mat_p01), "[P0-1] Material still spliced after save_pre"
        assert constants.TEXEL_ATTR_NAME not in obj_p01.data.attributes, "[P0-1] __flightcheck_td still present after save_pre"
        for cb in save_post_cbs:
            _invoke_handler(cb, "")
        assert shader_splicer.is_material_spliced(mat_p01), "[P0-1] Material not restored on save_post"
        assert pbr_shader_nodes.get_ctrl_mode() == constants.MODE_ALBEDO, "[P0-1] CTRL mode not restored on save_post"
        assert constants.TEXEL_ATTR_NAME in obj_p01.data.attributes, "[P0-1] __flightcheck_td not restored on save_post"

        # P0-2: unregister() resets CTRL mode to 0 and unsplices all spliced materials
        scene.flightcheck.diagnostic_mode = "COMBINED"
        assert shader_splicer.is_material_spliced(mat_p01)
        assert pbr_shader_nodes.get_ctrl_mode() == constants.MODE_COMBINED
        flightcheck.unregister()
        try:
            assert pbr_shader_nodes.get_ctrl_mode() == constants.MODE_OFF, "[P0-2] unregister() did not reset CTRL mode to 0"
            assert not shader_splicer.is_material_spliced(mat_p01), "[P0-2] unregister() left material spliced"
        finally:
            _ensure_registered()

        # P0-3: Changing diagnostic_mode while enabled=False does NOT splice or create __FLIGHTCHECK_CTRL__
        _fresh_scene()
        mat_p03, _, _, _ = generate_test_scene._create_principled_material("Mat_P03")
        generate_test_scene._link_cube_object(
            scene,
            "Obj_P03",
            generate_test_scene._create_cube_mesh("Mesh_P03", size=1.0),
            material=mat_p03,
        )
        assert scene.flightcheck.enabled is False
        for mode_name in ("ALBEDO", "ROUGHNESS", "TEXEL_DENSITY", "COMBINED", "OFF"):
            scene.flightcheck.diagnostic_mode = mode_name
            assert constants.CTRL_GROUP_NAME not in bpy.data.node_groups, (
                f"[P0-3] Setting diagnostic_mode={mode_name} while disabled created __FLIGHTCHECK_CTRL__"
            )
            assert not shader_splicer.is_material_spliced(mat_p03), (
                f"[P0-3] Setting diagnostic_mode={mode_name} while disabled spliced material"
            )

        # P0-4: Child parented with keep-transform (matrix_parent_inverse) keeps exact matrix_world
        _fresh_scene()
        mat_p04, _, _, _ = generate_test_scene._create_principled_material("Mat_P04")
        parent_obj = generate_test_scene._link_cube_object(
            scene,
            "Obj_P04_Parent",
            generate_test_scene._create_cube_mesh("Mesh_P04_Parent", size=1.0),
            material=mat_p04,
            location=(2.0, -1.0, 3.0),
            scale=(1.0, 2.0, 0.5),
        )
        parent_obj.rotation_euler = (0.1, 0.25, -0.15)
        child_obj = generate_test_scene._link_cube_object(
            scene,
            "Obj_P04_Child",
            generate_test_scene._create_cube_mesh("Mesh_P04_Child", size=1.0),
            material=mat_p04,
            location=(3.0, 4.0, -2.0),
            scale=(1.0, 1.0, 1.0),
        )
        child_obj.rotation_euler = (0.2, -0.3, 0.4)
        bpy.context.view_layer.update()
        expected_world = child_obj.matrix_world.copy()
        child_obj.parent = parent_obj
        child_obj.matrix_parent_inverse = parent_obj.matrix_world.inverted()
        bpy.context.view_layer.update()

        fix_transforms.fix_scene_transforms(scene, selected_only=False)
        bpy.context.view_layer.update()
        world_diff = max(
            abs(child_obj.matrix_world[r][c] - expected_world[r][c])
            for r in range(4)
            for c in range(4)
        )
        assert world_diff < 1e-5, f"[P0-4] Child matrix_world drifted by {world_diff}"
        assert max(abs(float(s) - 1.0) for s in child_obj.scale) < 1e-5, (
            f"[P0-4] Child scale changed: {tuple(child_obj.scale)}"
        )
        assert max(abs(float(s) - 1.0) for s in parent_obj.scale) < 1e-5, (
            f"[P0-4] Parent scale not baked: {tuple(parent_obj.scale)}"
        )

        # P0-5: Keyframed scale (Slotted Action) + driven scale -> fixable=False and skipped by fix_scene_transforms
        _fresh_scene()
        obj_anim = generate_test_scene._link_cube_object(
            scene,
            "Obj_P05_Anim",
            generate_test_scene._create_cube_mesh("Mesh_P05_Anim", size=1.0),
            scale=(2.0, 2.0, 2.0),
        )
        obj_anim.keyframe_insert(data_path="scale", frame=1)
        obj_drv = generate_test_scene._link_cube_object(
            scene,
            "Obj_P05_Driven",
            generate_test_scene._create_cube_mesh("Mesh_P05_Driven", size=1.0),
            scale=(1.5, 1.5, 1.5),
        )
        fcurve_drv = obj_drv.driver_add("scale", 0)
        fcurve_drv.driver.expression = "1.5"
        bpy.context.view_layer.update()

        _, anim_issues = scale_analyzer.analyze_object_scale(obj_anim)
        _, drv_issues = scale_analyzer.analyze_object_scale(obj_drv)
        assert anim_issues and all(not i.fixable for i in anim_issues), (
            f"[P0-5] Animated scale issues should be fixable=False: {anim_issues}"
        )
        assert drv_issues and all(not i.fixable for i in drv_issues), (
            f"[P0-5] Driven scale issues should be fixable=False: {drv_issues}"
        )
        tx_p05 = fix_transforms.fix_scene_transforms(scene, selected_only=False)
        assert tx_p05.get("skipped_animated", 0) >= 2, f"[P0-5] Expected skipped_animated >= 2, got {tx_p05}"
        assert abs(obj_anim.scale[0] - 2.0) < 1e-4, "[P0-5] Animated scale object was modified"
        assert abs(obj_drv.scale[0] - 1.5) < 1e-4, "[P0-5] Driven scale object was modified"

        # P0-6: fix_all() preserves UVs; RELATIVE keeps UVs in [0,1]; ABSOLUTE on 10m cube exceeds [0,1]
        _fresh_scene()
        img_p06 = generate_test_scene._create_filled_image("Img_P06", 1024, 1024)
        mat_p06, nt_p06, bsdf_p06, _ = generate_test_scene._create_principled_material("Mat_P06")
        generate_test_scene._attach_image_node(nt_p06, bsdf_p06, img_p06, ("Base Color",))

        mesh_multi = bpy.data.meshes.new("Mesh_P06_Multi")
        bm_multi = bmesh.new()
        try:
            v0 = bm_multi.verts.new((0.0, 0.0, 0.0))
            v1 = bm_multi.verts.new((1.0, 0.0, 0.0))
            v2 = bm_multi.verts.new((1.0, 1.0, 0.0))
            v3 = bm_multi.verts.new((0.0, 1.0, 0.0))
            f0 = bm_multi.faces.new((v0, v1, v2, v3))

            v4 = bm_multi.verts.new((2.0, 0.0, 0.0))
            v5 = bm_multi.verts.new((3.0, 0.0, 0.0))
            v6 = bm_multi.verts.new((3.0, 1.0, 0.0))
            v7 = bm_multi.verts.new((2.0, 1.0, 0.0))
            f1 = bm_multi.faces.new((v4, v5, v6, v7))

            uv_lay = bm_multi.loops.layers.uv.verify()
            for idx, lp in enumerate(f0.loops):
                lp[uv_lay].uv = ((0.0, 0.0), (0.25, 0.0), (0.25, 0.25), (0.0, 0.25))[idx]
            for idx, lp in enumerate(f1.loops):
                lp[uv_lay].uv = ((0.0, 0.5), (1.5, 0.5), (1.5, 2.0), (0.0, 2.0))[idx]
            bm_multi.normal_update()
            bm_multi.to_mesh(mesh_multi)
        finally:
            bm_multi.free()
        mesh_multi.update()

        obj_multi = generate_test_scene._link_cube_object(scene, "Obj_P06_Multi", mesh_multi, material=mat_p06)
        obj_10m = generate_test_scene._link_cube_object(
            scene,
            "Obj_P06_10m",
            generate_test_scene._create_cube_mesh("Mesh_P06_10m", size=10.0, uv_scale=1.0),
            material=mat_p06,
            location=(15.0, 0.0, 0.0),
        )
        uv_pre_p06 = _snapshot_uvs([obj_multi, obj_10m])
        assert bpy.ops.flightcheck.fix_all() == {"FINISHED"}
        _assert_uv_snapshots_equal(uv_pre_p06, _snapshot_uvs([obj_multi, obj_10m]))

        res_rel = equalize_texels.equalize_scene_texels(
            scene,
            target_px_per_m=1024.0,
            default_tex_size=1024,
            mode="RELATIVE",
            pack_islands=True,
        )
        assert res_rel.get("uvs_exceed_unit_square", 0) == 0, f"[P0-6] RELATIVE exceeded [0,1]: {res_rel}"
        for check_obj in (obj_multi, obj_10m):
            assert not equalize_texels.check_uvs_exceed_unit_square(check_obj.data), (
                f"[P0-6] {check_obj.name} exceeded [0,1] in RELATIVE mode"
            )
            uv_data = check_obj.data.uv_layers.active.data
            assert all(
                -1e-4 <= d.uv[0] <= 1.0 + 1e-4 and -1e-4 <= d.uv[1] <= 1.0 + 1e-4
                for d in uv_data
            ), f"[P0-6] {check_obj.name} UVs escaped [0,1] in RELATIVE mode"

        res_abs = equalize_texels.equalize_scene_texels(
            scene,
            target_px_per_m=1024.0,
            default_tex_size=1024,
            mode="ABSOLUTE",
        )
        assert res_abs.get("uvs_exceed_unit_square", 0) > 0, (
            f"[P0-6] Expected ABSOLUTE on 10m cube to report uvs_exceed_unit_square > 0: {res_abs}"
        )
        assert equalize_texels.check_uvs_exceed_unit_square(obj_10m.data) is True

        # P0-7: Edit Mode poll() guards and safe analyzer/fixer execution (no ValueError: to_mesh())
        _fresh_scene()
        obj_p07 = generate_test_scene._link_cube_object(
            scene,
            "Obj_P07_Edit",
            generate_test_scene._create_cube_mesh("Mesh_P07_Edit", size=1.0, uv_scale=1.0),
            scale=(2.0, 2.0, 2.0),
        )
        bpy.context.view_layer.objects.active = obj_p07
        obj_p07.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        try:
            assert bpy.ops.flightcheck.fix_transforms.poll() is False, "[P0-7] fix_transforms.poll() True in Edit Mode"
            assert bpy.ops.flightcheck.equalize_texels.poll() is False, "[P0-7] equalize_texels.poll() True in Edit Mode"
            assert bpy.ops.flightcheck.fix_all.poll() is False, "[P0-7] fix_all.poll() True in Edit Mode"

            ed_stats, _ = texel_analyzer.analyze_object_texel_density(obj_p07, write_attribute=True)
            assert ed_stats.face_count == 6 and ed_stats.has_uvs is True, f"[P0-7] Bad edit-mode texel stats: {ed_stats}"

            tx_ed = fix_transforms.fix_scene_transforms(scene)
            assert tx_ed.get("skipped_edit", 0) >= 1, f"[P0-7] Expected skipped_edit >= 1, got {tx_ed}"
            assert abs(obj_p07.scale[0] - 2.0) < 1e-4

            eq_ed = equalize_texels.equalize_scene_texels(scene)
            assert isinstance(eq_ed, dict)
        finally:
            if obj_p07.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")

        # P1-1: flush_pending_rescan(scene) refreshes cached report and spliced overlay defaults (0.8 -> 1.0)
        _fresh_scene()
        mat_p11, nt_p11, bsdf_p11, _ = generate_test_scene._create_principled_material(
            "Mat_P11", base_color=(0.8, 0.8, 0.8, 1.0)
        )
        generate_test_scene._link_cube_object(
            scene,
            "Obj_P11",
            generate_test_scene._create_cube_mesh("Mesh_P11", size=1.0),
            material=mat_p11,
        )
        scene.flightcheck.enabled = True
        scene.flightcheck.auto_rescan = True
        scene.flightcheck.diagnostic_mode = "ALBEDO"
        rep_p11_pre = CACHE.get_report(scene)
        assert rep_p11_pre is not None and any(
            i.code == "BLOWN_ALBEDO_WARN" and i.material_name == "Mat_P11" for i in rep_p11_pre.issues
        ), f"[P1-1] Expected BLOWN_ALBEDO_WARN initially, got {rep_p11_pre}"

        _get_enabled_socket(bsdf_p11.inputs, "Base Color").default_value = (1.0, 1.0, 1.0, 1.0)
        CACHE.mark_dirty(scene)
        state.schedule_rescan(scene, delay=0.1)
        state.flush_pending_rescan(scene)

        rep_p11_post = CACHE.get_report(scene)
        assert rep_p11_post is not None and any(
            i.code == "BLOWN_ALBEDO_CRIT" and i.material_name == "Mat_P11" for i in rep_p11_post.issues
        ), f"[P1-1] Expected BLOWN_ALBEDO_CRIT after flush_pending_rescan, got {rep_p11_post}"
        overlay_p11 = nt_p11.nodes.get(constants.OVERLAY_NODE_NAME)
        assert overlay_p11 is not None
        ov_col = _get_enabled_socket(overlay_p11.inputs, "Base Color").default_value
        assert abs(float(ov_col[0]) - 1.0) < 1e-4, f"[P1-1] Overlay Base Color did not refresh: {tuple(ov_col)}"

        # P1-2: ShaderNodeMix pure-white -> BLOWN_ALBEDO_CRIT; ShaderNodeTexNoise -> UNEVALUABLE_INPUT (INFO, keeping GO)
        _fresh_scene()
        mat_mix, nt_mix, bsdf_mix, _ = generate_test_scene._create_principled_material("Mat_P12_Mix")
        mix_node = nt_mix.nodes.new("ShaderNodeMix")
        mix_node.data_type = "RGBA"
        mix_node.blend_type = "MIX"
        _get_enabled_socket(mix_node.inputs, "Factor", "VALUE").default_value = 1.0
        _get_enabled_socket(mix_node.inputs, "A", "RGBA").default_value = (0.2, 0.2, 0.2, 1.0)
        _get_enabled_socket(mix_node.inputs, "B", "RGBA").default_value = (1.0, 1.0, 1.0, 1.0)
        nt_mix.links.new(
            _get_enabled_socket(mix_node.outputs, "Result", "RGBA"),
            _get_enabled_socket(bsdf_mix.inputs, "Base Color"),
        )
        mix_issues = graph_crawler.analyze_material(mat_mix)
        assert any(i.code == "BLOWN_ALBEDO_CRIT" for i in mix_issues), (
            f"[P1-2] ShaderNodeMix white did not trigger BLOWN_ALBEDO_CRIT: {mix_issues}"
        )

        mat_noise, nt_noise, bsdf_noise, _ = generate_test_scene._create_principled_material("Mat_P12_Noise")
        noise_node = nt_noise.nodes.new("ShaderNodeTexNoise")
        nt_noise.links.new(
            _get_enabled_socket(noise_node.outputs, "Factor"),
            _get_enabled_socket(bsdf_noise.inputs, "Roughness"),
        )
        generate_test_scene._link_cube_object(
            scene,
            "Obj_P12_Noise",
            generate_test_scene._create_cube_mesh("Mesh_P12_Noise", size=1.0),
            material=mat_noise,
        )
        noise_report = run_full_scan(scene, write_texel_attr=True)
        assert any(
            i.code == "UNEVALUABLE_INPUT" and i.severity == Severity.INFO and i.material_name == "Mat_P12_Noise"
            for i in noise_report.issues
        ), f"[P1-2] Expected UNEVALUABLE_INPUT (INFO), got {noise_report.issues}"
        assert noise_report.status == FlightStatus.GO, (
            f"[P1-2] UNEVALUABLE_INPUT should keep status GO, got {noise_report.status}"
        )

        # P1-3: Image connected ONLY via Alpha output into BSDF.Alpha is NOT flagged DATA_MAP_SRGB
        _fresh_scene()
        img_alpha = generate_test_scene._create_filled_image(
            "Img_P13_AlphaOnly", 32, 32, rgba=(0.5, 0.5, 0.5, 0.8), colorspace="sRGB"
        )
        mat_alpha, nt_alpha, bsdf_alpha, _ = generate_test_scene._create_principled_material("Mat_P13_Alpha")
        tex_alpha = nt_alpha.nodes.new("ShaderNodeTexImage")
        tex_alpha.image = img_alpha
        nt_alpha.links.new(
            _get_enabled_socket(tex_alpha.outputs, "Alpha"),
            _get_enabled_socket(bsdf_alpha.inputs, "Alpha"),
        )
        alpha_issues = graph_crawler.analyze_material(mat_alpha)
        assert not any(i.code == "DATA_MAP_SRGB" for i in alpha_issues), (
            f"[P1-3] Alpha-only image falsely flagged DATA_MAP_SRGB: {alpha_issues}"
        )

        # P1-4: After clamp_material_pbr, metallic data map stays is_shared_image=False and fixable=True
        _fresh_scene()
        img_met = generate_test_scene._create_filled_image(
            "Img_P14_Metal", 32, 32, rgba=(1.0, 1.0, 1.0, 1.0), colorspace="sRGB"
        )
        mat_met, nt_met, bsdf_met, _ = generate_test_scene._create_principled_material(
            "Mat_P14_Metal", base_color=(0.15, 0.15, 0.15, 1.0)
        )
        generate_test_scene._attach_image_node(nt_met, bsdf_met, img_met, ("Metallic",))
        assert clamp_pbr.clamp_material_pbr(mat_met) is True
        assert shader_splicer.is_material_clamped(mat_met) is True
        met_issues = graph_crawler.analyze_material(mat_met)
        cs_met = [i for i in met_issues if i.code == "DATA_MAP_SRGB" and i.image_name == "Img_P14_Metal"]
        assert cs_met, f"[P1-4] Missing DATA_MAP_SRGB after clamp: {met_issues}"
        assert cs_met[0].is_shared_image is False and cs_met[0].fixable is True, (
            f"[P1-4] Metallic data map misclassified after clamp: {cs_met[0]}"
        )

        # P1-5: Child of negatively scaled Empty is SCALE_INVERTED (fixable=False); delta_scale detected & fixed
        _fresh_scene()
        empty_neg = bpy.data.objects.new("Empty_P15_Neg", None)
        empty_neg.scale = (-1.0, 1.0, 1.0)
        scene.collection.objects.link(empty_neg)
        obj_neg_child = generate_test_scene._link_cube_object(
            scene,
            "Obj_P15_NegChild",
            generate_test_scene._create_cube_mesh("Mesh_P15_NegChild", size=1.0),
            scale=(1.0, 1.0, 1.0),
        )
        obj_neg_child.parent = empty_neg
        obj_delta = generate_test_scene._link_cube_object(
            scene,
            "Obj_P15_Delta",
            generate_test_scene._create_cube_mesh("Mesh_P15_Delta", size=1.0),
            scale=(1.0, 1.0, 1.0),
        )
        obj_delta.delta_scale = (2.0, 1.0, 1.0)
        bpy.context.view_layer.update()

        _, neg_issues = scale_analyzer.analyze_object_scale(obj_neg_child)
        assert any(i.code == "SCALE_INVERTED" and i.fixable is False for i in neg_issues), (
            f"[P1-5] Expected SCALE_INVERTED fixable=False on neg-parent child: {neg_issues}"
        )
        _, delta_issues = scale_analyzer.analyze_object_scale(obj_delta)
        assert any(i.code == "SCALE_NON_UNIFORM" and i.fixable is True for i in delta_issues), (
            f"[P1-5] Expected SCALE_NON_UNIFORM fixable=True on delta_scale: {delta_issues}"
        )
        fix_transforms.fix_scene_transforms(scene)
        assert max(abs(float(s) - 1.0) for s in obj_delta.scale) < 1e-4
        assert max(abs(float(s) - 1.0) for s in obj_delta.delta_scale) < 1e-4

        # P1-6 & P1-8: Single (0,0,0) pixel in 64x64 mid-gray texture does NOT trigger DIELECTRIC_CRUSHED_BLACK;
        # compliant textured material is NOT clamped by clamp_material_pbr
        _fresh_scene()
        img_outlier = generate_test_scene._create_filled_image(
            "Img_P16_Outlier", 64, 64, rgba=(0.45, 0.45, 0.45, 1.0), colorspace="sRGB"
        )
        px_buf = [0.45, 0.45, 0.45, 1.0] * (64 * 64)
        px_buf[0:4] = [0.0, 0.0, 0.0, 1.0]
        img_outlier.pixels.foreach_set(px_buf)
        img_outlier.update()

        mat_outlier, nt_outlier, bsdf_outlier, _ = generate_test_scene._create_principled_material("Mat_P16_Outlier")
        generate_test_scene._attach_image_node(nt_outlier, bsdf_outlier, img_outlier, ("Base Color",))
        outlier_issues = graph_crawler.analyze_material(mat_outlier)
        assert not any(i.code == "DIELECTRIC_CRUSHED_BLACK" for i in outlier_issues), (
            f"[P1-6] Single dark pixel falsely triggered DIELECTRIC_CRUSHED_BLACK: {outlier_issues}"
        )
        assert not outlier_issues, f"[P1-6] Expected 0 issues on compliant outlier texture: {outlier_issues}"
        assert clamp_pbr.clamp_material_pbr(mat_outlier) is False, (
            "[P1-8] Compliant textured material should not be clamped"
        )
        assert shader_splicer.is_material_clamped(mat_outlier) is False

        # P1-9: soft_knee_upper clamping + property auto-clamping
        assert constants.soft_knee_upper(0.95, 0.75, 0.50) <= 0.50, "[P1-9] soft_knee_upper exceeded ceiling"
        scene.flightcheck.clamp_dielectric_max = 0.60
        scene.flightcheck.clamp_dielectric_knee = 0.75
        assert scene.flightcheck.clamp_dielectric_max >= scene.flightcheck.clamp_dielectric_knee + 0.01 - 1e-6, (
            "[P1-9] Setting knee > max did not auto-clamp max"
        )
        scene.flightcheck.clamp_dielectric_knee = 0.70
        scene.flightcheck.clamp_dielectric_max = 0.55
        assert scene.flightcheck.clamp_dielectric_knee <= scene.flightcheck.clamp_dielectric_max - 0.01 + 1e-6, (
            "[P1-9] Setting max < knee did not auto-clamp knee"
        )

        # P1-11: Two scenes hold independent reports in CACHE
        _fresh_scene()
        mat_s1, _, _, _ = generate_test_scene._create_principled_material("Mat_S1_Healthy")
        generate_test_scene._link_cube_object(
            scene,
            "Obj_S1_Healthy",
            generate_test_scene._create_cube_mesh("Mesh_S1_Healthy", size=1.0),
            material=mat_s1,
        )
        extra_scene = bpy.data.scenes.new("FlightCheck_Scene2")
        mat_s2, _, _, _ = generate_test_scene._create_principled_material(
            "Mat_S2_Crushed", base_color=(0.005, 0.005, 0.005, 1.0)
        )
        generate_test_scene._link_cube_object(
            extra_scene,
            "Obj_S2_Crushed",
            generate_test_scene._create_cube_mesh("Mesh_S2_Crushed", size=1.0),
            material=mat_s2,
        )
        run_full_scan(scene, write_texel_attr=True)
        run_full_scan(extra_scene, write_texel_attr=True)
        assert CACHE.get_report(scene).status == FlightStatus.GO, "[P1-11] Scene 1 report overwritten"
        assert CACHE.get_report(extra_scene).status == FlightStatus.NO_GO, "[P1-11] Scene 2 report invalid"
        bpy.data.scenes.remove(extra_scene, do_unlink=True)
        extra_scene = None

        # P2-1: Full scan of build_remediable_scene AFTER clamp_scene_pbr + equalize_scene_texels < 500 ms
        generate_test_scene.build_remediable_scene(scene)
        _reset_scan_settings(scene)
        fix_transforms.fix_scene_transforms(scene, selected_only=False)
        fix_colorspace.fix_scene_colorspaces(scene, selected_only=False)
        clamp_pbr.clamp_scene_pbr(scene, selected_only=False)
        equalize_texels.equalize_scene_texels(
            scene,
            target_px_per_m=scene.flightcheck.target_texel_density,
            default_tex_size=scene.flightcheck.default_texture_size,
            selected_only=False,
            mode="ABSOLUTE",
        )
        t0 = time.perf_counter()
        perf_report = run_full_scan(scene, write_texel_attr=True)
        perf_ms = (time.perf_counter() - t0) * 1000.0
        assert perf_report.status == FlightStatus.GO, f"[P2-1] Expected GO, got {perf_report.status}"
        assert perf_ms < 500.0, f"[P2-1] Remediated scene scan too slow: {perf_ms:.1f} ms"

        # P2-6 & P2-7: Diamond-graph CombineColor evaluates cleanly; stale flightcheck_version forces rebuild
        _fresh_scene()
        mat_cc, nt_cc, bsdf_cc, _ = generate_test_scene._create_principled_material("Mat_P26_Combine")
        val_node = nt_cc.nodes.new("ShaderNodeValue")
        val_node.outputs[0].default_value = 0.92
        cc_node = nt_cc.nodes.new("ShaderNodeCombineColor")
        for chan in ("Red", "Green", "Blue"):
            nt_cc.links.new(val_node.outputs[0], _get_enabled_socket(cc_node.inputs, chan))
        nt_cc.links.new(
            _get_enabled_socket(cc_node.outputs, "Color"),
            _get_enabled_socket(bsdf_cc.inputs, "Base Color"),
        )
        cc_issues = graph_crawler.analyze_material(mat_cc)
        assert any(i.code == "BLOWN_ALBEDO_CRIT" for i in cc_issues), (
            f"[P2-6] Diamond-graph CombineColor did not emit BLOWN_ALBEDO_CRIT: {cc_issues}"
        )

        for ensure_fn in (
            pbr_shader_nodes.ensure_ctrl_group,
            pbr_shader_nodes.ensure_overlay_group,
            pbr_shader_nodes.ensure_clamp_group,
        ):
            ng = ensure_fn()
            ng["flightcheck_version"] = 1
            rebuilt_ng = ensure_fn()
            assert rebuilt_ng.get("flightcheck_version") == constants.NODE_GROUP_VERSION, (
                f"[P2-7] Stale {rebuilt_ng.name} not rebuilt to v{constants.NODE_GROUP_VERSION}"
            )
    finally:
        if extra_scene is not None and extra_scene.name in bpy.data.scenes:
            bpy.data.scenes.remove(extra_scene, do_unlink=True)
        _ensure_registered()
        shader_splicer.cleanup_all(remove_clamps=True)
        texel_analyzer.remove_texel_attributes(scene)
        unregister_all_draw_handlers()
        generate_test_scene.clear_scene(scene)
        _reset_scan_settings(scene)
        CACHE.clear()


TESTS: List[Tuple[str, Callable[[], None]]] = [
    ("1. Blender 5.0+ Compliance", test_blender5_compliance),
    ("2. Extension Lifecycle", test_extension_lifecycle),
    ("3. Phase 1 Static Detection", test_phase1_static_detection),
    ("4. Phase 2 Shader Splicing", test_phase2_shader_splicing),
    ("5. Phase 3 Texel Density", test_phase3_texel_density),
    ("6. Phase 4 HUD & 3D Overlays", test_phase4_hud_and_3d_overlays),
    ("7. Phase 5 Fixers & Cleanup", test_phase5_fixers_and_cleanup),
    ("8. Bug Review Regressions (P0-P3)", test_bug_regressions),
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
