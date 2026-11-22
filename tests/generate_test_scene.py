"""Headless test-scene generator for FlightCheck verification suites.

Builds deterministic Blender scenes containing calibrated PBR, colorspace,
texel density, scale, and dimension test cases without fragile bpy.ops calls.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import bmesh
import bpy


def _ensure_material_nodes(mat: bpy.types.Material):
    """Ensure material node tree exists across Blender 4.2 LTS and Blender 5.0+."""
    if mat.node_tree is None and bpy.app.version < (5, 0, 0):
        mat.use_nodes = True
    return mat.node_tree


def clear_scene(scene: Optional[bpy.types.Scene] = None) -> None:
    """Remove all objects, meshes, materials, images, and custom node groups."""
    if scene is None:
        scene = bpy.context.scene

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    for mesh in list(bpy.data.meshes):
        mesh.use_fake_user = False
        bpy.data.meshes.remove(mesh, do_unlink=True)

    for mat in list(bpy.data.materials):
        mat.use_fake_user = False
        bpy.data.materials.remove(mat, do_unlink=True)

    for img in list(bpy.data.images):
        if getattr(img, "type", "") in {"RENDER_RESULT", "COMPOSITING"}:
            continue
        img.use_fake_user = False
        try:
            bpy.data.images.remove(img, do_unlink=True)
        except RuntimeError:
            pass

    for ng in list(bpy.data.node_groups):
        ng.use_fake_user = False
        bpy.data.node_groups.remove(ng, do_unlink=True)

    for cam in list(bpy.data.cameras):
        cam.use_fake_user = False
        bpy.data.cameras.remove(cam, do_unlink=True)

    for light in list(bpy.data.lights):
        light.use_fake_user = False
        bpy.data.lights.remove(light, do_unlink=True)

    if scene is not None and scene.world is not None and scene.world.users == 0:
        bpy.data.worlds.remove(scene.world, do_unlink=True)


def _create_filled_image(
    name: str,
    width: int,
    height: int,
    rgba: Tuple[float, float, float, float] = (0.45, 0.45, 0.45, 1.0),
    colorspace: str = "sRGB",
) -> bpy.types.Image:
    """Create an in-memory image filled with a uniform RGBA value and pack it."""
    img = bpy.data.images.new(name=name, width=width, height=height, alpha=True)
    img.colorspace_settings.name = colorspace
    pixel_count = width * height
    img.pixels.foreach_set(list(rgba) * pixel_count)
    img.update()
    try:
        img.pack()
    except RuntimeError:
        pass
    return img


def _create_cube_mesh(
    name: str,
    size: float = 1.0,
    uv_scale: float = 1.0,
    create_uv: bool = True,
) -> bpy.types.Mesh:
    """Create a cube mesh with explicit per-face [0, uv_scale] UVs via bmesh."""
    mesh = bpy.data.meshes.new(name=name)
    bm = bmesh.new()
    try:
        bmesh.ops.create_cube(bm, size=size)
        if create_uv:
            uv_layer = bm.loops.layers.uv.verify()
            uv_corners = (
                (0.0, 0.0),
                (uv_scale, 0.0),
                (uv_scale, uv_scale),
                (0.0, uv_scale),
            )
            for face in bm.faces:
                for idx, loop in enumerate(face.loops):
                    u, v = uv_corners[idx % 4]
                    loop[uv_layer].uv = (u, v)
        bm.normal_update()
        bm.to_mesh(mesh)
    finally:
        bm.free()
    mesh.update()
    return mesh


def _link_cube_object(
    scene: bpy.types.Scene,
    name: str,
    mesh: bpy.types.Mesh,
    material: Optional[bpy.types.Material] = None,
    location: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> bpy.types.Object:
    """Instantiate a mesh object into the scene collection with optional material."""
    obj = bpy.data.objects.new(name=name, object_data=mesh)
    obj.location = location
    obj.scale = scale
    scene.collection.objects.link(obj)
    if material is not None:
        if len(obj.data.materials) == 0:
            obj.data.materials.append(material)
        else:
            obj.data.materials[0] = material
    return obj


def _set_socket_value(node: bpy.types.Node, socket_name: str, value) -> None:
    """Set a named node input socket default_value if present."""
    sock = node.inputs.get(socket_name)
    if sock is not None:
        sock.default_value = value


def _create_principled_material(
    name: str,
    base_color: Tuple[float, float, float, float] = (0.35, 0.35, 0.35, 1.0),
    metallic: float = 0.0,
    roughness: float = 0.5,
    specular_ior_level: float = 0.5,
    ior: float = 1.45,
) -> Tuple[bpy.types.Material, bpy.types.ShaderNodeTree, bpy.types.Node, bpy.types.Node]:
    """Create a material with healthy Principled BSDF defaults linked to Output."""
    mat = bpy.data.materials.new(name=name)
    nt = _ensure_material_nodes(mat)

    output = None
    bsdf = None
    for node in nt.nodes:
        if node.type == "OUTPUT_MATERIAL" and output is None:
            output = node
        elif node.type == "BSDF_PRINCIPLED" and bsdf is None:
            bsdf = node

    if output is None:
        output = nt.nodes.new("ShaderNodeOutputMaterial")
        output.location = (300.0, 0.0)
    if bsdf is None:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (0.0, 0.0)

    output.is_active_output = True
    surf_in = output.inputs.get("Surface")
    bsdf_out = bsdf.outputs.get("BSDF")
    if surf_in is not None and bsdf_out is not None and not surf_in.is_linked:
        nt.links.new(bsdf_out, surf_in)

    _set_socket_value(bsdf, "Base Color", base_color)
    _set_socket_value(bsdf, "Metallic", metallic)
    _set_socket_value(bsdf, "Roughness", roughness)
    _set_socket_value(bsdf, "Specular IOR Level", specular_ior_level)
    _set_socket_value(bsdf, "IOR", ior)

    return mat, nt, bsdf, output


def _attach_image_node(
    nt: bpy.types.ShaderNodeTree,
    bsdf: bpy.types.Node,
    image: bpy.types.Image,
    target_sockets: Tuple[str, ...],
    location: Tuple[float, float] = (-350.0, 0.0),
) -> bpy.types.Node:
    """Create a ShaderNodeTexImage and connect its Color output to target sockets."""
    tex_node = nt.nodes.new("ShaderNodeTexImage")
    tex_node.image = image
    tex_node.location = location
    color_out = tex_node.outputs.get("Color")
    for sock_name in target_sockets:
        target_in = bsdf.inputs.get(sock_name)
        if color_out is not None and target_in is not None:
            nt.links.new(color_out, target_in)
    return tex_node


def _grid_loc(index: int, spacing: float = 3.5) -> Tuple[float, float, float]:
    col = index % 5
    row = index // 5
    return (col * spacing, row * spacing, 0.0)


def _build_diagnostic_objects(
    scene: bpy.types.Scene,
    include_unfixable: bool = True,
) -> Dict[str, bpy.types.Object]:
    """Build diagnostic test objects; optionally excludes unfixable edge cases."""
    created: Dict[str, bpy.types.Object] = {}
    idx = 0

    mat_healthy, _, _, _ = _create_principled_material(
        "Mat_Healthy",
        base_color=(0.35, 0.35, 0.35, 1.0),
        metallic=0.0,
        roughness=0.5,
        specular_ior_level=0.5,
        ior=1.45,
    )

    pbr_cases = [
        ("Obj_CrushedBlack", "Mat_CrushedBlack", (0.005, 0.005, 0.005, 1.0), 0.0, 0.5, 0.5, 1.45),
        ("Obj_BlownAlbedoWarn", "Mat_BlownAlbedoWarn", (0.75, 0.75, 0.75, 1.0), 0.0, 0.5, 0.5, 1.45),
        ("Obj_BlownAlbedoCrit", "Mat_BlownAlbedoCrit", (0.95, 0.95, 0.95, 1.0), 0.0, 0.5, 0.5, 1.45),
        ("Obj_DarkMetal", "Mat_DarkMetal", (0.20, 0.20, 0.20, 1.0), 1.0, 0.3, 0.5, 1.45),
        ("Obj_SemiMetal", "Mat_SemiMetal", (0.55, 0.55, 0.55, 1.0), 0.45, 0.4, 0.5, 1.45),
        ("Obj_MirrorRough", "Mat_MirrorRough", (0.4, 0.4, 0.4, 1.0), 0.0, 0.005, 0.5, 1.45),
        ("Obj_DeadRough", "Mat_DeadRough", (0.4, 0.4, 0.4, 1.0), 0.0, 0.995, 0.5, 1.45),
        ("Obj_SpecIOROutlier", "Mat_SpecIOROutlier", (0.4, 0.4, 0.4, 1.0), 0.0, 0.5, 0.90, 2.10),
    ]

    for obj_name, mat_name, col, met, rgh, spec, ior in pbr_cases:
        mat, _, _, _ = _create_principled_material(
            mat_name,
            base_color=col,
            metallic=met,
            roughness=rgh,
            specular_ior_level=spec,
            ior=ior,
        )
        mesh = _create_cube_mesh(f"Mesh_{obj_name}", size=1.0, uv_scale=1.0)
        created[obj_name] = _link_cube_object(
            scene, obj_name, mesh, material=mat, location=_grid_loc(idx)
        )
        idx += 1

    # 9. Obj_BadColorspaceDataOnly
    img_rough = _create_filled_image(
        "Img_Rough_sRGB", 64, 64, rgba=(0.45, 0.45, 0.45, 1.0), colorspace="sRGB"
    )
    mat_data_only, nt_data, bsdf_data, _ = _create_principled_material("Mat_BadColorspaceDataOnly")
    _attach_image_node(nt_data, bsdf_data, img_rough, ("Roughness",))
    created["Obj_BadColorspaceDataOnly"] = _link_cube_object(
        scene,
        "Obj_BadColorspaceDataOnly",
        _create_cube_mesh("Mesh_Obj_BadColorspaceDataOnly", size=1.0, uv_scale=1.0),
        material=mat_data_only,
        location=_grid_loc(idx),
    )
    idx += 1

    # 10. Obj_BadColorspaceShared (unfixable protected shared image)
    if include_unfixable:
        img_shared = _create_filled_image(
            "Img_Shared_sRGB", 64, 64, rgba=(0.5, 0.5, 0.5, 1.0), colorspace="sRGB"
        )
        mat_shared, nt_shared, bsdf_shared, _ = _create_principled_material("Mat_BadColorspaceShared")
        _attach_image_node(nt_shared, bsdf_shared, img_shared, ("Base Color", "Metallic"))
        created["Obj_BadColorspaceShared"] = _link_cube_object(
            scene,
            "Obj_BadColorspaceShared",
            _create_cube_mesh("Mesh_Obj_BadColorspaceShared", size=1.0, uv_scale=1.0),
            material=mat_shared,
            location=_grid_loc(idx),
        )
        idx += 1

    # 11. Obj_NonUniformScale
    created["Obj_NonUniformScale"] = _link_cube_object(
        scene,
        "Obj_NonUniformScale",
        _create_cube_mesh("Mesh_Obj_NonUniformScale", size=1.0, uv_scale=1.0),
        material=mat_healthy,
        location=_grid_loc(idx),
        scale=(1.0, 2.5, 0.5),
    )
    idx += 1

    # 12. Obj_InvertedScale
    created["Obj_InvertedScale"] = _link_cube_object(
        scene,
        "Obj_InvertedScale",
        _create_cube_mesh("Mesh_Obj_InvertedScale", size=1.0, uv_scale=1.0),
        material=mat_healthy,
        location=_grid_loc(idx),
        scale=(-1.0, 1.0, 1.0),
    )
    idx += 1

    # 13. Obj_MultiUserA & Obj_MultiUserB sharing one mesh datablock
    shared_mesh = _create_cube_mesh("Mesh_MultiUserShared", size=1.0, uv_scale=1.0)
    shared_mesh.materials.append(mat_healthy)
    created["Obj_MultiUserA"] = _link_cube_object(
        scene,
        "Obj_MultiUserA",
        shared_mesh,
        material=None,
        location=_grid_loc(idx),
        scale=(1.5, 1.5, 1.5),
    )
    idx += 1
    created["Obj_MultiUserB"] = _link_cube_object(
        scene,
        "Obj_MultiUserB",
        shared_mesh,
        material=None,
        location=_grid_loc(idx),
        scale=(0.5, 1.0, 2.0),
    )
    idx += 1

    # 14 & 15. Unfixable dimension outliers
    if include_unfixable:
        created["Obj_DimTiny"] = _link_cube_object(
            scene,
            "Obj_DimTiny",
            _create_cube_mesh("Mesh_Obj_DimTiny", size=0.0002, uv_scale=1.0),
            material=mat_healthy,
            location=_grid_loc(idx),
        )
        idx += 1
        created["Obj_DimHuge"] = _link_cube_object(
            scene,
            "Obj_DimHuge",
            _create_cube_mesh("Mesh_Obj_DimHuge", size=600.0, uv_scale=1.0),
            material=mat_healthy,
            location=(0.0, -700.0, 0.0),
        )

    # 16, 17, 18. Texel density test cubes with 1024x1024 mid-gray image on Base Color
    img_texel_1024 = _create_filled_image(
        "Img_TexelRef_1024", 1024, 1024, rgba=(0.45, 0.45, 0.45, 1.0), colorspace="sRGB"
    )
    if include_unfixable:
        mat_tex_ref, nt_tex_ref, bsdf_tex_ref, _ = _create_principled_material("Mat_TexelRef1m")
        _attach_image_node(nt_tex_ref, bsdf_tex_ref, img_texel_1024, ("Base Color",))
        created["Obj_TexelRef1m"] = _link_cube_object(
            scene,
            "Obj_TexelRef1m",
            _create_cube_mesh("Mesh_Obj_TexelRef1m", size=1.0, uv_scale=1.0),
            material=mat_tex_ref,
            location=_grid_loc(idx),
        )
        idx += 1

    mat_tex_low, nt_tex_low, bsdf_tex_low, _ = _create_principled_material("Mat_TexelLow")
    _attach_image_node(nt_tex_low, bsdf_tex_low, img_texel_1024, ("Base Color",))
    created["Obj_TexelLow"] = _link_cube_object(
        scene,
        "Obj_TexelLow",
        _create_cube_mesh("Mesh_Obj_TexelLow", size=1.0, uv_scale=0.25),
        material=mat_tex_low,
        location=_grid_loc(idx),
    )
    idx += 1

    mat_tex_high, nt_tex_high, bsdf_tex_high, _ = _create_principled_material("Mat_TexelHigh")
    _attach_image_node(nt_tex_high, bsdf_tex_high, img_texel_1024, ("Base Color",))
    created["Obj_TexelHigh"] = _link_cube_object(
        scene,
        "Obj_TexelHigh",
        _create_cube_mesh("Mesh_Obj_TexelHigh", size=1.0, uv_scale=3.0),
        material=mat_tex_high,
        location=_grid_loc(idx),
    )
    idx += 1

    # 19. Obj_Healthy (zero issues)
    created["Obj_Healthy"] = _link_cube_object(
        scene,
        "Obj_Healthy",
        _create_cube_mesh("Mesh_Obj_Healthy", size=1.0, uv_scale=1.0),
        material=mat_healthy,
        location=_grid_loc(idx),
        scale=(1.0, 1.0, 1.0),
    )

    return created


def build_full_violation_scene(scene: Optional[bpy.types.Scene] = None) -> Dict[str, bpy.types.Object]:
    """Populate scene with every diagnostic violation case plus Obj_Healthy."""
    if scene is None:
        scene = bpy.context.scene
    clear_scene(scene)
    return _build_diagnostic_objects(scene, include_unfixable=True)


def build_remediable_scene(scene: Optional[bpy.types.Scene] = None) -> Dict[str, bpy.types.Object]:
    """Populate scene with all auto-fixable violations (transitions NO-GO -> GO)."""
    if scene is None:
        scene = bpy.context.scene
    clear_scene(scene)
    return _build_diagnostic_objects(scene, include_unfixable=False)


def _parse_cli_output_path() -> str:
    argv = sys.argv
    if "--" in argv:
        idx = argv.index("--")
        extra = argv[idx + 1 :]
        if extra:
            return extra[0]
    return "./test_out.blend"


if __name__ == "__main__":
    out_path = os.path.abspath(_parse_cli_output_path())
    parent_dir = os.path.dirname(out_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    target_scene = bpy.context.scene
    objs = build_full_violation_scene(target_scene)
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print(f"[FlightCheck] Saved test scene ({len(objs)} objects) to {out_path}")
