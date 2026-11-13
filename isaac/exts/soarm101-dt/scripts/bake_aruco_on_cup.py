# Reference: Blender headless script to project ArUco marker onto cup mesh
"""
Blender headless script: creates a UV-mapped plane, shrinkwraps it onto
the cup STL surface, and exports as a standalone USD mesh.

Usage:
    blender --background --python bake_aruco_on_cup.py -- \
        --cup-stl /path/to/cup.stl \
        --marker-png /path/to/aruco_4x4_000.png \
        --output /path/to/aruco_marker_000.usda \
        --marker-size 0.025 \
        --height-fraction 0.45 \
        --subdivisions 16

The output USD contains a single curved mesh with UV-mapped ArUco texture,
sized and positioned to sit on the cup surface at the specified height.
Import it as a child of the cup prim in Isaac Sim.
"""
import bpy
import bmesh
import sys
import os
import math
import argparse


def parse_args():
    """Parse arguments after the -- separator."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Bake ArUco marker onto cup mesh")
    parser.add_argument("--cup-stl", required=True, help="Path to cup STL file")
    parser.add_argument("--marker-png", required=True, help="Path to ArUco marker PNG")
    parser.add_argument("--output", required=True, help="Output USD/USDA path")
    parser.add_argument("--marker-size", type=float, default=0.025, help="Marker size in meters")
    parser.add_argument("--height-fraction", type=float, default=0.45, help="Height fraction on cup (0=bottom, 1=top)")
    parser.add_argument("--subdivisions", type=int, default=16, help="Grid subdivisions for the marker plane")
    parser.add_argument("--offset", type=float, default=0.0003, help="Normal offset from surface in meters")
    return parser.parse_args(argv)


def clear_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    for mat in bpy.data.materials:
        bpy.data.materials.remove(mat)
    for img in bpy.data.images:
        bpy.data.images.remove(img)


def import_cup(stl_path):
    """Import cup STL and return the object. Auto-detects mm vs m scale."""
    bpy.ops.wm.stl_import(filepath=stl_path)
    cup = bpy.context.selected_objects[0]
    cup.name = "cup_target"

    # Auto-detect scale: if bounding box height > 1m, assume mm and scale to m
    dims = cup.dimensions
    if max(dims) > 1.0:
        scale = 0.001
        cup.scale = (scale, scale, scale)
        bpy.context.view_layer.update()
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        print(f"  STL was in mm — scaled to meters (factor={scale})")

    return cup


def measure_cup(cup):
    """Measure cup dimensions from its bounding box."""
    import mathutils
    bbox = [cup.matrix_world @ mathutils.Vector(corner) for corner in cup.bound_box]
    xs = [v.x for v in bbox]
    ys = [v.y for v in bbox]
    zs = [v.z for v in bbox]
    return {
        "min": (min(xs), min(ys), min(zs)),
        "max": (max(xs), max(ys), max(zs)),
        "center_x": (min(xs) + max(xs)) / 2,
        "center_y": (min(ys) + max(ys)) / 2,
        "height": max(zs) - min(zs),
        "z_min": min(zs),
        "z_max": max(zs),
    }


def get_radius_at_height(cup, z, num_samples=72):
    """Raycast inward from a circle to find the cup surface radius at height z."""
    import mathutils
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cup_eval = cup.evaluated_get(depsgraph)

    max_radius = 0.06  # start rays from outside the cup
    radii = []
    for i in range(num_samples):
        angle = 2 * math.pi * i / num_samples
        origin = mathutils.Vector((max_radius * math.cos(angle),
                                   max_radius * math.sin(angle), z))
        direction = mathutils.Vector((-math.cos(angle), -math.sin(angle), 0))
        hit, loc, normal, face_idx = cup_eval.ray_cast(origin, direction)
        if hit:
            r = math.sqrt(loc.x**2 + loc.y**2)
            radii.append(r)

    return sum(radii) / len(radii) if radii else 0.035


def create_marker_plane(marker_size, subdivisions):
    """Create a subdivided plane with pre-set UVs mapping [0,1] x [0,1]."""
    bm = bmesh.new()
    half = marker_size / 2.0
    bmesh.ops.create_grid(bm, x_segments=subdivisions, y_segments=subdivisions, size=half)

    # Create UV layer and set UVs on the flat plane BEFORE any deformation.
    # Grid verts go from -half to +half in X and Y.
    # After rotation (0, 90°, 0): grid X → world -Z (height), grid Y → world Y (lateral).
    # For correct marker orientation: U = lateral (grid Y), V = height (grid -X flipped).
    uv_layer = bm.loops.layers.uv.new("UVMap")
    for face in bm.faces:
        for loop in face.loops:
            co = loop.vert.co
            # Current result is 180° rotated → flip both U and V.
            # face_origin 180° Z rotation mirrors Y in world space,
            # plus image V-flip (top-left origin) compounds to 180° rotation.
            u = (co.y + half) / (2.0 * half)
            v = (-co.x + half) / (2.0 * half)
            loop[uv_layer].uv = (u, v)

    mesh = bpy.data.meshes.new("aruco_marker_mesh")
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new("aruco_marker", mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def position_and_wrap_marker(marker_obj, cup_obj, cup_info, marker_size,
                             height_fraction, offset):
    """Position the marker plane on the cup surface and shrinkwrap it."""
    import mathutils

    # Target Z height for marker center
    target_z = cup_info["z_min"] + cup_info["height"] * height_fraction
    radius = get_radius_at_height(cup_obj, target_z)

    # Position the plane vertically at the cup surface along +X.
    # The flat grid is in XY. Rotate 90° around Y so:
    #   grid X → world -Z (but we fix with 90° around new X to get Z up)
    # Actually: we want the plane in the YZ plane with normal along +X.
    # Rotation: 90° around Y maps (X→-Z, Y→Y, Z→X).
    # So grid X→-Z, grid Y→Y. The plane is now in YZ with normal along X. But
    # X was the grid's depth, now Z. We need grid X→Z, grid Y→Y.
    # Rotation: -90° around Y maps (X→Z, Y→Y, Z→-X). Normal = -X (inward).
    # Use +90° around Y: X→-Z, Y→Y, Z→X. Then the plane is at Z<0 half.
    # Simplest: rotate 90° around Y, then flip. Or just rotate (0, 90°, 0):
    marker_obj.rotation_euler = (0, math.radians(90), 0)
    marker_obj.location = (radius + offset, 0, target_z)
    bpy.context.view_layer.update()

    bpy.context.view_layer.objects.active = marker_obj
    marker_obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Shrinkwrap: PROJECT along -X (inward toward cup center).
    # After transform_apply, the plane is vertical in world space at (radius, 0, z).
    # Project each vertex inward along -X to land on the cup surface.
    shrinkwrap = marker_obj.modifiers.new(name="Shrinkwrap", type='SHRINKWRAP')
    shrinkwrap.target = cup_obj
    shrinkwrap.wrap_method = 'PROJECT'
    shrinkwrap.use_project_x = True
    shrinkwrap.use_project_y = False
    shrinkwrap.use_project_z = False
    shrinkwrap.use_negative_direction = True
    shrinkwrap.use_positive_direction = False
    shrinkwrap.offset = offset

    bpy.ops.object.modifier_apply(modifier="Shrinkwrap")

    print(f"  Marker positioned at z={target_z:.4f}, radius={radius:.4f}")


def setup_uvs_and_material(marker_obj, marker_png_path):
    """Assign ArUco texture material. UVs are already set from create_marker_plane."""
    bpy.context.view_layer.objects.active = marker_obj
    marker_obj.select_set(True)

    # UVs were set on the flat plane before shrinkwrap — they survive
    # deformation because UVs are per-loop, not recomputed on vertex moves.

    # Create material with ArUco texture
    mat = bpy.data.materials.new(name="ArUcoMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Create nodes: Image Texture -> Principled BSDF -> Material Output
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    output_node.location = (300, 0)

    bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf_node.location = (0, 0)
    bsdf_node.inputs['Roughness'].default_value = 0.9
    bsdf_node.inputs['Metallic'].default_value = 0.0

    tex_node = nodes.new(type='ShaderNodeTexImage')
    tex_node.location = (-300, 0)
    tex_node.image = bpy.data.images.load(marker_png_path)
    tex_node.image.colorspace_settings.name = 'sRGB'

    # Link nodes
    links.new(tex_node.outputs['Color'], bsdf_node.inputs['Base Color'])
    links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])

    # Assign material to object
    marker_obj.data.materials.append(mat)

    print(f"  Material created with texture: {marker_png_path}")


def export_marker(marker_obj, output_path):
    """Export only the marker mesh as USD."""
    # Deselect all, select only marker
    bpy.ops.object.select_all(action='DESELECT')
    marker_obj.select_set(True)
    bpy.context.view_layer.objects.active = marker_obj

    # Move marker to origin for clean export (it will be re-positioned by the extension)
    # Actually NO — keep it at its position relative to cup origin so the extension
    # just parents it to the cup prim and it's in the right place.

    ext = os.path.splitext(output_path)[1].lower()
    if ext in ('.usd', '.usda', '.usdc'):
        bpy.ops.wm.usd_export(
            filepath=output_path,
            selected_objects_only=True,
            export_materials=True,
            export_textures=True,
            overwrite_textures=True,
            relative_paths=True,
        )
    elif ext in ('.obj',):
        bpy.ops.wm.obj_export(
            filepath=output_path,
            export_selected_objects=True,
            export_materials=True,
        )
    else:
        # Fallback to glTF
        bpy.ops.export_scene.gltf(
            filepath=output_path,
            use_selection=True,
            export_materials='EXPORT',
        )

    print(f"  Exported: {output_path}")


def main():
    args = parse_args()

    print(f"=== Baking ArUco marker onto cup ===")
    print(f"  Cup STL: {args.cup_stl}")
    print(f"  Marker PNG: {args.marker_png}")
    print(f"  Output: {args.output}")
    print(f"  Size: {args.marker_size}m, Height: {args.height_fraction:.0%}")

    clear_scene()

    # Import cup
    cup = import_cup(args.cup_stl)
    cup_info = measure_cup(cup)
    print(f"  Cup: height={cup_info['height']:.4f}m, "
          f"z=[{cup_info['z_min']:.4f}, {cup_info['z_max']:.4f}]")

    # Create marker plane
    marker = create_marker_plane(args.marker_size, args.subdivisions)

    # Position on cup surface and shrinkwrap
    position_and_wrap_marker(marker, cup, cup_info, args.marker_size,
                             args.height_fraction, args.offset)

    # UV map and texture
    setup_uvs_and_material(marker, args.marker_png)

    # Export marker only (not the cup)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    export_marker(marker, args.output)

    print(f"=== Done ===")


if __name__ == "__main__":
    main()
