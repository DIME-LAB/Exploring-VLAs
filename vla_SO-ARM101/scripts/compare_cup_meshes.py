#!/usr/bin/env python3
"""Spawn cup collision objects with different mesh representations side by side.

Adds 7 cups in a row behind the origin so you can visually compare in RViz.
Each uses a different collision geometry approach. Positions start at the
actual drop pose Y and stack behind the robot (negative X).

Usage:
    python3 scripts/compare_cup_meshes.py          # spawn all variants
    python3 scripts/compare_cup_meshes.py --clear   # remove all variants
"""
import argparse
import sys
import trimesh
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose
from moveit_msgs.msg import CollisionObject, PlanningScene as PlanningSceneMsg
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive, Mesh as ShapeMesh, MeshTriangle

STL_PATH = '/home/aaugus11/Projects/Exploring-VLAs/vla_SO-ARM101/src/so_arm101_description/meshes/cup/cup.stl'
SCALE = 0.001   # mm -> m
PADDING = 1.1   # 10% collision padding

# Layout: cups spaced 0.12m apart behind the robot (negative X), at drop pose Y
DROP_Y = -0.28   # actual drop pose Y from /drop_poses
BASE_Z = 0.0
START_X = -0.10   # first cup near the robot
SPACING_X = -0.12  # each cup further behind

VARIANTS = [
    'raw_mesh',
    'double_sided',
    'convex_hull',
    'vhacd_4',
    'vhacd_8',
    'vhacd_16',
    'cylinder',
]

LABELS = {
    'raw_mesh':      '1: Raw STL (24K faces, striped)',
    'double_sided':  '2: Double-sided (both windings, exact shape)',
    'convex_hull':   '3: Convex hull (solid, fills opening)',
    'vhacd_4':       '4: VHACD 4 hulls',
    'vhacd_8':       '5: VHACD 8 hulls',
    'vhacd_16':      '6: VHACD 16 hulls',
    'cylinder':      '7: Cylinder (old fallback)',
}


def trimesh_to_shape_mesh(tm, scale):
    """Convert a trimesh object to shape_msgs/Mesh."""
    mesh = ShapeMesh()
    for v in tm.vertices:
        pt = Point()
        pt.x = float(v[0]) * scale
        pt.y = float(v[1]) * scale
        pt.z = float(v[2]) * scale
        mesh.vertices.append(pt)
    for f in tm.faces:
        tri = MeshTriangle()
        tri.vertex_indices = [int(f[0]), int(f[1]), int(f[2])]
        mesh.triangles.append(tri)
    return mesh


def make_variant(name, idx):
    """Create CollisionObject(s) for the given variant. Returns a list."""
    raw = trimesh.load(STL_PATH)
    s = SCALE * PADDING

    x = START_X + idx * SPACING_X
    objects = []

    if name == 'raw_mesh':
        raw.fix_normals()
        co = CollisionObject()
        co.header.frame_id = 'base'
        co.id = f'compare_{name}'
        co.operation = CollisionObject.ADD
        p = Pose()
        p.position.x = x
        p.position.y = DROP_Y
        p.position.z = BASE_Z
        p.orientation.w = 1.0
        co.meshes.append(trimesh_to_shape_mesh(raw, s))
        co.mesh_poses.append(p)
        objects.append(co)

    elif name == 'double_sided':
        # Duplicate every face with reversed winding — renders both sides in RViz
        raw.fix_normals()
        flipped = raw.copy()
        flipped.faces = raw.faces[:, ::-1]
        doubled = trimesh.util.concatenate([raw, flipped])
        co = CollisionObject()
        co.header.frame_id = 'base'
        co.id = f'compare_{name}'
        co.operation = CollisionObject.ADD
        p = Pose()
        p.position.x = x
        p.position.y = DROP_Y
        p.position.z = BASE_Z
        p.orientation.w = 1.0
        co.meshes.append(trimesh_to_shape_mesh(doubled, s))
        co.mesh_poses.append(p)
        objects.append(co)

    elif name == 'convex_hull':
        hull = raw.convex_hull
        co = CollisionObject()
        co.header.frame_id = 'base'
        co.id = f'compare_{name}'
        co.operation = CollisionObject.ADD
        p = Pose()
        p.position.x = x
        p.position.y = DROP_Y
        p.position.z = BASE_Z
        p.orientation.w = 1.0
        co.meshes.append(trimesh_to_shape_mesh(hull, s))
        co.mesh_poses.append(p)
        objects.append(co)

    elif name.startswith('vhacd_'):
        max_hulls = int(name.split('_')[1])
        pieces = trimesh.decomposition.convex_decomposition(raw, maxConvexHulls=max_hulls)
        # All pieces go into a single CollisionObject with multiple meshes
        co = CollisionObject()
        co.header.frame_id = 'base'
        co.id = f'compare_{name}'
        co.operation = CollisionObject.ADD
        for piece in pieces:
            tm = trimesh.Trimesh(vertices=piece['vertices'], faces=piece['faces'])
            mesh = trimesh_to_shape_mesh(tm, s)
            p = Pose()
            p.position.x = x
            p.position.y = DROP_Y
            p.position.z = BASE_Z
            p.orientation.w = 1.0
            co.meshes.append(mesh)
            co.mesh_poses.append(p)
        objects.append(co)

    elif name == 'cylinder':
        co = CollisionObject()
        co.header.frame_id = 'base'
        co.id = f'compare_{name}'
        co.operation = CollisionObject.ADD
        cyl = SolidPrimitive()
        cyl.type = SolidPrimitive.CYLINDER
        cup_h = 0.0965 * PADDING
        cup_r = 0.039 * PADDING
        cyl.dimensions = [cup_h, cup_r]
        p = Pose()
        p.position.x = x
        p.position.y = DROP_Y
        p.position.z = BASE_Z + cup_h / 2.0
        p.orientation.w = 1.0
        co.primitives.append(cyl)
        co.primitive_poses.append(p)
        objects.append(co)

    elif name == 'cone':
        co = CollisionObject()
        co.header.frame_id = 'base'
        co.id = f'compare_{name}'
        co.operation = CollisionObject.ADD
        cone = SolidPrimitive()
        cone.type = SolidPrimitive.CONE
        cup_h = 0.0965 * PADDING
        cup_r = 0.039 * PADDING
        cone.dimensions = [cup_h, cup_r]  # [height, radius] — tip up, base at bottom
        p = Pose()
        p.position.x = x
        p.position.y = DROP_Y
        p.position.z = BASE_Z + cup_h / 2.0
        p.orientation.w = 1.0
        co.primitives.append(cone)
        co.primitive_poses.append(p)
        objects.append(co)

    elif name == 'sphere':
        co = CollisionObject()
        co.header.frame_id = 'base'
        co.id = f'compare_{name}'
        co.operation = CollisionObject.ADD
        sph = SolidPrimitive()
        sph.type = SolidPrimitive.SPHERE
        cup_r = 0.039 * PADDING
        sph.dimensions = [cup_r]
        p = Pose()
        p.position.x = x
        p.position.y = DROP_Y
        p.position.z = BASE_Z + cup_r
        p.orientation.w = 1.0
        co.primitives.append(sph)
        co.primitive_poses.append(p)
        objects.append(co)

    elif name == 'box':
        co = CollisionObject()
        co.header.frame_id = 'base'
        co.id = f'compare_{name}'
        co.operation = CollisionObject.ADD
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        extents = raw.extents * s
        box.dimensions = [float(extents[0]), float(extents[1]), float(extents[2])]
        p = Pose()
        p.position.x = x
        p.position.y = DROP_Y
        p.position.z = BASE_Z + float(extents[2]) / 2.0
        p.orientation.w = 1.0
        co.primitives.append(box)
        co.primitive_poses.append(p)
        objects.append(co)

    return objects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clear', action='store_true', help='Remove all comparison cups')
    args = parser.parse_args()

    rclpy.init()
    node = Node('compare_cup_meshes')
    cli = node.create_client(ApplyPlanningScene, '/apply_planning_scene')
    if not cli.wait_for_service(timeout_sec=5.0):
        print('ApplyPlanningScene service not available')
        rclpy.shutdown()
        sys.exit(1)

    scene = PlanningSceneMsg()
    scene.is_diff = True

    if args.clear:
        for name in VARIANTS:
            co = CollisionObject()
            co.header.frame_id = 'base'
            co.id = f'compare_{name}'
            co.operation = CollisionObject.REMOVE
            scene.world.collision_objects.append(co)
        print('Removing all comparison cups...')
    else:
        print('Spawning 7 cup variants in a row behind the robot:\n')
        for i, name in enumerate(VARIANTS):
            cos = make_variant(name, i)
            for co in cos:
                scene.world.collision_objects.append(co)
            x = START_X + i * SPACING_X
            detail = ''
            if name.startswith('vhacd_'):
                n_meshes = len(cos[0].meshes)
                detail = f' ({n_meshes} pieces)'
            print(f'  {LABELS[name]}{detail}  →  X={x:.2f}')
        print()

    req = ApplyPlanningScene.Request()
    req.scene = scene
    future = cli.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=30.0)

    if future.result() and future.result().success:
        if args.clear:
            print('All comparison cups removed.')
        else:
            print('All variants spawned. Check RViz to compare.')
            print('Run with --clear to remove them.')
    else:
        print('Failed to apply planning scene.')

    rclpy.shutdown()


if __name__ == '__main__':
    main()
