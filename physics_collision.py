"""Fit bone-attached body capsules and share their geometry with preview/export."""
import json

import numpy as np

VERTEX_MAP = 'deadlock_auto_body_collision'
BODY_ROLES = frozenset(('pelvis', 'spine_0', 'spine_1', 'spine_2', 'spine_3', 'neck_0', 'head',
                        'arm_upper_L', 'arm_lower_L', 'arm_upper_R', 'arm_lower_R',
                        'leg_upper_L', 'leg_lower_L', 'leg_upper_R', 'leg_lower_R'))


def fit_capsule(points, origin, direction):
    """A conservative capsule inside the main weighted surface, ignoring outliers."""
    points = np.asarray(points, dtype=float)
    if len(points) < 12 or not np.isfinite(points).all():
        return None
    axis = np.asarray(direction, dtype=float)
    length = np.linalg.norm(axis)
    if length < 1e-8:
        return None
    axis /= length
    offset = points - origin
    along = offset @ axis
    radial = offset - along[:, None] * axis
    center = np.median(radial, axis=0)
    center -= np.dot(center, axis) * axis
    distances = np.linalg.norm(radial - center, axis=1)
    lo, hi = np.percentile(along, (5, 95))
    # CSDK collapses zero-length capsules into spheres and loses their vertex
    # map. Keep a short axis so opt-in filtering survives native compilation.
    radius = min(float(np.percentile(distances, 65)) * 0.9, float(hi - lo) * 0.45)
    if radius < 1e-5:
        return None
    first, last = lo + radius, hi - radius
    base = np.asarray(origin) + center
    return base + first * axis, base + last * axis, radius


def fit_body_colliders(rig, names=None):
    """Fit once in rest pose; preserve every pose channel and the source geometry."""
    from .physics import chains, chain_bones
    entries = chains(rig)
    if not any(entry.get('auto_collision') for entry in entries):
        return []
    import bpy
    from mathutils import Vector
    from . import UNIT
    from .retarget_export import unit_matrix
    names = names or {bone.name: bone.name for bone in rig.data.bones}
    mapping = json.loads(rig.get('deadlock_retarget_mapping', '{}'))
    body = {name for name, role in mapping.items() if role in BODY_ROLES and name in rig.data.bones}
    excluded = {bone.name for entry in entries for bone in chain_bones(rig, entry['root'])}
    owners = {}
    for bone in rig.data.bones:
        current = bone
        while current and current.name not in body and current.name not in excluded:
            current = current.parent
        if current and current.name in body:
            owners[bone.name] = current.name
    objects = {obj for collection in rig.users_collection for obj in collection.all_objects
               if obj.type == 'MESH' and not obj.hide_render
               and any(mod.type == 'ARMATURE' and mod.object == rig for mod in obj.modifiers)}
    clouds = {name: [] for name in body}
    previous = rig.data.pose_position
    try:
        rig.data.pose_position = 'REST'
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        for obj in sorted(objects, key=lambda obj: obj.name):
            groups = {group.index: owners.get(group.name) for group in obj.vertex_groups}
            evaluated = obj.evaluated_get(depsgraph)
            mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
            try:
                for vertex in mesh.vertices:
                    influence = max(vertex.groups, key=lambda group: group.weight, default=None)
                    owner = groups.get(influence.group) if influence and influence.weight > 0 else None
                    if owner:
                        clouds[owner].append(tuple(obj.matrix_world @ vertex.co))
            finally:
                evaluated.to_mesh_clear()
    finally:
        rig.data.pose_position = previous
        bpy.context.view_layer.update()
    colliders = []
    for name in sorted(body):
        bone = rig.data.bones[name]
        frame = unit_matrix(rig.matrix_world @ bone.matrix_local)
        fitted = fit_capsule(clouds[name], np.asarray(frame.translation), np.asarray(frame.col[1][:3]))
        if fitted is None:
            continue
        point0, point1, radius = fitted
        inverse = frame.inverted()
        colliders.append({'_class': 'ClothShapeCapsule', 'name': 'auto_body_' + names[name],
                          'parent_bone': names[name], 'point0': list((inverse @ Vector(point0)) / UNIT),
                          'point1': list((inverse @ Vector(point1)) / UNIT),
                          'radius0': radius / UNIT, 'radius1': radius / UNIT,
                          'vertex_map': VERTEX_MAP, 'bounciness': 0.0})
    if not colliders:
        raise ValueError('Automatic collision fitting needs a weighted body mesh and mapped body bones')
    return colliders


def world_capsules(colliders, matrices, unit):
    """Transform fitted bone-local geometry once per preview tick."""
    result = []
    for collider in colliders:
        frame = np.array(matrices[collider['parent_bone']], dtype=float)
        rotation = frame[:3, :3] / np.linalg.norm(frame[:3, :3], axis=0)
        if np.linalg.det(rotation) < 0:
            rotation = -rotation
        points = [frame[:3, 3] + rotation @ (np.array(collider[key]) * unit) for key in ('point0', 'point1')]
        result.append([*points[0], *points[1], collider['radius0'] * unit])
    return np.asarray(result, dtype=float).reshape(-1, 7)


def project_capsules(points, indices, thickness, capsules):
    """Project movable points out of the deepest overlapping capsule in one batch."""
    if not len(capsules) or not len(indices):
        return
    start, axis = capsules[:, :3], capsules[:, 3:6] - capsules[:, :3]
    delta = points[indices][None, :, :] - start[:, None, :]
    along = np.clip(np.einsum('cni,ci->cn', delta, axis) /
                    np.maximum(np.einsum('ci,ci->c', axis, axis)[:, None], 1e-16), 0, 1)
    delta -= along[:, :, None] * axis[:, None, :]
    distance = np.linalg.norm(delta, axis=2)
    depth = capsules[:, 6:7] + thickness[None, :] - distance
    deepest = np.argmax(depth, axis=0)
    columns = np.arange(len(indices))
    penetration = depth[deepest, columns]
    normals = delta[deepest, columns] / np.maximum(distance[deepest, columns, None], 1e-12)
    # Points on the centerline still need a finite, perpendicular escape route.
    centered = distance[deepest, columns] < 1e-12
    if centered.any():
        directions = axis[deepest[centered]]
        references = np.eye(3)[np.argmin(np.abs(directions), axis=1)]
        perpendicular = np.cross(directions, references)
        norms = np.linalg.norm(perpendicular, axis=1, keepdims=True)
        perpendicular = np.where(norms > 1e-12, perpendicular / np.maximum(norms, 1e-12), np.array((1, 0, 0)))
        normals[centered] = perpendicular
    points[indices] += normals * np.maximum(penetration, 0)[:, None]


def validate_colliders(fe, colliders, rows, unit):
    """Require fitted geometry and a vertex map that excludes all other chains."""
    from mathutils import Vector
    names = [str(name) for name in fe['m_CtrlName']]
    maps = fe.get('m_VertexMaps', [])
    index = next((i for i, row in enumerate(maps) if str(row['sName']) == VERTEX_MAP), None)
    if index is None:
        raise ValueError('Compiler did not create the automatic body collision selection')
    selection = maps[index]
    weights = fe['m_VertexMapValues']
    def selected(node):
        offset = node - int(selection['nVertexBase'])
        return (0 <= offset < int(selection['nVertexCount'])
                and weights[int(selection['nMapOffset']) + offset] > 0)
    for row in rows:
        if row['simulate']:
            node = names.index(row['joint_name'])
            if selected(node) != (row.get('vertex_map') == VERTEX_MAP):
                raise ValueError('Automatic body collision selection changed: ' + row['joint_name'])
            if row.get('vertex_map') == VERTEX_MAP:
                radius = fe['m_NodeCollisionRadii'][node - int(fe['m_nStaticNodes'])]
                if abs(float(radius) - row['collision_radius']) * unit > 0.00002:
                    raise ValueError('Automatic collision thickness did not compile: ' + row['joint_name'])
    capsules = fe.get('m_TaperedCapsuleRigids', [])
    checked = []
    for collider in colliders:
        matches = [capsule for capsule in capsules if names[int(capsule['nNode'])] == collider['parent_bone']
                   and int(capsule['nVertexMapIndex']) == index]
        if not any(all((Vector(sphere[:3]) - Vector(collider[key])).length * unit < 0.0002
                       and abs(float(sphere[3]) - collider[radius]) * unit < 0.0002
                       for sphere, key, radius in zip(capsule['vSphere'], ('point0', 'point1'), ('radius0', 'radius1')))
                   for capsule in matches):
            raise ValueError('Automatic body collider did not compile: ' + collider['parent_bone'])
        checked.append(collider['parent_bone'])
    return {'fitted_capsules': len(checked), 'bones': checked, 'selection_verified': True}
