"""Fit the legs on a retarget copy while retaining its original vertex weights."""
import json

import bpy
import numpy as np
from mathutils import Matrix, Vector

KEY = 'deadlock_retarget_leg_fit'


def fit_legs(rig):
    from . import object_context
    from .retarget import contact_offsets
    from .preview import RESTORE_KEY, toggle_idle
    if rig.get('deadlock_retarget_preserve_proportions'):
        raise ValueError('Create a fresh retarget preview to fit individual legs instead of preserving proportions')
    if rig.get(KEY):
        raise ValueError('These legs are already fitted; create a fresh retarget preview to fit again')
    reference = {n: Matrix(m) for n, m in json.loads(rig['deadlock_retarget_reference']).items()}
    mapping = json.loads(rig['deadlock_retarget_mapping'])
    reverse = {role: name for name, role in mapping.items()}
    roles = {f'{stem}_{side}' for side in ('L', 'R') for stem in ('leg_upper', 'leg_lower', 'ankle')}
    if roles - reverse.keys():
        raise ValueError('Fitting legs requires both thighs, knees and ankles')
    offsets = contact_offsets(rig, reference)
    old = {b.name: rig.matrix_world @ b.matrix_local for b in rig.data.bones}
    old_tails = {b.name: rig.matrix_world @ b.tail_local for b in rig.data.bones}
    lengths = {b.name: (old_tails[b.name] - old[b.name].translation).length for b in rig.data.bones}
    new, warps, new_lengths = {}, {}, {}
    foot_descendants = {}
    for side in ('L', 'R'):
        foot = rig.data.bones[reverse[f'ankle_{side}']]
        for bone in [foot] + list(foot.children_recursive):
            foot_descendants[bone.name] = side
    pending = list(rig.data.bones)
    while pending:
        for bone in pending[:]:
            if bone.parent and bone.parent.name not in new:
                continue
            name, role = bone.name, mapping.get(bone.name)
            if role and role.startswith(('leg_upper_', 'leg_lower_')):
                child = f'leg_lower_{role[-1]}' if role.startswith('leg_upper_') else f'ankle_{role[-1]}'
                start = reference[role].translation
                end = reference[child] @ offsets.get(child, Vector((0, 0, 0)))
                direction = end - start
                if min(direction.length, lengths[name]) < 1e-6:
                    raise ValueError(f'Cannot fit a zero-length leg segment: {name}')
                rotation = (old[name].to_3x3() @ Vector((0, 1, 0))).normalized().rotation_difference(direction.normalized())
                new[name] = Matrix.LocRotScale(start, rotation @ old[name].to_quaternion(), old[name].to_scale())
                ratio = direction.length / lengths[name]
                warps[name] = new[name] @ Matrix.Diagonal((1, ratio, 1, 1)) @ old[name].inverted()
                new_lengths[name] = direction.length
            elif name in foot_descendants:
                side = foot_descendants[name]
                foot = reverse[f'ankle_{side}']
                shift = reference[f'ankle_{side}'] @ offsets[f'ankle_{side}'] - old[foot].translation
                new[name] = old[name].copy()
                new[name].translation += shift
                warps[name] = Matrix.Translation(shift)
                new_lengths[name] = lengths[name]
            elif bone.parent and bone.parent.name in warps:
                parent = bone.parent.name
                rotation = new[parent].to_quaternion() @ old[parent].to_quaternion().inverted() @ old[name].to_quaternion()
                position = warps[parent] @ old[name].translation
                new[name] = Matrix.LocRotScale(position, rotation, old[name].to_scale())
                end = warps[parent] @ old_tails[name]
                new_lengths[name] = (end - position).length
                warps[name] = new[name] @ old[name].inverted()
            else:
                new[name] = old[name].copy()
            pending.remove(bone)

    meshes = {o for c in rig.users_collection for o in c.objects if o.type == 'MESH' and o.find_armature() == rig}
    prepared = []
    changed_counts = {}
    for obj in meshes:
        mesh = obj.data
        count = len(mesh.vertices)
        groups = {g.index: g.name for g in obj.vertex_groups if g.name in old}
        totals = np.zeros(count)
        affected = [(v.index, groups[w.group], w.weight) for v in mesh.vertices for w in v.groups
                    if w.group in groups and w.weight > 0 and groups[w.group] in warps]
        for vertex in mesh.vertices:
            totals[vertex.index] = sum(w.weight for w in vertex.groups if w.group in groups and w.weight > 0)
        affine = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
        inverse = obj.matrix_world.inverted()
        for name, warp in warps.items():
            influences = [(i, weight) for i, group, weight in affected if group == name]
            if not influences:
                continue
            indices = np.asarray([i for i, _ in influences])
            weights = np.asarray([weight for _, weight in influences]) / totals[indices]
            delta = np.asarray(inverse @ warp @ obj.matrix_world) - np.eye(4)
            affine[indices] += weights[:, None, None] * delta
        changed = np.max(np.abs(affine - np.eye(4)), axis=(1, 2)) > 1e-8
        if not changed.any():
            continue
        changed_counts[obj.name] = int(changed.sum())
        # Stage coordinates first so malformed input does not partially fit a copy.
        channels = [('mesh', mesh.vertices)]
        if mesh.shape_keys:
            channels += [(key.name, key.data) for key in mesh.shape_keys.key_blocks]
        coordinates = []
        for label, data in channels:
            values = np.empty(count * 3, dtype=np.float32)
            data.foreach_get('co', values)
            values = values.reshape(-1, 3)
            output = values.copy()
            output[changed] = np.einsum('nij,nj->ni', affine[changed, :3, :3], values[changed]) + affine[changed, :3, 3]
            if not np.isfinite(output).all():
                raise ValueError(f'Invalid fitted mesh coordinates: {obj.name}, {label}')
            coordinates.append((data, output))
        normals = None
        if mesh.has_custom_normals:
            normals = np.asarray([normal.vector[:] for normal in mesh.corner_normals])
            vertices = np.asarray([loop.vertex_index for loop in mesh.loops])
            transforms = np.linalg.inv(affine[:, :3, :3]).transpose(0, 2, 1)
            normals = np.einsum('nij,nj->ni', transforms[vertices], normals)
            sizes = np.linalg.norm(normals, axis=1)
            if np.any(sizes < 1e-8) or not np.isfinite(normals).all():
                raise ValueError(f'Invalid fitted mesh normals: {obj.name}')
            normals /= sizes[:, None]
        prepared.append((obj, coordinates, normals))

    active = RESTORE_KEY in rig
    if active:
        toggle_idle(bpy.context, bpy.context.scene.deadlock_export)
    with object_context():
        rig.hide_set(False)
        for obj in list(bpy.context.selected_objects):
            obj.select_set(False)
        rig.select_set(True)
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode='EDIT')
        inverse = rig.matrix_world.inverted()
        for name in warps:
            bone = rig.data.edit_bones[name]
            bone.matrix = inverse @ new[name]
            bone.length = new_lengths[name] / rig.matrix_world.to_scale().y
        bpy.ops.object.mode_set(mode='OBJECT')
        for obj, coordinates, normals in prepared:
            for data, values in coordinates:
                data.foreach_set('co', values.ravel())
            obj.data.update()
            if normals is not None:
                obj.data.normals_split_custom_set(normals.tolist())
        report = {'affected_bones': sorted(warps), 'mesh_modified_vertices': changed_counts,
                  'bone_warps': {n: [list(row) for row in matrix] for n, matrix in warps.items()},
                  'old_lengths_meters': {n: lengths[n] for n in new_lengths},
                  'fitted_lengths_meters': new_lengths, 'weights_modified': False}
        rig[KEY] = json.dumps(report)
        bpy.context.view_layer.update()
    if active:
        toggle_idle(bpy.context, bpy.context.scene.deadlock_export)
    return report
