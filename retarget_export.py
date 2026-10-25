"""Preserve the avatar hierarchy and drive it with authored Source 2 constraints."""
import json
import math
import re

import bpy
from mathutils import Matrix, Quaternion, Vector

from .core import class_spans
from .retarget import retarget_bind_rotations, contact_offsets, position_roles, forearm_uses_aim


def unit_matrix(matrix):
    return Matrix.LocRotScale(matrix.translation, matrix.to_quaternion(), Vector((1, 1, 1)))


def runtime_root_reference(settings, reference, root):
    """Read the graph's root frame; decompiled NM animation uses an axis fixup."""
    from . import archive, read_resource, selected_hero, UNIT
    retail=archive(settings)
    path=selected_hero(settings)[2]
    model=read_resource(retail.read(path),path,model=True)
    data=model.get_block(None,block_name='DATA')
    bindings=data.get('m_vecNmSkeletonRefs',[])
    if not bindings:
        return reference[root].copy()
    for binding in bindings:
        skeleton_path=str(binding)
        if not skeleton_path.endswith('_c'): skeleton_path+='_c'
        skeleton=read_resource(retail.read(skeleton_path),skeleton_path).get_block(None,block_name='DATA')
        ids=[str(name) for name in skeleton['m_boneIDs']]
        if root in ids:
            row=skeleton['m_modelSpaceReferencePose'][ids.index(root)]
            rotation=Quaternion((float(row[7]),float(row[4]),float(row[5]),float(row[6])))
            return Matrix.LocRotScale(Vector(row[:3])*UNIT,rotation,Vector((float(row[3]),)*3))
    raise ValueError(f'Native animation skeleton does not contain the retarget root {root}')


def prepare_retarget_export(settings):
    from .physics_preview import stop
    stop()
    from . import source_objects, reference_bones, bone_matrices, object_context, selected_hero, UNIT
    source, meshes = source_objects(settings)
    if not source.get('deadlock_retarget_mapping'):
        raise ValueError('Create and select Retarget Original Rig Preview before exporting it')
    if settings.collection.get('deadlock_hero') != settings.hero:
        raise ValueError('The retarget preview belongs to a different hero; create a preview for the selected hero')
    native_bones = reference_bones(settings)
    reference = bone_matrices(native_bones)
    mapping = json.loads(source['deadlock_retarget_mapping'])
    # Upgrade existing previews created before native eye aliases were supported.
    from .conversion import semantic_mapping
    for name, role in semantic_mapping(source.data.bones.keys(), reference.keys()).items():
        if role.startswith('eye_'):
            mapping[name] = role
    stored = {n: Matrix(m) for n, m in json.loads(source['deadlock_retarget_reference']).items()}
    if set(stored) != set(reference) or any(
            max(abs(stored[n][i][j]-reference[n][i][j]) for i in range(4) for j in range(4)) > 1e-5
            for n in reference):
        raise ValueError('Hero skeleton changed since this preview was made; recreate the retarget preview')
    rest = {b.name: unit_matrix(source.matrix_world @ b.matrix_local) for b in source.data.bones}
    bind = retarget_bind_rotations(source, reference, mapping)
    position_offsets=contact_offsets(source,reference)
    positions=position_roles(source)
    # Namespaces also prevent case-insensitive collisions such as Head / head.
    names = {b.name: f'vrx_{i:03d}_' + re.sub(r'[^A-Za-z0-9_]', '_', b.name)[:40]
             for i, b in enumerate(source.data.bones)}
    from .physics import export_chains, KEY
    physics_nodes = export_chains(source, names)
    from .physics_collision import fit_body_colliders
    physics_colliders = fit_body_colliders(source, names)
    parents = {b.name: b.parent.name if b.parent else None for b in source.data.bones}
    root = 'root_motion' if 'root_motion' in reference else next(b.name for b in native_bones if not b.parent)
    runtime_root=runtime_root_reference(settings,reference,root)
    from SourceIO.library.source2.utils.murmurhash2 import murmur_hash2
    def identity(name):
        return {'m_nBoneHash': murmur_hash2(name.lower(), 0x31415926), 'm_sName': name}
    def quaternion(q):
        return [q.x, q.y, q.z, q.w]
    def rule(kind, name, target, position, rotation, weight=1.0):
        return {'_class': kind, 'm_name': f'deadlock_toolkit_retarget_{name}', 'm_vUpVector': [0.0]*3,
                'm_slaves': [{**identity(names[name]), 'm_flWeight': 1.0,
                              'm_qBaseOrientation': quaternion(rest[name].to_quaternion()),
                              'm_vBasePosition': list(rest[name].translation / UNIT)}],
                'm_targets': [{**identity(target), 'm_flWeight': weight, 'm_bIsAttachment': False,
                               'm_vOffset': list(position / UNIT), 'm_qOffset': quaternion(rotation)}]}
    constraints = []
    for bone in source.data.bones:
        name, parent = bone.name, parents[bone.name]
        # Only the driving frame changes. The authored bind hierarchy stays
        # intact so the imported mesh and its inverse bind poses are preserved.
        parent_rest = rest[parent] if parent else runtime_root
        local = parent_rest.inverted() @ rest[name]
        parent_name = names[parent] if parent else root
        role = mapping.get(name)
        position_target=parent_name
        position_offset=local.translation
        if role=='pelvis':
            position_target=role
            position_offset=reference[role].inverted()@rest[name].translation
        elif role in positions:
            position_target=role
            position_offset=position_offsets.get(role,Vector((0,0,0)))
        constraints.append(rule('CPointConstraint', name, position_target, position_offset,
                                Matrix.Identity(4).to_quaternion()))
        if role=='pelvis' and source.get('deadlock_retarget_preserve_proportions'):
            shift=runtime_root.to_quaternion().inverted()@position_offsets['pelvis_root_shift']
            # Affine weights sum to one: native pelvis + root-space lift.
            # Keeping lift on the actor root avoids rotating it with the hips.
            lift=rule('CPointConstraint',name,root,shift,Matrix.Identity(4).to_quaternion())['m_targets'][0]
            cancel=rule('CPointConstraint',name,root,Vector((0,0,0)),Matrix.Identity(4).to_quaternion(),-1.0)['m_targets'][0]
            constraints[-1]['m_targets'].extend([lift,cancel])
        if role:
            offset = reference[role].to_quaternion().inverted() @ bind[name]
            constraints.append(rule('COrientConstraint', name, role, Vector((0, 0, 0)), offset))
            if role == 'neck_0' and 'head' in reference:
                offset = reference['head'].to_quaternion().inverted() @ bind[name]
                constraints.append(rule('COrientConstraint', name, 'head', Vector((0, 0, 0)), offset, 0.5))
            if forearm_uses_aim(source,name,mapping):
                hand=f'hand_{role[-1]}'
                offset=reference[hand].to_quaternion().inverted()@bind[name]
                constraints.append(rule('COrientConstraint',name,hand,Vector((0,0,0)),offset,0.5))
                # Aim at self + native wrist - native elbow. This keeps the
                # native forearm direction while distributing the wrist twist.
                aim=rule('CAimConstraint',name,hand,Vector((0,0,0)),Matrix.Identity(4).to_quaternion())
                self_target={**identity(names[name]),'m_flWeight':1.0,'m_bIsAttachment':False,
                             'm_vOffset':[0.0]*3,'m_qOffset':[0.0,0.0,0.0,1.0]}
                negative={**self_target,**identity(role),'m_flWeight':-1.0}
                aim.update({'m_name':aim['m_name']+'_aim','m_vUpVector':[0.0,0.0,1.0],
                            'm_qAimOffset':[0.0,0.0,0.0,1.0],'m_nUpType':0,
                            'm_targets':[self_target.copy(),aim['m_targets'][0],negative,self_target.copy()]})
                constraints.append(aim)
            elif source.get('deadlock_retarget_leg_fit') and role.startswith(('leg_upper_','leg_lower_')):
                child=f'leg_lower_{role[-1]}' if role.startswith('leg_upper_') else f'ankle_{role[-1]}'
                aim=rule('CAimConstraint',name,child,position_offsets.get(child,Vector((0,0,0))),Matrix.Identity(4).to_quaternion())
                up=reference[role].to_quaternion().inverted()@(bind[name]@Vector((0,0,1)))
                up_target={**identity(role),'m_flWeight':1.0,'m_bIsAttachment':False,
                           'm_vOffset':[0.0]*3,'m_qOffset':[0.0,0.0,0.0,1.0]}
                aim.update({'m_name':aim['m_name']+'_aim','m_vUpVector':list(up),
                            'm_qAimOffset':[0.0,0.0,0.0,1.0],'m_nUpType':0})
                aim['m_targets'].append(up_target)
                constraints.append(aim)
        else:
            constraints.append(rule('COrientConstraint', name, parent_name, Vector((0, 0, 0)), local.to_quaternion()))
    from .weapon import weapon_binding
    weapon=weapon_binding(native_bones,mapping,names)
    if weapon:
        weapon_root=weapon['root']
        def weapon_target(name,weight):
            return {**identity(name),'m_flWeight':weight,'m_bIsAttachment':False,
                    'm_vOffset':[0.0]*3,'m_qOffset':[0.0,0.0,0.0,1.0]}
        # Native weapon + avatar wrist - native wrist. Native rotations and
        # reload animation survive; Source 2 carries the weapon descendants.
        weapon_rule={'_class':'CPointConstraint','m_name':'deadlock_toolkit_weapon_'+weapon_root,
                            'm_vUpVector':[0.0]*3,
                            'm_slaves':[{**identity(weapon_root),'m_flWeight':1.0,
                                         'm_qBaseOrientation':quaternion(reference[weapon_root].to_quaternion()),
                                         'm_vBasePosition':list(reference[weapon_root].translation/UNIT)}],
                            'm_targets':[weapon_target(weapon_root,1.0),weapon_target(weapon['avatar_hand'],1.0),
                                         weapon_target(weapon['native_hand'],-1.0)]}
        if not weapon['root_follows_native_hand']:constraints.append(weapon_rule)
        # Third-person items also use native hand attachments (weapon_attach
        # on Silver). Move that hand LAST, after its original animation drove
        # the avatar and the independent weapon root received the displacement.
        hand=weapon['native_hand']
        constraints.append({'_class':'CPointConstraint','m_name':'deadlock_toolkit_weapon_attachment_'+hand,
                            'm_vUpVector':[0.0]*3,
                            'm_slaves':[{**identity(hand),'m_flWeight':1.0,
                                         'm_qBaseOrientation':quaternion(reference[hand].to_quaternion()),
                                         'm_vBasePosition':list(reference[hand].translation/UNIT)}],
                            'm_targets':[weapon_target(weapon['avatar_hand'],1.0)]})
    collection = bpy.data.collections.new(f'Deadlock Retarget Export - {selected_hero(settings)[1]}')
    collection['deadlock_converted'] = True
    collection['deadlock_hero'] = settings.hero
    collection['deadlock_retarget_export'] = True
    bpy.context.scene.collection.children.link(collection)
    with object_context():
        rig = source.copy(); rig.data = source.data.copy(); rig.parent = None
        rig.name = f'Retarget Export Rig - {selected_hero(settings)[1]}'
        rig.animation_data_clear(); rig.hide_viewport = False
        collection.objects.link(rig); rig.hide_set(False)
        for constraint in list(rig.constraints): rig.constraints.remove(constraint)
        for bone in rig.pose.bones:
            for constraint in list(bone.constraints): bone.constraints.remove(constraint)
            bone.matrix_basis = Matrix.Identity(4)
        for obj in list(bpy.context.selected_objects): obj.select_set(False)
        rig.select_set(True); bpy.context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode='EDIT')
        for name, replacement in names.items(): rig.data.edit_bones[name].name = replacement
        inverse = rig.matrix_world.inverted()
        for bone in native_bones:
            edit = rig.data.edit_bones.new(bone.name)
            edit.head = inverse @ reference[bone.name].translation
            edit.tail = edit.head + Vector((0, 0.02, 0))
            edit.matrix = inverse @ reference[bone.name]
        for bone in native_bones:
            if bone.parent: rig.data.edit_bones[bone.name].parent = rig.data.edit_bones[bone.parent]
        for name, parent in parents.items():
            if parent is None: rig.data.edit_bones[names[name]].parent = rig.data.edit_bones[root]
        from .physics_swing import add_swing_helpers
        swing_helpers = add_swing_helpers(rig.data.edit_bones, physics_nodes, root)
        bpy.ops.object.mode_set(mode='OBJECT')
        for original in meshes:
            obj = original.copy(); obj.data = original.data.copy(); obj.animation_data_clear()
            if obj.data.shape_keys: obj.data.shape_keys.animation_data_clear()
            for constraint in list(obj.constraints): obj.constraints.remove(constraint)
            for modifier in obj.modifiers:
                if modifier.type == 'ARMATURE' and modifier.object == source: modifier.object = rig
            for group in obj.vertex_groups:
                if group.name in names: group.name = names[group.name]
            obj.parent = rig; collection.objects.link(obj)
            obj.matrix_world = original.matrix_world.copy()
            obj.hide_viewport = False; obj.hide_set(False); obj.hide_select = False
        rig['deadlock_retarget_mapping'] = json.dumps({names[n]: role for n, role in mapping.items()})
        rig['deadlock_retarget_original_names'] = json.dumps(names)
        rig['deadlock_retarget_constraints'] = json.dumps(constraints)
        rig['deadlock_retarget_weapon_binding'] = json.dumps(weapon)
        rig['deadlock_physics_nodes'] = json.dumps(physics_nodes)
        rig['deadlock_physics_colliders'] = json.dumps(physics_colliders)
        if KEY in rig:
            del rig[KEY]
        rig['deadlock_retarget_runtime_roots'] = json.dumps({root:[list(row) for row in runtime_root]})
        rig['deadlock_retarget_pelvis_motion'] = True
        if 'deadlock_idle_restore' in rig: del rig['deadlock_idle_restore']
        rig.data.pose_position = 'REST'
        bpy.context.view_layer.update()
    skeleton = {b.name: {'parent': b.parent, 'matrix': unit_matrix(reference[b.name])} for b in native_bones}
    skeleton.update({names[n]: {'parent': names[p] if p else root, 'matrix': matrix} for n, matrix in rest.items()
                     for p in [parents[n]]})
    skeleton.update({name: {'parent': rig.data.bones[name].parent.name,
                            'matrix': unit_matrix(rig.matrix_world @ rig.data.bones[name].matrix_local)}
                     for name in swing_helpers})
    report = {'hero': selected_hero(settings)[1], 'method': 'Original rig runtime retargeting',
              'source_bones': len(source.data.bones), 'native_bones': len(native_bones),
              'export_bones': len(skeleton), 'runtime_constraints': len(constraints),
              'bone_mapping': mapping, 'export_bone_names': names, 'weights_modified': False,
              'physics_chains': json.loads(source.get(KEY, '[]')),
              'weapon_binding':weapon,
              'pelvis_position_follows_native': True,
              'leg_joint_positions_follow_native': not bool(source.get('deadlock_retarget_preserve_proportions')),
              'preserve_avatar_proportions':bool(source.get('deadlock_retarget_preserve_proportions')),
              'uniform_avatar_scale':source.matrix_world.to_scale().x,
              'whole_pose_offset_meters':list(position_offsets.get('pelvis_root_shift',Vector((0,0,0)))),
              'foot_contact_offsets_meters':{name:list(offset) for name,offset in position_offsets.items()},
              'root_motion_binding': {'driver':root,'model_bind_transform':[list(row) for row in reference[root]],
                                      'runtime_reference_transform':[list(row) for row in runtime_root]},
              'rest_mesh_modified': bool(source.get('deadlock_retarget_leg_fit')), 'original_bone_lengths_preserved': False,
              'original_rest_bone_lengths_preserved': not bool(source.get('deadlock_retarget_leg_fit')),
              'leg_proportions_fit':json.loads(source.get('deadlock_retarget_leg_fit','null')),
              'note': 'Original vertex weights are retained. Optional leg fitting changes rest leg proportions and leg-weighted geometry on the copy. Hero motion drives the avatar; fingers retain continuous roll, and forearms share wrist twist. Legs follow native contacts with calibrated sole height. The compiler may reduce skinning influences; consult compiled pose validation.'}
    if source.get('deadlock_retarget_preserve_proportions'):
        report['note']='Authored mesh, rest bone proportions and source weights are preserved. One uniform scale sizes the whole avatar. Only the pelvis receives position animation and a root-space ground offset; knees and ankles retain authored parent spacing. Finger roll and wrist twist corrections are included. Unequal avatar/hero segment ratios can still produce foot drift; inspect in game.'
    return collection, report, skeleton, constraints


def retarget_modeldoc(template, skeleton):
    from . import UNIT
    children = {}
    for name, bone in skeleton.items(): children.setdefault(bone['parent'], []).append(name)
    def node(name):
        bone = skeleton[name]
        parent = bone['parent']
        local = skeleton[parent]['matrix'].inverted() @ bone['matrix'] if parent else bone['matrix']
        euler = local.to_quaternion().to_euler('XYZ')
        angles = [math.degrees(euler.y), math.degrees(euler.z), math.degrees(euler.x)]
        return ('{ _class = "Bone" name = ' + json.dumps(name) +
                ' origin = ' + json.dumps(list(local.translation / UNIT)) +
                ' angles = ' + json.dumps(angles) + ' do_not_discard = true children = [ ' +
                ', '.join(node(child) for child in children.get(name, [])) + ' ] }')
    replacement = '{ _class = "Skeleton" children = [ ' + ', '.join(node(n) for n in children[None]) + ' ] }'
    spans = [(start, end) for start, end, cls in class_spans(template) if cls == 'Skeleton']
    if len(spans) != 1: raise ValueError('Expected exactly one authored hero Skeleton')
    start, end = spans[0]
    return template[:start] + replacement + template[end:]
