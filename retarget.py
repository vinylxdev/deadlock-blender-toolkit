"""Retarget hero motion onto the original avatar hierarchy without editing weights."""

import json

import bpy
from mathutils import Matrix, Vector

from .conversion import semantic_mapping, model_frame, joint_frame

POSITION_ROLES = {'pelvis','leg_lower_L','leg_lower_R','ankle_L','ankle_R'}
CONTACT_VERSION = 2  # Recalibrate previews grounded with the old branched-toe axes.


def position_roles(rig):
    if rig.get('deadlock_retarget_preserve_proportions'):
        return {'pelvis'}
    return POSITION_ROLES | ({'leg_upper_L','leg_upper_R'} if rig.get('deadlock_retarget_leg_fit') else set())


def aim_rotation(direction, up):
    """Source 2 aim convention: bone +Y forward, +Z towards projected up."""
    direction=direction.normalized()
    up=(up-direction*up.dot(direction)).normalized()
    if direction.length<0.5 or up.length<0.5:
        raise ValueError('Degenerate limb aim frame')
    return Matrix((direction.cross(up),direction,up)).transposed().to_quaternion()


def target_joint_axis(role, reference):
    """Use anatomical joint direction; mirrored native bones can point along -X."""
    next_joint = {'clavicle': 'arm_upper', 'arm_upper': 'arm_lower',
                  'arm_lower': 'hand', 'leg_upper': 'leg_lower',
                  'leg_lower': 'ankle', 'ankle': 'ball', 'ball': 'ball_end'}
    stem, _, side = role.rpartition('_')
    child = None
    if side in ('L', 'R'):
        if stem in next_joint:
            child = f'{next_joint[stem]}_{side}'
        elif stem.startswith('finger_'):
            finger, _, segment = stem.rpartition('_')
            if segment in ('0', '1', '2'):
                child = f'{finger}_{int(segment)+1 if segment != "2" else "end"}_{side}'
    if child:
        if child not in reference:
            if stem.startswith('finger_') and segment == '2':
                previous = f'{finger}_1_{side}'
                if previous in reference:
                    incoming = reference[role].translation-reference[previous].translation
                    direction = reference[role].to_3x3()@Vector((1,0,0))
                    alignment = direction.dot(incoming)
                    if abs(alignment) > 1e-8:
                        # Terminal joints need no end helper. Native +X can
                        # face inward on mirrored hands; retain the tip's bend.
                        return (direction if alignment > 0 else -direction).normalized()
            raise ValueError(f'Missing target joint {child} for retargeting {role}')
        direction = reference[child].translation-reference[role].translation
        if direction.length_squared < 1e-12:
            raise ValueError(f'Target joints {role} and {child} overlap')
        return direction.normalized()
    return reference[role].to_3x3()@Vector((1,0,0))


def source_joint_axis(name, role, rig, rest, reverse):
    """Read anatomical directions even when imported bone tails point sideways."""
    stem, _, side = role.rpartition('_')
    next_stem = {'clavicle': 'arm_upper', 'arm_upper': 'arm_lower',
                 'arm_lower': 'hand', 'hand': 'finger_middle_0',
                 'leg_upper': 'leg_lower', 'leg_lower': 'ankle', 'ankle': 'ball',
                 'ball': 'ball_end'}.get(stem)
    previous = None
    if stem.startswith('finger_'):
        finger, _, segment = stem.rpartition('_')
        if segment in ('0', '1'):
            next_stem = f'{finger}_{int(segment)+1}'
        elif segment == '2':
            previous = reverse.get(f'{finger}_1_{side}')
    child = reverse.get(f'{next_stem}_{side}') if next_stem else None
    children = [child] if child else [b.name for b in rig.data.bones[name].children]
    if stem == 'ball' and len(children) > 1:
        # A shared toe hub can fan out sideways into individual toe roots.
        # Their centroid is not the foot's forward direction; retain the
        # authored toe axis unless a mapped end joint gives a continuation.
        return (rest[name].to_3x3() @ Vector((0, 1, 0))).normalized()
    if children:
        direction = sum((rest[n].translation-rest[name].translation for n in children), Vector()) / len(children)
        if direction.length_squared > 1e-12:
            return direction.normalized()
    axis = (rest[name].to_3x3()@Vector((0,1,0))).normalized()
    if previous:
        incoming = rest[name].translation-rest[previous].translation
        previous_axis = (rest[previous].to_3x3()@Vector((0,1,0))).normalized()
        if incoming.length_squared > 1e-12 and previous_axis.dot(incoming.normalized()) < 0.9999:
            return incoming.normalized()
    return axis


def forearm_uses_aim(rig, name, mapping):
    """The native +Y aim convention requires an anatomically aligned source axis."""
    role = mapping.get(name, '')
    if not role.startswith('arm_lower_'):
        return False
    hand = next(n for n, r in mapping.items() if r == f'hand_{role[-1]}')
    bone = rig.data.bones[name]
    direction = rig.data.bones[hand].head_local-bone.head_local
    axis = bone.matrix_local.to_3x3()@Vector((0,1,0))
    return direction.length_squared > 1e-12 and axis.normalized().dot(direction.normalized()) > 0.9999


def create_retarget_preview(settings):
    from . import source_objects, reference_bones, bone_matrices, selected_hero, object_context
    source, meshes = source_objects(settings)
    target = bone_matrices(reference_bones(settings))
    mapping = semantic_mapping(source.data.bones.keys(), target.keys())
    # These are the main spine sections; spine_0 is a short pelvis helper on Lash.
    if all(name in target for name in ('spine_1', 'spine_2', 'spine_3')):
        for name, role in list(mapping.items()):
            if name not in target and role == 'spine_0': mapping[name] = 'spine_1'
            elif name not in target and role == 'spine_1': mapping[name] = 'spine_2'
    reverse = {role: name for name, role in mapping.items()}
    heads = {name: source.matrix_world @ bone.head_local for name, bone in source.data.bones.items()}
    landmarks = ('pelvis', 'head', 'arm_upper_L', 'arm_upper_R')
    source_frame = model_frame(*(heads[reverse[name]] for name in landmarks))
    target_frame = model_frame(*(target[name].translation for name in landmarks))
    src_feet = (heads[reverse['ankle_L']] + heads[reverse['ankle_R']]) * 0.5
    dst_feet = (target['ankle_L'].translation + target['ankle_R'].translation) * 0.5
    scale = (target['head'].translation-dst_feet).length / (heads[reverse['head']]-src_feet).length
    source_frame.translation=src_feet;target_frame.translation=dst_feet
    fit = target_frame @ Matrix.Scale(scale, 4) @ source_frame.inverted()
    _, hero_name, _ = selected_hero(settings)
    collection = bpy.data.collections.new(f'Deadlock Retarget Preview - {hero_name}')
    collection['deadlock_converted'] = True
    collection['deadlock_hero'] = settings.hero
    collection['deadlock_preview_only'] = True
    collection['deadlock_source_blend'] = bpy.data.filepath
    bpy.context.scene.collection.children.link(collection)
    created=[]
    try:
        with object_context():
            rig = source.copy();rig.data=source.data.copy()
            rig.name=f'Original Rig Retarget - {hero_name}'
            rig.animation_data_clear();rig.parent=None
            collection.objects.link(rig);created.append(rig)
            rig.matrix_world=fit@source.matrix_world
            rig.show_in_front=False;rig.hide_viewport=False;rig.hide_set(False)
            rig.data.pose_position='POSE'
            for constraint in list(rig.constraints):rig.constraints.remove(constraint)
            for bone in rig.pose.bones:
                for constraint in list(bone.constraints):bone.constraints.remove(constraint)
                bone.matrix_basis=Matrix.Identity(4)
            rig['deadlock_retarget_mapping']=json.dumps(mapping)
            rig['deadlock_retarget_reference']=json.dumps({name:[list(row) for row in matrix] for name,matrix in target.items()})
            rig['deadlock_retarget_across']=list(target_frame.col[0].to_3d())
            rig['deadlock_retarget_up']=list(target_frame.col[2].to_3d())
            for original in meshes:
                obj=original.copy();obj.data=original.data.copy()
                obj.animation_data_clear()
                if obj.data.shape_keys and obj.data.shape_keys != original.data.shape_keys:
                    obj.data.shape_keys.animation_data_clear()
                for modifier in obj.modifiers:
                    if modifier.type=='ARMATURE' and modifier.object==source:modifier.object=rig
                for constraint in list(obj.constraints):obj.constraints.remove(constraint)
                obj.parent=rig
                collection.objects.link(obj);created.append(obj)
                obj.matrix_world=fit@original.matrix_world
                obj.hide_viewport=False;obj.hide_set(False);obj.hide_select=False
                obj['deadlock_original_mesh']=original.name
            bpy.context.view_layer.update()
        report={'hero':hero_name,'method':'Original rig rotation retargeting',
                'source_bones':len(source.data.bones),'preview_bones':len(rig.data.bones),
                'uniform_scale':scale,'bone_mapping':mapping,'weights_modified':False,
                'rest_mesh_modified':False,'original_bone_lengths_preserved':False,
                'original_rest_bone_lengths_preserved':True,
                'preview_only':True,
                'note':'Native hero rotations and pelvis movement drive the original hierarchy. Animated knees and ankles follow native contacts, with sole-height calibration for the avatar. Original rest data, weights and accessory bones are retained. Use Build Retargeted VPK to compile runtime bone constraints.'}
        if settings.proportions=='PRESERVE':
            from .retarget_placement import preserve_proportions
            report.update(preserve_proportions(rig,restore_original=False))
            report['preserve_avatar_proportions']=True
            report['uniform_scale']=scale*rig.matrix_world.to_scale().x/(fit@source.matrix_world).to_scale().x
            report['note']='The entire avatar is sized uniformly and grounded together. Only the pelvis receives animated position changes; knees and ankles keep authored joint spacing. Original rest data and weights are retained. Build Retargeted VPK uses the same placement.'
        return collection,report
    except Exception:
        for obj in reversed(created):bpy.data.objects.remove(obj,do_unlink=True)
        bpy.data.collections.remove(collection)
        raise


def retarget_bind_rotations(rig, reference, mapping):
    """World-space avatar joint orientations aligned with the native neutral pose."""
    reverse={role:name for name,role in mapping.items()}
    across=Vector(rig['deadlock_retarget_across']);up=Vector(rig['deadlock_retarget_up'])
    rest={bone.name:rig.matrix_world@bone.matrix_local for bone in rig.data.bones}
    palm_rotations={};palm_normals={};palm_frames={}
    for side in ('L','R'):
        roles=[f'hand_{side}',f'finger_middle_0_{side}',f'finger_index_0_{side}',f'finger_pinky_0_{side}']
        if all(role in reverse for role in roles):
            src=[rest[reverse[role]].translation for role in roles];dst=[reference[role].translation for role in roles]
            sf=joint_frame(src[1]-src[0],src[2]-src[3]);tf=joint_frame(dst[1]-dst[0],dst[2]-dst[3])
            palm_rotations[roles[0]]=tf@sf.inverted()
            palm_normals[side]=(sf.col[2].to_3d(),tf.col[2].to_3d())
            palm_frames[side]=(sf,tf)
    # Carry the transverse frame along each finger instead of repeatedly
    # projecting the palm normal onto curled segments. Near a right angle that
    # projection changes sign, adding an artificial half-turn at the tip.
    finger_rotations={}
    for side,(source_normal,target_normal) in palm_normals.items():
        for finger in ('thumb','index','middle','ring','pinky'):
            source_side=source_normal.copy();target_side=target_normal.copy()
            previous_source=previous_target=None
            for segment in range(3):
                role=f'finger_{finger}_{segment}_{side}'
                if role not in reverse:break
                name=reverse[role]
                source_axis=source_joint_axis(name,role,rig,rest,reverse)
                target_axis=target_joint_axis(role,reference)
                if previous_source is not None:
                    source_side=previous_source.rotation_difference(source_axis)@source_side
                    target_side=previous_target.rotation_difference(target_axis)@target_side
                source_frame=joint_frame(source_axis,source_side)
                target_frame=joint_frame(target_axis,target_side)
                alignment=target_frame@source_frame.inverted()
                finger_rotations[role]=alignment.to_quaternion()@rest[name].to_quaternion()
                source_side=source_frame.col[0].to_3d();target_side=target_frame.col[0].to_3d()
                previous_source=source_axis;previous_target=target_axis
    # Anchor the neutral arm roll to its palm, then carry that frame back
    # through the forearm and upper arm. Global body-up gives a different
    # frame from the hand and creates an artificial wrist twist.
    arm_rotations={}
    for side in ('L','R'):
        hand=f'hand_{side}'
        if hand not in palm_rotations:continue
        source_frame,target_frame=palm_frames[side]
        source_axis=source_frame.col[1].to_3d();source_side=source_frame.col[0].to_3d()
        target_axis=target_frame.col[1].to_3d();target_side=target_frame.col[0].to_3d()
        for stem in ('arm_lower','arm_upper','clavicle'):
            role=f'{stem}_{side}'
            if role not in reverse:break
            name=reverse[role]
            next_source=source_joint_axis(name,role,rig,rest,reverse)
            next_target=target_joint_axis(role,reference)
            source_side=source_axis.rotation_difference(next_source)@source_side
            target_side=target_axis.rotation_difference(next_target)@target_side
            sf=joint_frame(next_source,source_side);tf=joint_frame(next_target,target_side)
            arm_rotations[role]=(tf@sf.inverted()).to_quaternion()@rest[name].to_quaternion()
            source_axis=next_source;target_axis=next_target
            source_side=sf.col[0].to_3d();target_side=tf.col[0].to_3d()
    result={}
    for name,role in mapping.items():
        if role in {'pelvis','neck_0','head'} or role.startswith(('spine_', 'eye_')):
            result[name]=rest[name].to_quaternion()
            continue
        if role in finger_rotations:
            result[name]=finger_rotations[role]
            continue
        if role in arm_rotations:
            result[name]=arm_rotations[role]
            continue
        if role in palm_rotations:
            alignment=palm_rotations[role]
        else:
            source_axis=source_joint_axis(name,role,rig,rest,reverse)
            target_axis=target_joint_axis(role,reference)
            source_side=target_side=up if role.startswith(('arm_','clavicle_')) else across
            if role.startswith('finger_') and role[-1] in palm_normals:
                source_side,target_side=palm_normals[role[-1]]
            alignment=joint_frame(target_axis,target_side)@joint_frame(source_axis,source_side).inverted()
        result[name]=alignment.to_quaternion()@rest[name].to_quaternion()
    return result


def retarget_idle_matrices(rig, path, time_seconds=0.0):
    from .preview import idle_matrices
    reference={name:Matrix(rows) for name,rows in json.loads(rig['deadlock_retarget_reference']).items()}
    mapping=json.loads(rig['deadlock_retarget_mapping'])
    for name,role in semantic_mapping(rig.data.bones.keys(),reference.keys()).items():
        if role.startswith('eye_'):mapping[name]=role
    rig['deadlock_retarget_mapping']=json.dumps(mapping)
    # Decode the native pose with a minimal virtual armature interface.
    class Bone:
        def __init__(self,name,matrix,parent=None):self.name=name;self.matrix_local=matrix;self.parent=parent
    from . import reference_bones
    settings=bpy.context.scene.deadlock_export
    bones={bone.name:Bone(bone.name,reference[bone.name]) for bone in reference_bones(settings)}
    for bone in reference_bones(settings):
        bones[bone.name].parent=bones.get(bone.parent)
    class Data:pass
    class Rig:pass
    native=Rig();native.data=Data();native.data.bones=list(bones.values())
    native_basis=idle_matrices(native,path,time_seconds)
    animated={};pending=list(bones.values())
    while pending:
        for bone in pending[:]:
            if bone.parent and bone.parent.name not in animated:continue
            local=bone.parent.matrix_local.inverted()@bone.matrix_local if bone.parent else bone.matrix_local
            animated[bone.name]=(animated[bone.parent.name] if bone.parent else Matrix.Identity(4))@local@native_basis[bone.name]
            pending.remove(bone)
    return retarget_pose_matrices(rig,animated,reference,contact_offsets(rig,reference))


def retarget_pose_matrices(rig, animated, reference, position_offsets=None):
    mapping=json.loads(rig['deadlock_retarget_mapping'])
    rest={bone.name:rig.matrix_world@bone.matrix_local for bone in rig.data.bones}
    bind_rotations=retarget_bind_rotations(rig,reference,mapping)
    positions=position_roles(rig)
    desired={};basis={}
    pending=list(rig.data.bones)
    while pending:
        for bone in pending[:]:
            if bone.parent and bone.parent.name not in desired:continue
            parent_world=desired[bone.parent.name] if bone.parent else rig.matrix_world
            local=bone.parent.matrix_local.inverted()@bone.matrix_local if bone.parent else bone.matrix_local
            base=parent_world@local
            role=mapping.get(bone.name)
            if role:
                delta=animated[role].to_quaternion()@reference[role].to_quaternion().inverted()
                if role=='neck_0' and 'head' in animated:
                    # The avatar has a much shorter neck than the hero. Share
                    # the head turn between its original neck/head joints rather
                    # than putting the whole twist into the final joint.
                    head_delta=animated['head'].to_quaternion()@reference['head'].to_quaternion().inverted()
                    delta=delta.slerp(head_delta,0.5)
                if forearm_uses_aim(rig,bone.name,mapping) and f'hand_{role[-1]}' in animated:
                    hand=f'hand_{role[-1]}'
                    hand_delta=animated[hand].to_quaternion()@reference[hand].to_quaternion().inverted()
                    delta=delta.slerp(hand_delta,0.5)
                rotation=delta@bind_rotations[bone.name]
                position=base.translation
                if role=='pelvis':
                    # Body bob, crouch and slide lowering live on the pelvis,
                    # not the actor's movement root. Keep the avatar bind offset.
                    offset=reference[role].inverted()@rest[bone.name].translation
                    position=animated[role]@offset
                    if rig.get('deadlock_retarget_preserve_proportions'):
                        root='root_motion' if 'root_motion' in reference else next(iter(reference))
                        root_delta=animated[root].to_quaternion()@reference[root].to_quaternion().inverted()
                        position+=root_delta@Vector((position_offsets or {}).get('pelvis_root_shift',(0,0,0)))
                elif role in positions:
                    position=animated[role]@Vector((position_offsets or {}).get(role,(0,0,0)))
                if forearm_uses_aim(rig,bone.name,mapping):
                    hand=f'hand_{role[-1]}'
                    direction=animated[hand].translation-animated[role].translation
                    rotation=aim_rotation(direction,rotation@Vector((0,0,1)))
                elif rig.get('deadlock_retarget_leg_fit') and role.startswith(('leg_upper_','leg_lower_')):
                    child=f'leg_lower_{role[-1]}' if role.startswith('leg_upper_') else f'ankle_{role[-1]}'
                    target=animated[child]@Vector((position_offsets or {}).get(child,(0,0,0)))
                    rotation=aim_rotation(target-position,rotation@Vector((0,0,1)))
                desired[bone.name]=Matrix.LocRotScale(position,rotation,rest[bone.name].to_scale())
                # Preserve local offsets except for the animated pelvis movement.
                relative=base.inverted()@desired[bone.name]
                basis[bone.name]=relative.to_quaternion().to_matrix().to_4x4()
                if role in positions:basis[bone.name].translation=relative.translation
                desired[bone.name]=base@basis[bone.name]
            else:
                basis[bone.name]=Matrix.Identity(4);desired[bone.name]=base
            pending.remove(bone)
    return basis


def contact_offsets(rig, reference):
    """Calibrate the avatar's soles against the standing idle ground plane."""
    if (rig.get('deadlock_retarget_contact_offsets') and
            rig.get('deadlock_retarget_contact_version') == CONTACT_VERSION):
        return {name:Vector(value) for name,value in json.loads(rig['deadlock_retarget_contact_offsets']).items()}
    from .preview import idle_source
    from .retarget_validation import native_idle
    from . import reference_bones
    settings=bpy.context.scene.deadlock_export
    _,path=idle_source(settings)
    animated=native_idle(reference,reference_bones(settings),path)
    basis=retarget_pose_matrices(rig,animated,reference,{})
    rest={bone.name:rig.matrix_world@bone.matrix_local for bone in rig.data.bones}
    posed={};skin={}
    for bone in rig.data.bones:
        local=bone.parent.matrix_local.inverted()@bone.matrix_local if bone.parent else bone.matrix_local
        posed[bone.name]=(posed[bone.parent.name] if bone.parent else rig.matrix_world)@local@basis[bone.name]
        skin[bone.name]=posed[bone.name]@rest[bone.name].inverted()
    mapping=json.loads(rig['deadlock_retarget_mapping']);reverse={role:name for name,role in mapping.items()}
    meshes={obj for collection in rig.users_collection for obj in collection.objects
            if obj.type=='MESH' and obj.find_armature()==rig}
    offsets={};counts={};sole_minima={}
    for side in ('L','R'):
        role=f'ankle_{side}';ankle=rig.data.bones[reverse[role]]
        foot_bones={ankle.name}|{b.name for b in ankle.children_recursive}
        heights=[]
        for obj in meshes:
            names={group.index:group.name for group in obj.vertex_groups if group.name in skin}
            foot_groups={index for index,name in names.items() if name in foot_bones}
            for vertex in obj.data.vertices:
                if sum(group.weight for group in vertex.groups if group.group in foot_groups)<0.8:continue
                point=obj.matrix_world@vertex.co
                if point.z>rest[ankle.name].translation.z+0.08:continue
                influences=[(names[group.group],group.weight) for group in vertex.groups if group.group in names and group.weight>0]
                total=sum(weight for _,weight in influences)
                height=sum((skin[name]@point).z*weight for name,weight in influences)/total
                heights.append(height)
        if len(heights)<10:raise ValueError(f'Cannot identify sole vertices for the {side} foot')
        sole_minima[side]=min(heights)
        offsets[role]=animated[role].to_quaternion().inverted()@Vector((0,0,-min(heights)))
        counts[role]=len(heights)
    if rig.get('deadlock_retarget_preserve_proportions'):
        # Lift the entire hierarchy once. Independent ankle/knee corrections
        # would change the avatar's joint spacing and stretch the hips again.
        offsets={'pelvis_root_shift':Vector((0,0,-min(sole_minima.values())))}
        rig['deadlock_retarget_standing_soles']=json.dumps({s:z+offsets['pelvis_root_shift'].z for s,z in sole_minima.items()})
    rig['deadlock_retarget_contact_offsets']=json.dumps({name:list(value) for name,value in offsets.items()})
    rig['deadlock_retarget_contact_vertices']=json.dumps(counts)
    rig['deadlock_retarget_contact_version']=CONTACT_VERSION
    return offsets
