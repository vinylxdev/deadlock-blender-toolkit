"""Check the serialized skeleton, runtime rules and compiled skinning against the preview."""
import json
from types import SimpleNamespace

import numpy as np
import bpy
from mathutils import Matrix, Quaternion, Vector
from mathutils.kdtree import KDTree

from .retarget import retarget_bind_rotations, position_roles, aim_rotation, forearm_uses_aim
from .retarget_export import unit_matrix, runtime_root_reference


def native_idle(reference, native_bones, path, time_seconds=0.0, model=None):
    from .preview import idle_matrices
    bones={b.name:SimpleNamespace(name=b.name,matrix_local=reference[b.name],parent=None) for b in native_bones}
    for bone in native_bones: bones[bone.name].parent=bones.get(bone.parent)
    basis=idle_matrices(SimpleNamespace(data=SimpleNamespace(bones=list(bones.values()))),path,time_seconds,model)
    pose={};pending=list(bones.values())
    while pending:
        for bone in pending[:]:
            if bone.parent and bone.parent.name not in pose: continue
            local=bone.parent.matrix_local.inverted()@bone.matrix_local if bone.parent else bone.matrix_local
            pose[bone.name]=(pose[bone.parent.name] if bone.parent else Matrix.Identity(4))@local@basis[bone.name]
            pending.remove(bone)
    return pose


def evaluate_rules(rest, native_pose, constraints, skeleton=None):
    """Evaluate the authored point/orient/aim constraints in Source 2 order."""
    from . import UNIT
    pose={name:matrix.copy() for name,matrix in rest.items()}
    pose.update({name:matrix.copy() for name,matrix in native_pose.items()})
    children={name:[] for name in rest}
    for name,bone in (skeleton or {}).items():
        if bone['parent'] in children:children[bone['parent']].append(name)
    descendants={}
    def descendants_of(name):
        if name not in descendants:
            descendants[name]=[n for child in children[name] for n in [child]+descendants_of(child)]
        return descendants[name]
    for rule in constraints:
        target=rule['m_targets'][0];slave=rule['m_slaves'][0]
        name=slave['m_sName'];source=pose[target['m_sName']]
        current=pose[name];weight=target['m_flWeight']*slave['m_flWeight']
        previous=current.copy()
        if rule['_class']=='CPointConstraint':
            # The target matrix has meter translation; scale only the authored offset.
            total=sum(t['m_flWeight'] for t in rule['m_targets'])
            position=sum(((pose[t['m_sName']]@(Vector(t['m_vOffset'])*UNIT))*t['m_flWeight']
                          for t in rule['m_targets']),Vector((0,0,0)))/total
            current.translation=current.translation.lerp(position,total*slave['m_flWeight'])
        elif rule['_class']=='COrientConstraint':
            q=target['m_qOffset']
            rotation=source.to_quaternion()@Quaternion((q[3],q[0],q[1],q[2]))
            rotation=current.to_quaternion().slerp(rotation,weight)
            pose[name]=Matrix.LocRotScale(current.translation,rotation,Vector((1,1,1)))
        elif rule['_class']=='CAimConstraint':
            targets=rule['m_targets']
            total=sum(t['m_flWeight'] for t in targets[:-1])
            position=sum(((pose[t['m_sName']]@(Vector(t['m_vOffset'])*UNIT))*t['m_flWeight']
                         for t in targets[:-1]),Vector((0,0,0)))/total
            up_target=targets[-1]
            up=pose[up_target['m_sName']].to_quaternion()@Vector(rule['m_vUpVector'])
            rotation=aim_rotation(position-current.translation,up)
            pose[name]=Matrix.LocRotScale(current.translation,rotation,Vector((1,1,1)))
        else: raise ValueError('Unsupported exported retarget rule')
        if skeleton:
            delta=pose[name]@previous.inverted()
            q=delta.to_quaternion()
            near_identity=(max(abs(x) for x in delta.translation)<=0.0001*UNIT
                           and max(abs(q.x),abs(q.y),abs(q.z))<=0.1 and abs(abs(q.w)-1)<=0.1)
            if not near_identity:
                for child in descendants_of(name):pose[child]=delta@pose[child]
    return pose


def expected_pose(rig, reference, animated, mapping, runtime_roots=None):
    bind=retarget_bind_rotations(rig,reference,mapping)
    rest={b.name:unit_matrix(rig.matrix_world@b.matrix_local) for b in rig.data.bones}
    position_offsets=json.loads(rig.get('deadlock_retarget_contact_offsets','{}'))
    positions=position_roles(rig)
    expected=dict(animated)
    original_names=set(json.loads(rig['deadlock_retarget_original_names']).values())
    pending=[b for b in rig.data.bones if b.name in original_names]
    while pending:
        for bone in pending[:]:
            if bone.parent and bone.parent.name not in expected: continue
            parent_rest=(runtime_roots or {}).get(bone.parent.name,rest[bone.parent.name]) if bone.parent else None
            local=parent_rest.inverted()@rest[bone.name] if parent_rest is not None else rest[bone.name]
            parent=expected[bone.parent.name] if bone.parent else Matrix.Identity(4)
            base=parent@local
            role=mapping.get(bone.name)
            if role:
                delta=animated[role].to_quaternion()@reference[role].to_quaternion().inverted()
                if role=='neck_0' and 'head' in animated:
                    head=animated['head'].to_quaternion()@reference['head'].to_quaternion().inverted()
                    delta=delta.slerp(head,0.5)
                if forearm_uses_aim(rig,bone.name,mapping):
                    hand=f'hand_{role[-1]}'
                    hand_delta=animated[hand].to_quaternion()@reference[hand].to_quaternion().inverted()
                    delta=delta.slerp(hand_delta,0.5)
                rotation=delta@bind[bone.name]
                position=base.translation
                if role=='pelvis':
                    offset=reference[role].inverted()@rest[bone.name].translation
                    position=animated[role]@offset
                    if rig.get('deadlock_retarget_preserve_proportions'):
                        root='root_motion' if 'root_motion' in reference else next(iter(reference))
                        root_rest=(runtime_roots or {}).get(root,reference[root])
                        delta_root=animated[root].to_quaternion()@root_rest.to_quaternion().inverted()
                        position+=delta_root@Vector(position_offsets.get('pelvis_root_shift',(0,0,0)))
                elif role in positions:
                    position=animated[role]@Vector(position_offsets.get(role,(0,0,0)))
                if forearm_uses_aim(rig,bone.name,mapping):
                    hand=f'hand_{role[-1]}'
                    rotation=aim_rotation(animated[hand].translation-animated[role].translation,rotation@Vector((0,0,1)))
                elif rig.get('deadlock_retarget_leg_fit') and role.startswith(('leg_upper_','leg_lower_')):
                    child=f'leg_lower_{role[-1]}' if role.startswith('leg_upper_') else f'ankle_{role[-1]}'
                    point=animated[child]@Vector(position_offsets.get(child,(0,0,0)))
                    rotation=aim_rotation(point-position,rotation@Vector((0,0,1)))
                expected[bone.name]=Matrix.LocRotScale(position,rotation,Vector((1,1,1)))
            else: expected[bone.name]=base
            pending.remove(bone)
    return expected


def validate_body_motion(rig, rest, reference, native_bones, skeleton, constraints, runtime_roots, settings):
    from .preview import animation_sources, idle_source
    from SourceIO.library.utils import datamodel
    mapping=json.loads(rig['deadlock_retarget_mapping'])
    hip=next(name for name,role in mapping.items() if role=='pelvis')
    original_names=set(json.loads(rig['deadlock_retarget_original_names']).values())
    native_rest={name:rest[name].copy() for name in reference}
    native_rest.update(runtime_roots)
    baseline=evaluate_rules(rest,native_rest,constraints,skeleton)[hip].translation.copy()
    translation=Vector((0.12,-0.08,-0.4))
    shifted={name:matrix.copy() for name,matrix in native_rest.items()}
    shifted['pelvis'].translation+=translation
    transferred=evaluate_rules(rest,shifted,constraints,skeleton)[hip].translation-baseline
    motion_error=(transferred-translation).length
    if motion_error>0.0002:
        raise ValueError(f'Avatar ignores native pelvis translation (motion error {motion_error:.6f} m)')
    sources=animation_sources(settings)
    idle_label,_=idle_source(settings)
    labels=[idle_label]+[name for name in ('primary_crouch_idle','crouch_idle','weapon_crouch_idle','slide_forward','slide_intro','slide_getup')
                         if name in sources and name!=idle_label]
    results={}
    for label in labels:
        path=sources[label]
        model=datamodel.load(str(path))
        duration=max((float(layer['times'][-1]) for element in model.elements if element.type=='DmeChannel'
                      for layer in element['log']['layers'] if layer['times']),default=0.0)
        times=np.linspace(0,duration,9) if duration else [0.0]
        positions=[];native_positions=[];ankle_offsets=[]
        max_position=max_rotation=0.0
        for time in times:
            animated=native_idle({n:rest[n] for n in reference},native_bones,path,float(time),model)
            for root,runtime_reference in runtime_roots.items():
                animated[root]=animated[root]@rest[root].inverted()@runtime_reference
            actual=evaluate_rules(rest,animated,constraints,skeleton)
            expected=expected_pose(rig,reference,animated,mapping,runtime_roots)
            max_position=max(max_position,max((actual[n].translation-expected[n].translation).length for n in original_names))
            max_rotation=max(max_rotation,max(max(abs(actual[n].to_3x3()[i][j]-expected[n].to_3x3()[i][j])
                                                   for i in range(3) for j in range(3)) for n in original_names))
            positions.append(list(actual[hip].translation));native_positions.append(list(animated['pelvis'].translation))
            reverse={role:name for name,role in mapping.items()}
            ankle_offsets.append({side:list(actual[reverse[f'ankle_{side}']].translation-animated[f'ankle_{side}'].translation)
                                  for side in ('L','R')})
        if max(max_position,max_rotation)>0.0003:
            raise ValueError(f'Retarget animation {label} differs from expected pose: position {max_position:.6f} m, rotation {max_rotation:.6f}')
        results[label]={'sampled_frames':len(times),'duration_seconds':duration,
                        'max_joint_position_error_meters':max_position,'max_joint_rotation_matrix_error':max_rotation,
                        'avatar_pelvis_range_meters':np.ptp(np.asarray(positions),axis=0).tolist(),
                        'native_pelvis_range_meters':np.ptp(np.asarray(native_positions),axis=0).tolist(),
                        'avatar_pelvis_positions_meters':positions,'native_pelvis_positions_meters':native_positions,
                        'ankle_offsets_meters':ankle_offsets}
    return {'pelvis_translation_error_meters':motion_error,'animation_clips':results}


def validate_weapon_motion(rig,rest,reference,native_bones,skeleton,constraints,runtime_roots,settings):
    """Check the weapon, attachment hand and muzzle across complete native clips."""
    from .preview import animation_sources,idle_source
    from SourceIO.library.utils import datamodel
    binding=json.loads(rig.get('deadlock_retarget_weapon_binding','null'))
    if not binding:return {'enabled':False}
    mapping=json.loads(rig['deadlock_retarget_mapping'])
    sources=animation_sources(settings);idle,_=idle_source(settings)
    labels=list(dict.fromkeys([idle]+[name for name in ('weapon_crouch_idle','shoot_idle','aim_weapon_idle',
                                                       'weapon_run_n','weapon_run_e') if name in sources]))
    driven=set(binding['bones'])|set(binding['hand_bones'])
    results={}
    for label in labels:
        path=sources[label];model=datamodel.load(str(path))
        duration=max((float(layer['times'][-1]) for e in model.elements if e.type=='DmeChannel'
                      for layer in e['log']['layers'] if layer['times']),default=0.0)
        maximum_position=maximum_rotation=0.0;before=after=0.0
        times=np.linspace(0,duration,9) if duration else [0.0]
        for time in times:
            native=native_idle({n:rest[n] for n in reference},native_bones,path,float(time),model)
            for root,matrix in runtime_roots.items():native[root]=native[root]@rest[root].inverted()@matrix
            expected=expected_pose(rig,reference,native,mapping,runtime_roots)
            actual=evaluate_rules(rest,native,constraints,skeleton)
            shift=expected[binding['avatar_hand']].translation-native[binding['native_hand']].translation
            before=max(before,shift.length)
            after=max(after,(actual[binding['native_hand']].translation-
                             expected[binding['avatar_hand']].translation).length)
            for name in driven:
                maximum_position=max(maximum_position,(actual[name].translation-native[name].translation-shift).length)
                maximum_rotation=max(maximum_rotation,max(abs(actual[name][i][j]-native[name][i][j])
                                                           for i in range(3) for j in range(3)))
        if max(maximum_position,maximum_rotation,after)>0.00003:
            raise ValueError(f'Weapon grip or muzzle drift in {label}: position {maximum_position:.6f} m, rotation {maximum_rotation:.6f}, grip {after:.6f} m')
        results[label]={'sampled_frames':len(times),'max_weapon_position_error_meters':maximum_position,
                        'max_weapon_rotation_matrix_error':maximum_rotation,
                        'max_original_grip_offset_meters':before,'max_corrected_grip_error_meters':after}
    return {'enabled':True,'weapon_root':binding['root'],'native_attachment_hand':binding['native_hand'],
            'avatar_hand':binding['avatar_hand'],'weapon_and_hand_bones_checked':len(driven),
            'muzzle_and_reload_offsets_preserved':True,'animation_clips':results}


def validate_retarget_export(compiled,path,collection,skeleton,constraints,settings):
    from . import read_resource,bone_matrices,reference_bones,UNIT
    from .preview import idle_source
    from SourceIO.library.source2.blocks.vertex_index_buffer.vertex_buffer import VertexBuffer
    resource=read_resource(compiled,path,model=True)
    bones=resource.get_bones();by_name={b.name:b for b in bones};rest=bone_matrices(bones)
    maximum=0.0
    for name,bone in skeleton.items():
        if name not in by_name or by_name[name].parent!=bone['parent']:
            raise ValueError(f'Compiled retarget bone/parent missing: {name}')
        maximum=max(maximum,max(abs(rest[name][i][j]-bone['matrix'][i][j]) for i in range(4) for j in range(4)))
    if maximum>0.0002: raise ValueError(f'Compiled avatar rest skeleton differs by {maximum:.6f}')
    names=list(skeleton)
    for rule in constraints:
        for item in rule['m_slaves']+rule['m_targets']:
            if item['m_sName'] not in rest: raise ValueError(f"Retarget rule references missing bone {item['m_sName']}")
    rig=next(o for o in collection.objects if o.type=='ARMATURE')
    mapping=json.loads(rig['deadlock_retarget_mapping'])
    original_names=set(json.loads(rig['deadlock_retarget_original_names']).values())
    reference={n:Matrix(m) for n,m in json.loads(rig['deadlock_retarget_reference']).items()}
    label,idle_path=idle_source(settings)
    animated=native_idle({n:rest[n] for n in reference},reference_bones(settings),idle_path)
    roots={skeleton[name]['parent'] for name in original_names if skeleton[name]['parent'] not in original_names}
    runtime_roots={root:runtime_root_reference(settings,reference,root) for root in roots}
    # Test the actual graph's rest frame and a translated/yawed actor, rather
    # than accepting only the differently oriented decompiled DMX root.
    avatar_roots=[name for name in original_names if skeleton[name]['parent'] in runtime_roots]
    native_rest={name:rest[name].copy() for name in reference}
    native_rest.update(runtime_roots)
    root_error=0.0
    motions=(Matrix.Identity(4),Matrix.Translation(Vector((0.3,-0.2,0.5)))@Matrix.Rotation(0.7,4,'Z'))
    for motion in motions:
        runtime_pose=evaluate_rules(rest,{name:motion@matrix for name,matrix in native_rest.items()},constraints,skeleton)
        lift=Vector(json.loads(rig.get('deadlock_retarget_contact_offsets','{}')).get('pelvis_root_shift',(0,0,0)))
        root_error=max(root_error,max((runtime_pose[name].translation-motion@(skeleton[name]['matrix'].translation+
                                      (lift if mapping.get(name)=='pelvis' else Vector((0,0,0))))).length
                                      for name in avatar_roots))
    if root_error>0.0002:
        raise ValueError(f'Compiled avatar root uses the wrong animation frame (position error {root_error:.6f} m)')
    body_motion=validate_body_motion(rig,rest,reference,reference_bones(settings),skeleton,constraints,runtime_roots,settings)
    weapon_motion=validate_weapon_motion(rig,rest,reference,reference_bones(settings),skeleton,constraints,runtime_roots,settings)
    for root,runtime_reference in runtime_roots.items():
        # VRF applies the export-axis correction on the right of the NM root
        # rotation. Undo it here; other bones retain their model-space pose.
        animated[root]=animated[root]@rest[root].inverted()@runtime_reference
    actual=evaluate_rules(rest,animated,constraints,skeleton)
    expected=expected_pose(rig,reference,animated,mapping,runtime_roots)
    position_error=max((actual[n].translation-expected[n].translation).length for n in original_names)
    rotation_error=max(max(abs(actual[n].to_3x3()[i][j]-expected[n].to_3x3()[i][j]) for i in range(3) for j in range(3)) for n in original_names)
    if max(position_error,rotation_error)>0.0003:
        raise ValueError(f'Compiled retarget rules differ from preview: joint position {position_error:.6f} m, rotation matrix {rotation_error:.6f}')

    # Compare actual compiled four-influence skinning with all original weights.
    rest_avatar={b.name:unit_matrix(rig.matrix_world@b.matrix_local) for b in rig.data.bones}
    palette=np.asarray([np.asarray(expected[n]@rest_avatar[n].inverted()) for n in names])
    bone_index={name:i for i,name in enumerate(names)}
    expected_points=[]
    depsgraph=bpy.context.evaluated_depsgraph_get()
    for obj in collection.objects:
        if obj.type!='MESH': continue
        evaluated=obj.evaluated_get(depsgraph)
        mesh=evaluated.to_mesh(preserve_all_data_layers=True,depsgraph=depsgraph)
        vertices=np.asarray([obj.matrix_world@v.co for v in mesh.vertices])
        homogeneous=np.column_stack((vertices,np.ones(len(vertices))))
        group_bones={g.index:bone_index[g.name] for g in obj.vertex_groups if g.name in bone_index}
        influences=[(v.index,group_bones[g.group],g.weight) for v in mesh.vertices for g in v.groups
                    if g.group in group_bones and g.weight>0]
        evaluated.to_mesh_clear()
        points=np.zeros_like(vertices);totals=np.zeros(len(vertices))
        if influences:
            vertex_indices=np.asarray([x[0] for x in influences]);indices=np.asarray([x[1] for x in influences]);weights=np.asarray([x[2] for x in influences])
            transformed=np.einsum('nij,nj->ni',palette[indices],homogeneous[vertex_indices])[:,:3]
            np.add.at(points,vertex_indices,transformed*weights[:,None]);np.add.at(totals,vertex_indices,weights)
            weighted=totals>0;points[weighted]/=totals[weighted,None];points[~weighted]=vertices[~weighted]
        else: points=vertices
        expected_points.extend(points)
    tree=KDTree(len(expected_points))
    for i,point in enumerate(expected_points): tree.insert(point,i)
    tree.balance()
    ctrl=resource.get_block(None,block_name='CTRL')
    mesh=next(row for row in ctrl['embedded_meshes'] if row.get('m_Name') in ('deadlock_toolkit','vinylrex'))
    mdat=resource.get_block(None,block_id=mesh['m_nDataBlock'])
    mesh_bones=mdat['m_skeleton']['m_bones']
    skin=[]
    for bone in mesh_bones:
        inv=np.asarray(bone['m_invBindPose']).reshape(3,4)
        inverse=np.eye(4);inverse[:3]=inv;inverse[:3,3]*=UNIT
        skin.append(np.asarray(actual[str(bone['m_boneName'])])@inverse)
    skin=np.asarray(skin)
    errors=[];vertex_count=0;weighted_bones=set()
    reverse={role:name for name,role in mapping.items()}
    sole_heights={side:[] for side in ('L','R')}
    foot_indices={}
    for side in sole_heights:
        ankle=rig.data.bones[reverse[f'ankle_{side}']]
        names_for_foot={ankle.name}|{b.name for b in ankle.children_recursive}
        foot_indices[side]=[i for i,b in enumerate(mesh_bones) if str(b['m_boneName']) in names_for_foot]
    for row in mesh['m_vertexBuffers']:
        vertex_buffer=VertexBuffer.from_kv(row).get_vertices(resource)
        if 'POSITION' not in vertex_buffer.dtype.names:continue
        weight_field=next((n for n in vertex_buffer.dtype.names if n.startswith('BLENDWEIGHT')),None)
        if 'BLENDINDICES' not in vertex_buffer.dtype.names or weight_field is None:
            raise ValueError('Compiled avatar lost skinning data')
        indices=vertex_buffer['BLENDINDICES'].astype(np.int64)
        weights=vertex_buffer[weight_field].astype(float)
        if np.any(indices<0) or np.any(indices>=len(skin)):raise ValueError('Compiled skinning references invalid bone indices')
        totals=weights.sum(axis=1)
        if np.any(totals<=0):raise ValueError('Compiled avatar has unweighted vertices')
        weights/=totals[:,None]
        weighted_bones.update(str(mesh_bones[int(i)]['m_boneName']) for i in np.unique(indices[weights>0]))
        positions=vertex_buffer['POSITION']*UNIT
        for side in sole_heights:
            foot_weight=(np.isin(indices,foot_indices[side])*weights).sum(axis=1)
            mask=(foot_weight>=0.8)&(positions[:,2]<=rest_avatar[reverse[f'ankle_{side}']].translation.z+0.08)
            if mask.any():
                points=np.column_stack((positions[mask],np.ones(int(mask.sum()))))
                transformed=np.einsum('nkij,nj->nki',skin[indices[mask]],points)
                foot=(transformed[:,:,:3]*weights[mask,:,None]).sum(axis=1)
                sole_heights[side].extend(foot[:,2].tolist())
        sample=np.linspace(0,len(positions)-1,min(4096,len(positions)),dtype=int)
        homogeneous=np.column_stack((positions[sample],np.ones(len(sample))))
        transformed=np.einsum('nkij,nj->nki',skin[indices[sample]],homogeneous)
        posed=(transformed[:,:,:3]*weights[sample,:,None]).sum(axis=1)
        errors.extend(tree.find(point)[2] for point in posed)
        vertex_count+=len(positions)
    if not weighted_bones.issubset(original_names):raise ValueError('Avatar vertices were rebound to native hero bones')
    if any(not heights for heights in sole_heights.values()):raise ValueError('Compiled avatar sole vertices are missing')
    soles={side:min(heights) for side,heights in sole_heights.items()}
    ground_error=(abs(min(soles.values())) if rig.get('deadlock_retarget_preserve_proportions')
                  else max(abs(height) for height in soles.values()))
    if ground_error>0.006:
        raise ValueError(f'Compiled idle feet miss the ground plane: {soles}')
    percentile=float(np.percentile(errors,99));skin_error=max(errors)
    if skin_error>0.03 or percentile>0.005:
        raise ValueError(f'Compiled skinning differs visibly from the retarget preview: maximum {skin_error:.4f} m, 99th percentile {percentile:.4f} m')
    return {'avatar_bones_preserved':len(original_names),'total_bones':len(bones),
            'runtime_constraints_verified':len(constraints),'max_rest_matrix_error':maximum,
            'runtime_root_frame_verified':True,
            'max_runtime_root_position_error_meters':root_error,
            'runtime_root_translation_and_yaw_verified':True,
            'body_motion_validation':body_motion,
            'weapon_motion_validation':weapon_motion,
            'compiled_idle_sole_heights_meters':soles,
            'idle_animation':label,'max_idle_joint_position_error_meters':position_error,
            'max_idle_joint_rotation_matrix_error':rotation_error,
            'compiled_vertices':vertex_count,'sampled_posed_vertices':len(errors),
            'compiled_weighted_avatar_bones':len(weighted_bones),'compiled_max_influences':4,
            'max_sampled_idle_skin_error_meters':skin_error,'p99_idle_skin_error_meters':percentile,
            'in_game_verified':False}
