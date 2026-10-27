"""Restore authored proportions on a preview and ground the whole posed avatar."""
import json

import bpy
from mathutils import Matrix


def preserve_proportions(rig, restore_original=True):
    from .preview import RESTORE_KEY, toggle_idle
    from .retarget import contact_offsets
    meshes=[o for c in rig.users_collection for o in c.objects
            if o.type=='MESH' and o.find_armature()==rig]
    originals=[]
    for mesh in meshes:
        original=bpy.data.objects.get(mesh.get('deadlock_original_mesh',''))
        if original is None or original.type!='MESH':
            raise ValueError(f'Original source mesh is unavailable for {mesh.name}')
        if [(g.name,g.index) for g in mesh.vertex_groups]!=[(g.name,g.index) for g in original.vertex_groups]:
            raise ValueError(f'Vertex groups changed on {mesh.name}; create a fresh original-rig preview')
        originals.append((mesh,original))
    sources={o.find_armature() for _,o in originals}
    if len(sources)!=1 or None in sources or rig in sources:
        raise ValueError('The preview must reference one intact original avatar rig')
    source=next(iter(sources))
    if set(source.data.bones.keys())!=set(rig.data.bones.keys()):
        raise ValueError('The source skeleton changed; create a fresh original-rig preview')
    reference={n:Matrix(m) for n,m in json.loads(rig['deadlock_retarget_reference']).items()}
    mapping=json.loads(rig['deadlock_retarget_mapping']);reverse={t:n for n,t in mapping.items()}
    # Match total leg reach with ONE scale for the entire avatar. This can make
    # the avatar taller, but never changes thigh/shin or torso/limb proportions.
    lengths=[]
    for side in ('L','R'):
        roles=[f'{stem}_{side}' for stem in ('leg_upper','leg_lower','ankle')]
        if any(r not in reverse for r in roles):raise ValueError('Both leg chains must be mapped')
        authored=[rig.matrix_world@source.data.bones[reverse[r]].head_local for r in roles]
        native=[reference[r].translation for r in roles]
        lengths.append((sum((b-a).length for a,b in zip(native,native[1:])),
                        sum((b-a).length for a,b in zip(authored,authored[1:]))))
    from .retarget_size import avatar_scale
    scale=sum(a for a,b in lengths)/sum(b for a,b in lengths)*avatar_scale(rig)
    if not 0.1<scale<10:raise ValueError('Invalid avatar leg scale')
    anchor=(reference['ankle_L'].translation+reference['ankle_R'].translation)*0.5
    fit=Matrix.Translation(anchor)@Matrix.Scale(scale,4)@Matrix.Translation(-anchor)
    rig_world=fit@rig.matrix_world
    mesh_world={o:fit@o.matrix_world for o in meshes}
    was_idle=RESTORE_KEY in rig
    if was_idle:del rig[RESTORE_KEY]
    # Restore only duplicated data; original meshes, keys, weights and rig stay intact.
    if restore_original:rig.data=source.data.copy()
    rig.matrix_world=rig_world
    rig.data.pose_position='POSE'
    for bone in rig.pose.bones:bone.matrix_basis=Matrix.Identity(4)
    for mesh,original in originals:
        if restore_original:
            values={k.name:k.value for k in mesh.data.shape_keys.key_blocks} if mesh.data.shape_keys else {}
            mesh.data=original.data.copy()
        mesh.matrix_world=mesh_world[mesh]
        if restore_original and mesh.data.shape_keys:
            mesh.data.shape_keys.animation_data_clear()
            for key in mesh.data.shape_keys.key_blocks:
                if key.name in values:key.value=values[key.name]
    for key in ('deadlock_retarget_leg_fit','deadlock_retarget_contact_offsets','deadlock_retarget_contact_vertices'):
        if key in rig:del rig[key]
    rig['deadlock_retarget_preserve_proportions']=True
    bpy.context.view_layer.update()
    offsets=contact_offsets(rig,reference)
    if was_idle:toggle_idle(bpy.context,bpy.context.scene.deadlock_export)
    return {'method':'Preserve authored proportions; ground the whole avatar',
            'original_rest_bone_lengths_preserved':True,'rest_mesh_modified':False,'weights_modified':False,
            'whole_pose_offset_meters':list(offsets['pelvis_root_shift']),
            'uniform_avatar_scale':rig.matrix_world.to_scale().x,
            'standing_sole_heights_meters':json.loads(rig['deadlock_retarget_standing_soles']),
            'leg_joint_positions_follow_native':False}
