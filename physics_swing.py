"""Native pitch/yaw jiggle bones behind the swing-only chain option.

JiggleBone and ClothFollowBone fields follow Valve's ModelDoc definitions and
Source 2 Viewer's ClothExtract.JiggleBones.cs / ClothExtract.Effects.cs.
Helpers align the native jiggle +Z axis with each original bone's +Y axis.
"""
import math


def export_swing(rows, lengths, parents, entry, unit):
    moving = {row['joint_name']: '_dlt_swing_' + row['joint_name']
              for row in rows if row['allow_rotation']}
    helpers, jiggles, cloth = [], [], []
    # Native flexible jiggles use separate pitch/yaw springs, rather than
    # cloth's normalized goal attraction. Keep their authored ranges bounded.
    stiffness = 1000.0 * entry['stiffness']
    damping = 10.0 * entry['damping']
    for row in rows:
        name = row['joint_name']
        cloth.append({'_class': 'ClothNode', 'name': 'avatar_swing_node_' + name,
                      'cloth_node_root_bone': name, 'is_static_node': True,
                      'allow_rotation': False, 'lock_translation': row['lock_translation']})
        if name not in moving:
            continue
        helper = moving[name]
        parent = parents[name]
        helpers.append({'name': helper, 'target': name, 'parent': moving.get(parent, parent)})
        jiggles.append({'_class': 'JiggleBone', 'name': 'avatar_swing_' + name,
                        'jiggle_root_bone': helper, 'jiggle_type': 1,
                        'length': lengths[name], 'tip_mass': entry['gravity'] * 9.80665 / unit,
                        'yaw_stiffness': stiffness, 'pitch_stiffness': stiffness,
                        'yaw_damping': damping, 'pitch_damping': damping,
                        'allow_flex_length': False, 'has_base_spring': False,
                        'has_yaw_constraint': False, 'has_pitch_constraint': False,
                        'has_angle_constraint': False, 'has_collision': False,
                        'invert_axes': False})
        # ClothFollowBone requires a simulated cloth leader. A JiggleBone
        # declaration alone does not register it during cloth resolution.
        cloth.append({'_class': 'ClothNode', 'name': 'avatar_swing_helper_' + name,
                      'cloth_node_root_bone': helper, 'is_static_node': False,
                      'allow_rotation': False})
        cloth.append({'_class': 'ClothFollowBone', 'name': 'avatar_swing_follow_' + name,
                      'leader_type': 0, 'leader_bone': helper, 'follower_bone': name})
    return {'helpers': helpers, 'jigglebones': jiggles, 'cloth': cloth}


def add_swing_helpers(edit_bones, nodes, fallback_parent):
    """Add only export-copy helpers; keep original bones and skinning intact."""
    from mathutils import Matrix
    created = []
    for node in nodes:
        for definition in node.get('_swing_only', {}).get('helpers', []):
            name, target = definition['name'], definition['target']
            if name in edit_bones:
                raise ValueError('Physics helper bone name is already in use: ' + name)
            original = edit_bones[target]
            matrix, length = original.matrix.copy(), original.length
            helper = edit_bones.new(name)
            # EditBone.matrix preserves length; seed a nonzero tail before
            # assigning its frame or Blender cannot retain the requested axes.
            helper.tail = (0, length, 0)
            helper.matrix = matrix @ Matrix.Rotation(-math.pi / 2, 4, 'X')
            helper.parent = edit_bones[definition['parent'] or fallback_parent]
            helper.use_deform = False
            created.append(name)
    return created


def validate_swing(fe, node, skeleton, check_rest):
    """Require the native motion controls and links when a user builds a VPK."""
    names = [str(name) for name in fe['m_CtrlName']]
    indices = {name: i for i, name in enumerate(names)}
    jiggles = {int(jiggle['m_nNode']): (i, jiggle)
               for i, jiggle in enumerate(fe.get('m_JiggleBones', []))}
    links = {(int(link['m_nParent']), int(link['m_nChild']))
             for link in fe.get('m_DynKinLinks', [])}
    rows = node['chain']['joints']
    authored = {jiggle['jiggle_root_bone']: jiggle
                for jiggle in node['_swing_only']['jigglebones']}
    for row in rows:
        name = row['joint_name']
        if name not in skeleton or name not in indices:
            raise ValueError('Compiled swing-only bone is missing: ' + name)
        check_rest(name)
        if not row['allow_rotation'] and indices[name] >= int(fe['m_nRotLockStaticNodes']):
            raise ValueError('Swing-only anchor rotation is not locked: ' + name)
    for helper in node['_swing_only']['helpers']:
        name, target = helper['name'], helper['target']
        if name not in skeleton or name not in indices or indices[name] not in jiggles:
            raise ValueError('Compiled swing-only jiggle bone is missing: ' + name)
        check_rest(name)
        _, jiggle = jiggles[indices[name]]
        settings = jiggle['m_jiggleBone']
        flags = int(settings['m_nFlags'])
        # Flexible, fixed length, no base spring or flipped coordinate frame.
        if (flags & 0x3FF) != 0x21:
            raise ValueError('Swing-only jiggle settings did not compile as requested: ' + name)
        for field, key in (('m_flLength', 'length'), ('m_flTipMass', 'tip_mass'),
                           ('m_flYawStiffness', 'yaw_stiffness'), ('m_flPitchStiffness', 'pitch_stiffness'),
                           ('m_flYawDamping', 'yaw_damping'), ('m_flPitchDamping', 'pitch_damping')):
            if not math.isclose(float(settings[field]), authored[name][key], rel_tol=1e-4, abs_tol=1e-4):
                raise ValueError('Swing-only jiggle parameter did not compile as requested: ' + name + ' ' + key)
        if (indices[name], indices[target]) not in links:
            raise ValueError('Compiled swing-only follower link is missing: ' + target)
        parent_index = indices.get(helper['parent'])
        expected_parent = jiggles[parent_index][0] if parent_index in jiggles else -1
        actual_parent = int(jiggle['m_nJiggleParent'])
        if actual_parent == 0xFFFFFFFF:
            actual_parent = -1
        if actual_parent != expected_parent:
            raise ValueError('Swing-only jiggle hierarchy did not compile as requested: ' + name)
    return {'root': node['root_bone'], 'bones': len(rows),
            'dynamic_bones': [row['joint_name'] for row in rows if row['simulate']],
            'anchored_bones': [row['joint_name'] for row in rows if not row['simulate']],
            'root_rotation_locked': not rows[0]['allow_rotation'],
            'lock_twist': True, 'simulation': 'native_jiggle_bones',
            'helper_bones': [helper['name'] for helper in node['_swing_only']['helpers']],
            'runtime_simulation_verified': False}
