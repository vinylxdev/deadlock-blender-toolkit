"""Manually authored secondary-motion chains compiled into Source 2 softbody data."""
import json
import math
import os
import tempfile
from pathlib import Path

KEY = 'deadlock_physics_chains'
PRESET_FORMAT = 'deadlock_physics_chains'
PRESET_VERSION = 1


def physics_rig(context, settings):
    candidates = [c for c in context.scene.collection.children_recursive
                  if c.get('deadlock_preview_only') and c.get('deadlock_hero') == settings.hero]
    selected = [c for c in candidates if context.object and context.object in c.all_objects[:]]
    if not (selected or candidates):
        return None
    return next((o for o in (selected or candidates)[-1].all_objects if o.type == 'ARMATURE'), None)


def chains(rig):
    return json.loads(rig.get(KEY, '[]')) if rig else []


def chain_bones(rig, root):
    bone = rig.data.bones.get(root)
    if bone is None:
        raise ValueError(f'Physics root bone no longer exists: {root}')
    members = [bone] + list(bone.children_recursive)
    mapped = json.loads(rig.get('deadlock_retarget_mapping', '{}'))
    conflicts = [b.name for b in members if b.name in mapped]
    if conflicts:
        raise ValueError('Choose an accessory chain without mapped animation bones: ' + ', '.join(conflicts[:5]))
    return members


def validated_entries(entries):
    """Validate portable settings independently of a Blender rig."""
    if not isinstance(entries, list):
        raise ValueError('Physics chains must be a list')
    result = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get('root'), str) or not entry['root']:
            raise ValueError('Each physics chain needs a root bone name')
        for key in ('stiffness', 'damping', 'gravity'):
            value = entry.get(key)
            maximum = 2 if key == 'gravity' else 1
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= maximum:
                raise ValueError(f'Invalid physics {key} for {entry["root"]}')
        normalized={key:entry[key] for key in ('root','stiffness','damping','gravity')}
        if 'twist_resistance' in entry:
            value=entry['twist_resistance']
            if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:
                raise ValueError(f'Invalid physics twist_resistance for {entry["root"]}')
            normalized['twist_resistance']=value
        for key in ('lock_root_rotation','anchor_child_roots','stable_rotation_frames'):
            if key in entry:
                if type(entry[key]) is not bool:
                    raise ValueError(f'Invalid physics {key} for {entry["root"]}')
                normalized[key]=entry[key]
        result.append(normalized)
    return result


def validate_chains(rig, entries=None):
    entries = validated_entries(chains(rig) if entries is None else entries)
    claimed = set()
    for entry in entries:
        members = chain_bones(rig, entry['root'])
        if entry.get('anchor_child_roots') and not any(b.parent!=members[0] for b in members[1:]):
            raise ValueError('Anchoring child roots needs lower descendant bones to simulate: '+entry['root'])
        overlap = claimed.intersection(b.name for b in members)
        if overlap:
            raise ValueError('Physics chains overlap at ' + ', '.join(sorted(overlap)[:5]))
        claimed.update(b.name for b in members)
    return entries


def save_preset(rig, filepath):
    """Save original bone names and motion settings without hero/export names."""
    entries = validate_chains(rig)
    document = {'format': PRESET_FORMAT, 'version': PRESET_VERSION, 'chains': entries}
    path = Path(filepath)
    text = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', suffix='.tmp', delete=False) as file:
            temporary = Path(file.name)
            file.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return len(entries)


def load_preset(rig, filepath):
    """Replace all chains only after the complete preset passes rig validation."""
    try:
        document = json.loads(Path(filepath).read_text(encoding='utf-8-sig'))
    except json.JSONDecodeError as error:
        raise ValueError(f'Invalid physics preset JSON: {error.msg}') from error
    if not isinstance(document, dict) or document.get('format') != PRESET_FORMAT:
        raise ValueError('This file is not a Deadlock physics chain preset')
    if type(document.get('version')) is not int or document['version'] != PRESET_VERSION:
        raise ValueError(f'Unsupported physics preset version: {document.get("version")}')
    entries = validate_chains(rig, validated_entries(document.get('chains')))
    rig[KEY] = json.dumps(entries)
    return len(entries)


def export_chains(rig, names):
    """Use a pinned root, full 3D joint bases and leaf tips in original bone axes."""
    from . import UNIT
    result = []
    for entry in validate_chains(rig):
        members = chain_bones(rig, entry['root'])
        rows = []
        for bone in members:
            anchored=bone==members[0] or (entry.get('anchor_child_roots',False) and bone.parent==members[0])
            length = ((rig.matrix_world @ bone.tail_local) - (rig.matrix_world @ bone.head_local)).length / UNIT
            if length < 1e-4:
                raise ValueError(f'Physics bone has no usable length: {bone.name}')
            rows.append({'joint_name': names[bone.name],
                         'joint_parent': names[bone.parent.name] if bone != members[0] else '',
                         'simulate': not anchored,
                         'allow_rotation': not anchored or (bone==members[0] and not entry.get('lock_root_rotation',False)),
                         'lock_translation': anchored,
                         'end_effector': length if not bone.children else 0.0,
                         'extrude_radius': max(0.1, length * 0.1),
                         'extrude_sides':3 if entry.get('stable_rotation_frames',False) else 2,
                         'goal_strength': entry['stiffness'], 'goal_damping': entry['damping'],
                         'torsion_spring':entry.get('twist_resistance',0.2),
                         'gravity_z': entry['gravity']})
        defaults = {
            'joint_name': '', 'joint_parent': '', 'simulate': True, 'allow_rotation': True,
            'lock_translation': False, 'stretch_spring': 1.0, 'bend_spring': 0.8,
            'torsion_spring': 0.2, 'goal_strength': 0.5, 'goal_damping': 0.5,
            'gravity_z': 0.3, 'drag': 0.2, 'mass': 1.0, 'world_collision': False,
            'end_effector': 0.0, 'extrude_sides': 2, 'extrude_radius': 1.0,
            'extrude_forward_axis': 'y', 'twist_relax': 1.0, 'extra_iterations': 2,
            'collision_radius': 0.0, 'antishrink': 1.0,
        }
        # Original Blender bones run along +Y. Extruded bases with twist_relax
        # avoid the legacy rope solver's +X-only alignment requirement.
        attrs = {key: {'display': key, 'default': value, 'show': True, 'ui_order': i}
                 for i, (key, value) in enumerate(defaults.items())}
        # ModelDoc treats ':' in a node name as a path qualifier and silently
        # drops the chain. Export names are already safe, unique identifiers.
        result.append({'_class': 'ClothChain', 'name': 'avatar_physics_' + names[entry['root']],
                       'root_bone': names[entry['root']],
                       'chain': {'version': 2, 'joints': rows, 'attrs': attrs, 'selection': []}})
    return result


def append_physics_modeldoc(document, nodes):
    if not nodes:
        return document
    from .core import class_spans
    def kv(value):
        if isinstance(value, dict):
            return '{ ' + ' '.join(key + ' = ' + kv(child) for key, child in value.items()) + ' }'
        if isinstance(value, list):
            return '[ ' + ', '.join(kv(child) for child in value) + ' ]'
        return json.dumps(value, allow_nan=False)
    # A single Softbody contains native and avatar chains, when the hero has one.
    spans = [(a, b) for a, b, cls in class_spans(document) if cls == 'Softbody']
    if len(spans) > 1:
        raise ValueError('Multiple native Softbody nodes require manual ModelDoc preparation')
    if spans:
        start, end = spans[0]
        fragment = document[start:end]
        import re
        children = re.search(r'\bchildren\s*=\s*\[', fragment)
        if children is None:
            raise ValueError('Native Softbody has no children list')
        offset = start + children.end()
        addition = ', '.join(kv(node) for node in nodes) + ', '
    else:
        import re
        root = re.search(r'\brootNode\s*=\s*\{', document)
        if root is None:
            raise ValueError('ModelDoc root node is missing')
        children = re.search(r'\bchildren\s*=\s*\[', document[root.end():])
        if children is None:
            raise ValueError('ModelDoc root children are missing')
        offset = root.end() + children.end()
        params = {'_class': 'ClothParams', 'new_style': True, 'unitless_damping': True,
                  'default_gravity_scale': 1.0, 'default_vel_air_drag': 0.1, 'extra_iterations': 2}
        addition = kv({'_class': 'Softbody', 'children': [params] + nodes}) + ', '
    return document[:offset] + addition + document[offset:]


def validate_compiled_physics(compiled, path, nodes):
    """Require real compiled dynamic nodes and correct skeleton bindings."""
    if not nodes:
        return {'enabled': False, 'chains': []}
    from . import read_resource, bone_matrices, UNIT
    from mathutils import Quaternion, Vector
    resource = read_resource(compiled, path, model=True)
    phys = resource.get_block(None, block_name='PHYS')
    fe = phys.get('m_pFeModel') if phys else None
    if not fe or int(fe['m_nNodeCount']) <= int(fe['m_nStaticNodes']):
        raise ValueError('Compiler did not produce dynamic physics nodes')
    names = [str(n) for n in fe['m_CtrlName']]
    static = int(fe['m_nStaticNodes'])
    skeleton = {bone.name for bone in resource.get_bones()}
    rest = bone_matrices(resource.get_bones())
    maximum_rest_error = 0.0
    checked = []
    for node in nodes:
        rows = node['chain']['joints']
        root = node['root_bone']
        if root not in names or names.index(root) >= static:
            raise ValueError(f'Physics chain root is not anchored: {root}')
        rotation_locked=names.index(root)<int(fe['m_nRotLockStaticNodes'])
        if rotation_locked!=(not rows[0]['allow_rotation']):
            raise ValueError(f'Physics chain root rotation lock did not compile as requested: {root}')
        dynamic = [row['joint_name'] for row in rows if row['simulate']]
        for name in dynamic:
            if name not in skeleton or name not in names or names.index(name) < static:
                raise ValueError(f'Compiled physics bone is missing or static: {name}')
        for row in rows:
            name = row['joint_name']
            if not row['simulate']:
                index=names.index(name)
                if index>=static or (index<int(fe['m_nRotLockStaticNodes']))!=(not row['allow_rotation']):
                    raise ValueError(f'Physics anchor or rotation lock did not compile as requested: {name}')
            if row['simulate'] and row.get('extrude_sides',2)>2:
                prefix='$cc'+name+'_'
                proxies=sum(n.startswith(prefix) and i>=static for i,n in enumerate(names))
                if proxies<row['extrude_sides']:
                    raise ValueError(f'Physics rotation frame is missing support nodes: {name}')
            pose = fe['m_InitPose'][names.index(name)]
            position_error = (Vector(pose[:3]) * UNIT - rest[name].translation).length
            rotation = Quaternion((float(pose[7]), float(pose[4]), float(pose[5]), float(pose[6])))
            if position_error > 0.0002 or abs(rotation.dot(rest[name].to_quaternion())) < 0.99999:
                raise ValueError(f'Physics rest pose differs from the original bone: {name}')
            maximum_rest_error = max(maximum_rest_error, position_error)
        for row in rows:
            if row['end_effector'] > 0:
                prefix = '$cc' + row['joint_name'] + '_'
                if not any(n.startswith(prefix) and i >= static for i, n in enumerate(names)):
                    raise ValueError(f'Physics chain has no simulated leaf tip: {row["joint_name"]}')
        checked.append({'root': root, 'dynamic_bones': dynamic, 'bones': len(rows),
                        'root_rotation_locked':rotation_locked,
                        'anchored_bones':[row['joint_name'] for row in rows if not row['simulate']],
                        'rotation_frame_points':rows[0].get('extrude_sides',2),
                        'twist_resistance':rows[0].get('torsion_spring',0.2)})
    return {'enabled': True, 'chains': checked, 'compiled_nodes': int(fe['m_nNodeCount']),
            'static_nodes': static, 'runtime_simulation_verified': False,
            'max_physics_rest_position_error_meters': maximum_rest_error,
            'collision': 'No body or world collision; accessory secondary motion only'}
