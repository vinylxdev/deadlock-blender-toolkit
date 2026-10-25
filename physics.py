"""Manually authored secondary-motion chains compiled into Source 2 softbody data."""
import json
import math
import os
import tempfile
from pathlib import Path

from .physics_curve import CURVE_SETTINGS, SETTINGS, validated_curve

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
        for key in CURVE_SETTINGS:
            if key + '_curve' in entry:
                normalized[key + '_curve'] = validated_curve(entry[key + '_curve'])
        for key, maximum in (('motion_limit', None), ('motion_limit_strength', 1)):
            if key in entry:
                value = entry[key]
                if (type(value) not in (int, float) or not math.isfinite(value) or value < 0
                        or (maximum is not None and value > maximum)):
                    raise ValueError(f'Invalid physics {key} for {entry["root"]}')
                normalized[key] = value
        if 'twist_resistance' in entry:
            value=entry['twist_resistance']
            if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:
                raise ValueError(f'Invalid physics twist_resistance for {entry["root"]}')
            normalized['twist_resistance']=value
        for key in ('lock_root_rotation','anchor_child_roots','stable_rotation_frames','lock_twist','use_motion_limits','auto_collision'):
            if key in entry:
                if type(entry[key]) is not bool:
                    raise ValueError(f'Invalid physics {key} for {entry["root"]}')
                normalized[key]=entry[key]
        if normalized.get('use_motion_limits') and normalized.get('lock_twist'):
            raise ValueError('Motion limits require cloth simulation; turn off Lock twist for ' + entry['root'])
        if normalized.get('auto_collision') and normalized.get('lock_twist'):
            raise ValueError('Automatic body collisions require cloth simulation; turn off Lock twist for ' + entry['root'])
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
        values, positions = {}, None
        settings = dict(SETTINGS)
        if entry.get('use_motion_limits'):
            settings.update(motion_limit=('stray_radius', 0.01),
                            motion_limit_strength=('stray_radius_stretchiness', 1.0))
        for key, (field, default) in settings.items():
            base = float(entry.get(key, default))
            values[field] = {bone.name: base for bone in members}
            curve = entry.get(key + '_curve')
            if curve and curve['enabled']:
                from .physics_curve import chain_positions, multipliers
                if positions is None:
                    positions = chain_positions(members, rig.matrix_world)
                scales = multipliers(curve, [positions[bone.name] for bone in members])
                values[field] = {bone.name: base * scale for bone, scale in zip(members, scales)}
        if entry.get('use_motion_limits'):
            for bone in members:
                radius = values['stray_radius'][bone.name]
                strength = values['stray_radius_stretchiness'][bone.name]
                # Source 2 treats a zero radius or full stretchiness as no
                # constraint. Distances in presets and Blender are in meters.
                active = radius > 0 and strength > 0
                values['stray_radius'][bone.name] = radius / UNIT if active else 0.0
                values['stray_radius_stretchiness'][bone.name] = 1.0 - strength if active else 0.0
        if entry.get('auto_collision'):
            from .physics_collision import VERTEX_MAP
            mapped = json.loads(rig.get('deadlock_retarget_mapping', '{}'))
            heights = [(rig.matrix_world @ rig.data.bones[name].head_local).z for name in mapped if name in rig.data.bones]
            thickness = max(max(heights, default=1) - min(heights, default=0), 0.01) * 0.0015 / UNIT
        rows = []
        lengths, parents = {}, {}
        for bone in members:
            anchored=bone==members[0] or (entry.get('anchor_child_roots',False) and bone.parent==members[0])
            length = ((rig.matrix_world @ bone.tail_local) - (rig.matrix_world @ bone.head_local)).length / UNIT
            if length < 1e-4:
                raise ValueError(f'Physics bone has no usable length: {bone.name}')
            lengths[names[bone.name]] = length
            parents[names[bone.name]] = names[bone.parent.name] if bone.parent else ''
            rows.append({'joint_name': names[bone.name],
                         'joint_parent': names[bone.parent.name] if bone != members[0] else '',
                         'simulate': not anchored,
                         'allow_rotation': not anchored or (bone==members[0] and not entry.get('lock_root_rotation',False)),
                         'lock_translation': anchored,
                         'end_effector': length if not bone.children else 0.0,
                         'extrude_radius': max(0.1, length * 0.1),
                         'extrude_sides':3 if entry.get('stable_rotation_frames',False) else 2,
                         **{field: per_bone[bone.name] for field, per_bone in values.items()}})
            if entry.get('auto_collision'):
                rows[-1].update(vertex_map=VERTEX_MAP, collision_radius=thickness)
        defaults = {
            'joint_name': '', 'joint_parent': '', 'simulate': True, 'allow_rotation': True,
            'lock_translation': False, 'stretch_spring': 1.0, 'bend_spring': 0.8,
            'torsion_spring': 0.2, 'goal_strength': 0.5, 'goal_damping': 0.5,
            'gravity_z': 0.3, 'drag': 0.2, 'mass': 1.0, 'world_collision': False,
            'end_effector': 0.0, 'extrude_sides': 2, 'extrude_radius': 1.0,
            'extrude_forward_axis': 'y', 'twist_relax': 1.0, 'extra_iterations': 2,
            'collision_radius': 0.0, 'antishrink': 1.0,
        }
        if entry.get('use_motion_limits'):
            defaults.update(stray_radius=0.0, stray_radius_stretchiness=0.0)
        if entry.get('auto_collision'):
            defaults['vertex_map'] = ''
        # Original Blender bones run along +Y. Extruded bases with twist_relax
        # avoid the legacy rope solver's +X-only alignment requirement.
        attrs = {key: {'display': key, 'default': value, 'show': True, 'ui_order': i}
                 for i, (key, value) in enumerate(defaults.items())}
        # ModelDoc treats ':' in a node name as a path qualifier and silently
        # drops the chain. Export names are already safe, unique identifiers.
        node = {'_class': 'ClothChain', 'name': 'avatar_physics_' + names[entry['root']],
                'root_bone': names[entry['root']],
                'chain': {'version': 2, 'joints': rows, 'attrs': attrs, 'selection': []}}
        if entry.get('lock_twist') and any(row['allow_rotation'] for row in rows):
            from .physics_swing import export_swing
            node['_swing_only'] = export_swing(rows, lengths, parents, UNIT)
        result.append(node)
    return result


def append_physics_modeldoc(document, nodes, colliders=()):
    if not nodes:
        return document
    from .core import class_spans
    def kv(value):
        if isinstance(value, dict):
            return '{ ' + ' '.join(key + ' = ' + kv(child) for key, child in value.items()) + ' }'
        if isinstance(value, list):
            return '[ ' + ', '.join(kv(child) for child in value) + ' ]'
        return json.dumps(value, allow_nan=False)
    def append_list(document, node_class, children, defaults=()):
        # Extend singleton lists so native hero physics and jiggles survive.
        spans = [(a, b) for a, b, cls in class_spans(document) if cls == node_class]
        if len(spans) > 1:
            raise ValueError(f'Multiple native {node_class} nodes require manual ModelDoc preparation')
        import re
        if spans:
            start, end = spans[0]
            match = re.search(r'\bchildren\s*=\s*\[', document[start:end])
            if match is None:
                raise ValueError(f'Native {node_class} has no children list')
            offset = start + match.end()
            addition = ', '.join(kv(node) for node in children) + ', '
        else:
            root = re.search(r'\brootNode\s*=\s*\{', document)
            if root is None:
                raise ValueError('ModelDoc root node is missing')
            match = re.search(r'\bchildren\s*=\s*\[', document[root.end():])
            if match is None:
                raise ValueError('ModelDoc root children are missing')
            offset = root.end() + match.end()
            addition = kv({'_class': node_class, 'children': list(defaults) + children}) + ', '
        return document[:offset] + addition + document[offset:]

    cloth, jiggles = list(colliders), []
    for node in nodes:
        if '_swing_only' in node:
            cloth.extend(node['_swing_only']['cloth'])
            jiggles.extend(node['_swing_only']['jigglebones'])
        else:
            cloth.append(node)
    params = {'_class': 'ClothParams', 'new_style': True, 'unitless_damping': True,
              'default_gravity_scale': 1.0, 'default_vel_air_drag': 0.1, 'extra_iterations': 2}
    document = append_list(document, 'Softbody', cloth, [params])
    if jiggles:
        document = append_list(document, 'JiggleBoneList', jiggles)
    return document


def validate_compiled_limits(fe, rows, unit):
    """Require authored bounds on simulated joints and their support points."""
    names = [str(name) for name in fe['m_CtrlName']]
    static = int(fe['m_nStaticNodes'])
    stretch_scale = math.exp(-float(fe.get('m_flDefaultThreadStretch', 0.0)))
    records = {}
    for record in fe.get('m_AnimStrayRadii', []):
        left, right = (int(value) for value in record['nNode'])
        if left == right:
            records[left] = (float(record['flMaxDist']), float(record['flRelaxationFactor']))
    checked = []
    for row in rows:
        # Allowing an anchor to rotate does not simulate its position. The
        # compiler keeps its controls static and omits redundant distance limits.
        if 'stray_radius' not in row or not row['simulate']:
            continue
        name = row['joint_name']
        radius = row['stray_radius']
        strength = 1.0 - row.get('stray_radius_stretchiness', 0.0)
        actual = [records[index] for index, control in enumerate(names) if index >= static and index in records
                  and (control == name or control.startswith('$cc' + name + '_'))]
        if radius <= 0 or strength <= 0:
            if actual:
                raise ValueError('Disabled motion limit still compiled: ' + name)
            continue
        if not actual or any(not math.isclose(distance, radius, rel_tol=1e-4, abs_tol=1e-5)
                             or not math.isclose(relaxation, strength * stretch_scale, rel_tol=1e-4, abs_tol=1e-5)
                             for distance, relaxation in actual):
            raise ValueError('Motion limit did not compile as requested: ' + name)
        checked.append({'bone': name, 'radius_meters': radius * unit, 'strength': strength})
    return checked


def validate_compiled_physics(compiled, path, nodes, colliders=()):
    """Require real compiled dynamic nodes and correct skeleton bindings."""
    if not nodes:
        return {'enabled': False, 'chains': []}
    from . import read_resource, bone_matrices, UNIT
    from mathutils import Quaternion, Vector
    resource = read_resource(compiled, path, model=True)
    phys = resource.get_block(None, block_name='PHYS')
    fe = phys.get('m_pFeModel') if phys else None
    if not fe or (int(fe['m_nNodeCount']) <= int(fe['m_nStaticNodes']) and not fe.get('m_JiggleBones')):
        raise ValueError('Compiler did not produce dynamic physics nodes')
    names = [str(n) for n in fe['m_CtrlName']]
    static = int(fe['m_nStaticNodes'])
    skeleton = {bone.name for bone in resource.get_bones()}
    rest = bone_matrices(resource.get_bones())
    maximum_rest_error = 0.0
    checked = []
    def check_rest(name):
        nonlocal maximum_rest_error
        pose = fe['m_InitPose'][names.index(name)]
        position_error = (Vector(pose[:3]) * UNIT - rest[name].translation).length
        rotation = Quaternion((float(pose[7]), float(pose[4]), float(pose[5]), float(pose[6])))
        if position_error > 0.0002 or abs(rotation.dot(rest[name].to_quaternion())) < 0.99999:
            raise ValueError(f'Physics rest pose differs from the original bone: {name}')
        maximum_rest_error = max(maximum_rest_error, position_error)
    for node in nodes:
        if '_swing_only' in node:
            from .physics_swing import validate_swing
            checked.append(validate_swing(fe, node, skeleton, check_rest))
            continue
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
            check_rest(name)
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
        if any('stray_radius' in row for row in rows):
            checked[-1]['motion_limits'] = validate_compiled_limits(fe, rows, UNIT)
    collision = 'No body or world collision; accessory secondary motion only'
    if colliders:
        from .physics_collision import validate_colliders
        collision = validate_colliders(fe, colliders, [row for node in nodes for row in node['chain']['joints']], UNIT)
    return {'enabled': True, 'chains': checked, 'compiled_nodes': int(fe['m_nNodeCount']),
            'static_nodes': static, 'runtime_simulation_verified': False,
            'max_physics_rest_position_error_meters': maximum_rest_error,
            'collision': collision}
