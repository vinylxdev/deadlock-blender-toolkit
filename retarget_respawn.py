"""Bake the idle pose for countdown entities that do not run graph retargeting."""
import json
import re

from .core import class_spans, resource_path


COUNTDOWN_ANIMATIONS = frozenset(('respawn_countdown_idle', 'rebirth_countdown_idle'))


def respawn_modeldoc(template, animation_path):
    """Replace only countdown animation sources, retaining their sequence settings."""
    edits = []
    names = []
    for start, end, cls in class_spans(template):
        if cls != 'AnimFile':
            continue
        fragment = template[start:end]
        name = re.search(r'\bname\s*=\s*"([^"]+)"', fragment)
        if not name or name.group(1) not in COUNTDOWN_ANIMATIONS:
            continue
        source = re.search(r'\bsource_filename\s*=\s*"[^"]*"', fragment)
        if source is None:
            raise ValueError('Countdown animation has no source: ' + name.group(1))
        edits.append((start + source.start(), start + source.end(),
                      'source_filename = ' + json.dumps(resource_path(animation_path))))
        names.append(name.group(1))
    for start, end, value in sorted(edits, reverse=True):
        template = template[:start] + value + template[end:]
    return template, names


def write_pose_animation(path, skeleton, pose):
    """Write two identical frames, with explicit local channels for every bone."""
    from SourceIO.library.utils import datamodel as dm
    from . import UNIT
    model = dm.DataModel('model', 22)
    root = model.add_element('root')
    skeleton_node = model.add_element('', 'DmeModel')
    root['skeleton'] = skeleton_node
    transform = model.add_element('', 'DmeTransform')
    transform['position'] = dm.Vector3((0, 0, 0))
    transform['orientation'] = dm.Quaternion((0, 0, 0, 1))
    skeleton_node['transform'] = transform
    axis = model.add_element('', 'DmeAxisSystem')
    axis['upAxis'], axis['forwardParity'], axis['coordSys'] = 3, 1, 0
    skeleton_node['axisSystem'] = axis
    joints = {name: model.add_element(name, 'DmeJoint') for name in skeleton}
    skeleton_node['jointList'] = dm.make_array(list(joints.values()), dm.Element)
    skeleton_node['children'] = dm.make_array([joints[n] for n, b in skeleton.items() if not b['parent']], dm.Element)
    animation_list = model.add_element('', 'DmeAnimationList')
    root['animationList'] = animation_list
    clip = model.add_element('deadlock_toolkit_respawn_idle', 'DmeChannelsClip')
    animation_list['animations'] = dm.make_array([clip], dm.Element)
    clip['frameRate'] = 30.0
    timeframe = model.add_element('', 'DmeTimeFrame')
    timeframe['start'] = timeframe['offset'] = dm.Time(0)
    timeframe['duration'] = dm.Time(1 / 30)
    timeframe['scale'] = 1.0
    clip['timeFrame'] = timeframe
    channels = []
    for name, bone in skeleton.items():
        parent = bone['parent']
        local = pose[parent].inverted() @ pose[name] if parent else pose[name]
        rest = skeleton[parent]['matrix'].inverted() @ bone['matrix'] if parent else bone['matrix']
        transform = model.add_element(name, 'DmeTransform')
        joints[name]['transform'] = transform
        joints[name]['children'] = dm.make_array([joints[n] for n, b in skeleton.items() if b['parent'] == name], dm.Element)
        for attribute, kind, value_type in (('position', 'Vector3', dm.Vector3), ('orientation', 'Quaternion', dm.Quaternion)):
            def value(matrix):
                if attribute == 'position':
                    return value_type(matrix.translation / UNIT)
                q = matrix.to_quaternion()
                return value_type((q.x, q.y, q.z, q.w))
            transform[attribute] = value(rest)
            channel = model.add_element(name + '_' + attribute, 'DmeChannel')
            channel['toElement'], channel['toAttribute'], channel['mode'] = transform, attribute, 3
            log = model.add_element(name, 'Dme' + kind + 'Log')
            layer = model.add_element(name, 'Dme' + kind + 'LogLayer')
            layer['times'] = dm.make_array([dm.Time(0), dm.Time(1 / 30)], dm.Time)
            layer['values'] = dm.make_array([value(local), value(local)], value_type)
            log['layers'] = dm.make_array([layer], dm.Element)
            log['layerCount'] = 1
            channel['log'] = log
            channels.append(channel)
    clip['channels'] = dm.make_array(channels, dm.Element)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.write(str(path), 'binary', 9)


def bake_respawn_idle(settings, skeleton, constraints, path):
    from . import reference_bones
    from .preview import idle_source
    from .retarget_export import runtime_root_reference
    from .retarget_validation import native_idle, evaluate_rules
    native_bones = reference_bones(settings)
    reference = {bone.name: skeleton[bone.name]['matrix'] for bone in native_bones}
    label, source = idle_source(settings)
    native = native_idle(reference, native_bones, source, time_seconds=0.0)
    roots = {bone['parent'] for name, bone in skeleton.items()
             if name not in reference and bone['parent'] in reference}
    model_roots = {name: native[name].copy() for name in roots}
    for name in roots:
        native[name] = native[name] @ reference[name].inverted() @ runtime_root_reference(settings, reference, name)
    rest = {name: bone['matrix'] for name, bone in skeleton.items()}
    pose = evaluate_rules(rest, native, constraints, skeleton)
    # Legacy animation retains the model's root axes. Child locals are derived
    # from the finished world pose, including the graph-root retarget correction.
    pose.update(model_roots)
    write_pose_animation(path, skeleton, pose)
    return {'idle_animation': label, 'idle_frame': 0, 'source': str(path), 'bones': len(pose)}, pose


def validate_respawn_idle(compiled, resource_path, names, expected):
    """Check the actual compiled channels without applying any runtime rules."""
    from mathutils import Matrix, Quaternion, Vector
    from . import read_resource, UNIT
    from SourceIO.library.source2.utils.decode_animations import parse_anim
    resource = read_resource(compiled, resource_path, model=True)
    bones = resource.get_bones()
    groups = resource.get_blocks(None, 'AGRP')
    blocks = resource.get_blocks(None, 'ANIM')
    results = {}
    for name in names:
        matches = [(block, anim) for block in blocks for anim in block['m_animArray'] if anim['m_name'] == name]
        if len(matches) != 1 or len(groups) != 1:
            raise ValueError('Compiled countdown animation missing or ambiguous: ' + name)
        block, anim = matches[0]
        decoded = parse_anim(anim, groups[0]['m_decodeKey'], block['m_decoderArray'], block['m_segmentArray'])
        position_error = rotation_error = 0.0
        for frame in decoded.frames:
            pose = {}
            for bone in bones:
                channels = frame.bone_data.get(bone.name, {})
                position = channels.get('Position', (None, bone.pos))[1]
                angle = channels.get('Angle')
                rotation = Quaternion((angle[1][3], *angle[1][:3])) if angle else Quaternion(bone.rot)
                local = Matrix.LocRotScale(Vector(position) * UNIT, rotation, Vector((1, 1, 1)))
                pose[bone.name] = pose[bone.parent] @ local if bone.parent else local
                target = expected[bone.name]
                position_error = max(position_error, (pose[bone.name].translation - target.translation).length)
                rotation_error = max(rotation_error, max(abs(pose[bone.name][i][j] - target[i][j])
                                                        for i in range(3) for j in range(3)))
        if not decoded.frames or position_error > 0.003 or rotation_error > 0.003:
            raise ValueError(f'Compiled countdown pose differs from idle frame 0: {name} '
                             f'(position {position_error:.6f} m, rotation {rotation_error:.6f})')
        results[name] = {'frames': len(decoded.frames), 'bones': len(bones),
                         'max_position_error_meters': position_error, 'max_rotation_matrix_error': rotation_error}
    return results
