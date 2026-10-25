"""Portable physics curves, with Blender's curve editor and interpolation."""
import json
import math

EDITOR_KEY = 'deadlock_physics_curve_editor'
SETTINGS = {
    'stiffness': ('goal_strength', 0.5),
    'damping': ('goal_damping', 0.5),
    'gravity': ('gravity_z', 0.3),
    'twist_resistance': ('torsion_spring', 0.2),
}
CURVE_SETTINGS = (*SETTINGS, 'motion_limit', 'motion_limit_strength')


def default_curve():
    return {'enabled': False, 'extend': 'HORIZONTAL',
            'points': [[0.0, 1.0, 'AUTO'], [1.0, 1.0, 'AUTO']]}


def validated_curve(value):
    if not isinstance(value, dict) or type(value.get('enabled')) is not bool:
        raise ValueError('Invalid physics curve')
    extend = value.get('extend', 'HORIZONTAL')
    if extend not in ('HORIZONTAL', 'EXTRAPOLATED'):
        raise ValueError('Invalid physics curve extension')
    points = value.get('points')
    if not isinstance(points, list) or len(points) < 2:
        raise ValueError('Physics curves need at least two points')
    result, previous = [], -1.0
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 3:
            raise ValueError('Invalid physics curve point')
        x, y, handle = point
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in (x, y)):
            raise ValueError('Physics curve coordinates must be between 0 and 1')
        if x <= previous or handle not in ('AUTO', 'AUTO_CLAMPED', 'VECTOR'):
            raise ValueError('Physics curve points need increasing positions and valid handles')
        result.append([x, y, handle])
        previous = x
    return {'enabled': value['enabled'], 'extend': extend, 'points': result}


def new_editor(definition=None):
    import bpy
    definition = validated_curve(default_curve() if definition is None else definition)
    tree = bpy.data.node_groups.new('.Deadlock Physics Curve', 'ShaderNodeTree')
    tree[EDITOR_KEY] = True
    try:
        node = tree.nodes.new('ShaderNodeFloatCurve')
        mapping = node.mapping
        mapping.initialize()
        mapping.extend = definition['extend']
        mapping.use_clip = True
        mapping.clip_min_x = mapping.clip_min_y = 0.0
        mapping.clip_max_x = mapping.clip_max_y = 1.0
        curve = mapping.curves[0]
        points = definition['points']
        curve.points[0].location = points[0][:2]
        curve.points[-1].location = points[-1][:2]
        for x, y, _ in points[1:-1]:
            curve.points.new(x, y)
        mapping.update()
        for point, (_, _, handle) in zip(curve.points, points):
            point.handle_type = handle
        mapping.update()
        return tree
    except Exception:
        bpy.data.node_groups.remove(tree)
        raise


def editor(name):
    import bpy
    tree = bpy.data.node_groups.get(name)
    return tree if tree and tree.get(EDITOR_KEY) else None


def read_editor(tree, enabled):
    mapping = tree.nodes[0].mapping
    points = mapping.curves[0].points
    return validated_curve({'enabled': enabled, 'extend': mapping.extend,
                            'points': [[float(p.location.x), float(p.location.y), p.handle_type] for p in points]})


def remove_editor(name):
    import bpy
    tree = editor(name)
    if tree:
        bpy.data.node_groups.remove(tree)


def clear_editors(operator):
    for key in CURVE_SETTINGS:
        name = key + '_curve_name'
        remove_editor(getattr(operator, name, ''))
        setattr(operator, name, '')


def prepare_editors(operator, entry):
    """Open independent editors without retaining state from another chain."""
    clear_editors(operator)
    try:
        for key in CURVE_SETTINGS:
            definition = entry.get(key + '_curve')
            setattr(operator, 'use_' + key + '_curve', bool(definition and definition['enabled']))
            setattr(operator, key + '_curve_data', json.dumps(definition) if definition else '')
            setattr(operator, key + '_curve_name', new_editor(definition).name)
    except Exception:
        clear_editors(operator)
        raise


def store_curves(operator, entry):
    """Retain disabled shapes and redo data after temporary editors close."""
    for key in CURVE_SETTINGS:
        enabled = getattr(operator, 'use_' + key + '_curve')
        data = getattr(operator, key + '_curve_data')
        if not enabled and not data:
            continue
        tree = editor(getattr(operator, key + '_curve_name'))
        definition = (read_editor(tree, enabled) if tree else
                      json.loads(data) if data else default_curve())
        definition['enabled'] = enabled
        entry[key + '_curve'] = definition
        setattr(operator, key + '_curve_data', json.dumps(definition))


def multipliers(definition, positions):
    """Use the editor's actual interpolation, without retaining export datablocks."""
    tree = new_editor(definition)
    try:
        mapping = tree.nodes[0].mapping
        return [max(0.0, min(1.0, mapping.evaluate(mapping.curves[0], position))) for position in positions]
    finally:
        remove_editor(tree.name)


def chain_positions(members, matrix):
    """Rest distance from the root, normalized to the furthest tip in each branch.

    Sample at bone heads: the selected root is 0 and every descendant leaf is
    1. Shared sections use their longest remaining branch. Object scale is
    included, and detached bone heads contribute their actual separation.
    """
    bones = {bone.name: bone for bone in members}
    heads = {name: matrix @ bone.head_local for name, bone in bones.items()}
    distances, remaining = {}, {}

    def distance(bone):
        if bone.name not in distances:
            parent = bone.parent
            distances[bone.name] = (distance(parent) + (heads[bone.name] - heads[parent.name]).length
                                    if parent and parent.name in bones else 0.0)
        return distances[bone.name]

    def rest(bone):
        if bone.name not in remaining:
            remaining[bone.name] = max(((heads[child.name] - heads[bone.name]).length + rest(child)
                                        for child in bone.children if child.name in bones), default=0.0)
        return remaining[bone.name]

    return {bone.name: distance(bone) / (distance(bone) + rest(bone))
            if distance(bone) + rest(bone) > 1e-12 else 0.0 for bone in members}
