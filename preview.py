"""Reversible native idle-pose previews on converted avatar rigs."""

import json
import re
from pathlib import Path

from mathutils import Matrix, Quaternion, Vector

from .core import class_spans, resource_path

RESTORE_KEY = 'deadlock_idle_restore'


def preview_rig(context, settings):
    collections = [collection for collection in context.scene.collection.children_recursive
                   if collection.get('deadlock_converted') and collection.get('deadlock_hero') == settings.hero]
    if context.object:
        selected = [collection for collection in collections if context.object in collection.all_objects[:]]
        if selected:
            collections = selected
    for collection in reversed(collections):
        rig = next((obj for obj in collection.all_objects if obj.type == 'ARMATURE'), None)
        if rig and rig.name in context.view_layer.objects:
            return rig
    return None


def animation_sources(settings):
    from . import local_path, selected_hero, archive
    from .pipeline import extract_hero
    key, _, replacement = selected_hero(settings)
    content = local_path(settings.sdk_root) / 'content/citadel_addons' / f'deadlock_toolkit_{key}'
    native = content / replacement.removesuffix('_c')
    if not native.is_file():
        output = local_path(settings.output_dir) / key
        native = extract_hero(local_path(settings.viewer_cli), archive(settings).read(replacement), replacement,
                              local_path(settings.game_dir), content, output, lambda _: None)
    template = native.read_text(encoding='utf-8-sig')
    sources = {}
    for start, end, cls in class_spans(template):
        if cls != 'AnimFile':
            continue
        fragment = template[start:end]
        name = re.search(r'\bname\s*=\s*"([^"]+)"', fragment)
        source = re.search(r'\bsource_filename\s*=\s*"([^"]+)"', fragment)
        if not name or not source or re.search(r'\b(?:delta|worldSpace)\s*=\s*true', fragment):
            continue
        path = content / resource_path(source.group(1))
        if path.is_file():
            sources[name.group(1).lower()] = path
    return sources


def idle_source(settings):
    candidates = []
    priority = ('primary_idle', 'idle', 'primary_stand_idle', 'idle_loadout', 'stand_idle', 'weapon_stand_idle',
                'weapon_idle', 'ui_hero_select', 'ui_main_menu')
    for label, path in animation_sources(settings).items():
        if label not in priority and ('idle' not in label or any(term in label for term in ('aim', 'shoot', 'cast', 'crouch', 'reload'))):
            continue
        candidates.append((priority.index(label) if label in priority else len(priority), label, path))
    if not candidates:
        raise ValueError('No usable native idle animation was found for this hero')
    _, label, path = min(candidates)
    return label, path


def idle_matrices(rig, path, time_seconds=0.0, model=None):
    from SourceIO.library.utils import datamodel
    from . import UNIT
    if model is None:
        model = datamodel.load(str(path))
    channels = {}
    for element in model.elements:
        if element.type != 'DmeChannel' or element.get('toElement') is None:
            continue
        attribute = element.get('toAttribute')
        if attribute not in ('position', 'orientation'):
            continue
        layers = element['log']['layers']
        if not layers or not layers[0]['values']:
            continue
        layer = layers[0]
        index = min(range(len(layer['times'])), key=lambda i: abs(float(layer['times'][i])-time_seconds))
        channels.setdefault(element['toElement'].name, {})[attribute] = layer['values'][index]
    if not {'pelvis', 'head', 'hand_L', 'hand_R'}.issubset(channels):
        raise ValueError('The native idle animation does not contain a full humanoid pose')
    matrices = {}
    for bone in rig.data.bones:
        local = bone.parent.matrix_local.inverted() @ bone.matrix_local if bone.parent else bone.matrix_local.copy()
        values = channels.get(bone.name, {})
        position = Vector(values['position']) * UNIT if 'position' in values else local.translation
        q = values.get('orientation')
        rotation = Quaternion((q[3], q[0], q[1], q[2])) if q is not None else local.to_quaternion()
        matrices[bone.name] = local.inverted() @ Matrix.LocRotScale(position, rotation, Vector((1, 1, 1)))
    return matrices


def restore_pose(rig):
    saved = json.loads(rig[RESTORE_KEY])
    for name, rows in saved['bones'].items():
        if name in rig.pose.bones:
            rig.pose.bones[name].matrix_basis = Matrix(rows)
    rig.data.pose_position = saved['pose_position']
    del rig[RESTORE_KEY]


def toggle_idle(context, settings):
    from .physics_preview import stop
    stop()
    rig = preview_rig(context, settings)
    if rig is None:
        raise ValueError('Create Retarget Original Rig Preview first, then select its rig or mesh')
    if RESTORE_KEY in rig:
        restore_pose(rig)
        context.view_layer.update()
        return f'Restored previous pose on {rig.name}'
    label, path = idle_source(settings)
    if rig.get('deadlock_retarget_mapping'):
        from .retarget import retarget_idle_matrices
        matrices = retarget_idle_matrices(rig, path)
    else:
        matrices = idle_matrices(rig, path)
    saved = {'pose_position': rig.data.pose_position,
             'bones': {bone.name: [list(row) for row in bone.matrix_basis] for bone in rig.pose.bones}}
    rig[RESTORE_KEY] = json.dumps(saved)
    try:
        for name, matrix in matrices.items():
            rig.pose.bones[name].matrix_basis = matrix
        rig.data.pose_position = 'POSE'
        context.view_layer.update()
    except Exception:
        restore_pose(rig)
        context.view_layer.update()
        raise
    return f'Previewing {label} on {rig.name}; toggle off to restore the previous pose'
