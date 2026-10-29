"""Uniform sizing of a retarget preview, shared by the sidebar and height fit."""
import json
import math
import re

import bpy
from mathutils import Matrix

SCALE_KEY = 'deadlock_avatar_scale'
CONTACT_KEYS = ('deadlock_retarget_contact_offsets', 'deadlock_retarget_contact_vertices',
                'deadlock_retarget_contact_version', 'deadlock_retarget_standing_soles')


def avatar_scale(rig):
    return rig.get(SCALE_KEY, 1.0)


def avatar_meshes(rig):
    return {obj for collection in rig.users_collection for obj in collection.all_objects
            if obj.type == 'MESH' and obj.find_armature() == rig}


def scale_avatar(rig, value):
    """Set an absolute multiplier; keep mesh data and the saved pre-idle pose intact."""
    from .physics_preview import stop
    from .preview import RESTORE_KEY, idle_source
    from .retarget import contact_offsets, retarget_idle_matrices

    if not math.isfinite(value) or value <= 0:
        raise ValueError('Overall size must be a positive finite value')
    if bpy.context.window_manager.get('deadlock_build_running', False):
        raise ValueError('Wait for the active build before resizing the avatar')
    if not rig.get('deadlock_retarget_mapping'):
        raise ValueError('Create Retarget Original Rig Preview first')
    previous = avatar_scale(rig)
    if abs(value - previous) < 1e-7:
        return
    reference = {name: Matrix(rows) for name, rows in json.loads(rig['deadlock_retarget_reference']).items()}
    anchor = (reference['ankle_L'].translation + reference['ankle_R'].translation) * 0.5
    fit = Matrix.Translation(anchor) @ Matrix.Scale(value / previous, 4) @ Matrix.Translation(-anchor)
    stop()
    had_scale = SCALE_KEY in rig
    worlds = {obj: obj.matrix_world.copy() for obj in avatar_meshes(rig)}
    rig_world = rig.matrix_world.copy()
    cached = {key: rig[key] for key in CONTACT_KEYS if key in rig}
    poses = {bone.name: bone.matrix_basis.copy() for bone in rig.pose.bones}
    try:
        rig.matrix_world = fit @ rig_world
        bpy.context.view_layer.update()
        for obj, world in worlds.items():
            obj.matrix_world = fit @ world
        for key in CONTACT_KEYS:
            if key in rig:
                del rig[key]
        bpy.context.view_layer.update()
        contact_offsets(rig, reference)
        if RESTORE_KEY in rig:
            _, path = idle_source(bpy.context.scene.deadlock_export)
            for name, matrix in retarget_idle_matrices(rig, path).items():
                rig.pose.bones[name].matrix_basis = matrix
        rig[SCALE_KEY] = value
        bpy.context.view_layer.update()
    except Exception:
        if had_scale:
            rig[SCALE_KEY] = previous
        elif SCALE_KEY in rig:
            del rig[SCALE_KEY]
        rig.matrix_world = rig_world
        bpy.context.view_layer.update()
        for obj, world in worlds.items():
            obj.matrix_world = world
        for key in CONTACT_KEYS:
            if key in rig:
                del rig[key]
        for key, data in cached.items():
            rig[key] = data
        for name, matrix in poses.items():
            rig.pose.bones[name].matrix_basis = matrix
        bpy.context.view_layer.update()
        raise


def set_avatar_scale(rig, value):
    # RNA setters cannot report operator errors. Leave the old value and
    # transforms intact and show the failure in the sidebar's status area.
    settings = bpy.context.scene.deadlock_export
    try:
        scale_avatar(rig, value)
        settings.status = 'Avatar resized; Build Retargeted VPK to export the new size.'
    except Exception as error:
        settings.status = str(error)


def avatar_bounds(rig):
    """World Z bounds of the exportable rest mesh, including keys and modifiers."""
    previous = rig.data.pose_position
    lower, upper = math.inf, -math.inf
    try:
        rig.data.pose_position = 'REST'
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        for obj in avatar_meshes(rig):
            if obj.hide_render:
                continue
            evaluated = obj.evaluated_get(depsgraph)
            mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
            try:
                for vertex in mesh.vertices:
                    z = (evaluated.matrix_world @ vertex.co).z
                    lower, upper = min(lower, z), max(upper, z)
            finally:
                evaluated.to_mesh_clear()
    finally:
        rig.data.pose_position = previous
        bpy.context.view_layer.update()
    height = upper - lower
    if not math.isfinite(height) or height <= 1e-6:
        raise ValueError('The avatar has no measurable render-enabled mesh height')
    return lower, upper


def avatar_height(rig):
    lower, upper = avatar_bounds(rig)
    return upper - lower


def eye_height(heights, ground, subject):
    """Average the available eye centers; ignore eye-end and other facial bones."""
    eyes = []
    for side in ('L', 'R'):
        name = next((name for name in (f'eye_0_{side}', f'eye_{side}') if name in heights), None)
        if name:
            eyes.append(heights[name])
    if not eyes:
        raise ValueError(f'{subject} has no mapped eye bones; adjust Overall size manually')
    height = sum(eyes) / len(eyes) - ground
    if not math.isfinite(height) or height <= 1e-6:
        raise ValueError(f'{subject} has no measurable ground-to-eye height')
    return height


def avatar_eye_height(rig):
    """Rest eye-bone height above the evaluated render mesh's lowest point."""
    mapping = json.loads(rig['deadlock_retarget_mapping'])
    heights = {role: (rig.matrix_world @ rig.data.bones[name].head_local).z
               for name, role in mapping.items() if name in rig.data.bones}
    ground, _ = avatar_bounds(rig)
    return eye_height(heights, ground, 'The avatar')


def hero_height(settings):
    """Rest eye-bone height above default LOD0 body bounds, omitting weapon props."""
    from . import UNIT, archive, read_resource, selected_hero, bone_matrices
    _, _, path = selected_hero(settings)
    retail = archive(settings)
    resource = read_resource(retail.read(path), path, model=True)
    data = resource.get_block(None, block_name='DATA')
    control = resource.get_block(None, block_name='CTRL')
    embedded = {int(row['m_nMeshIndex']): row for row in control.get('embedded_meshes', [])}
    refs = data.get('m_refMeshes', [])
    indices = range(len(refs)) if len(refs) else embedded.keys()
    bounds = []
    for index in indices:
        if not int(data['m_refLODGroupMasks'][index]) & 1:
            continue
        if not int(data['m_refMeshGroupMasks'][index]) & int(data['m_nDefaultMeshGroupMask']):
            continue
        info = embedded.get(index)
        name = str(info['m_Name'] if info else refs[index])
        if set(re.split(r'[^a-z0-9]+', name.lower())) & {'weapon', 'gun', 'book', 'pillow', 'balloon', 'hammer'}:
            continue
        if info:
            mesh = resource.get_block(None, block_id=int(info['m_nDataBlock']))
        else:
            mesh_path = str(refs[index])
            if not mesh_path.endswith('_c'):
                mesh_path += '_c'
            mesh = read_resource(retail.read(mesh_path), mesh_path).get_block(None, block_name='DATA')
        for obj in mesh['m_sceneObjects']:
            bounds.append((float(obj['m_vMinBounds'][2]), float(obj['m_vMaxBounds'][2])))
    if not bounds or any(not math.isfinite(z) for pair in bounds for z in pair):
        raise ValueError('The selected hero has no measurable body render bounds')
    reference = bone_matrices(resource.get_bones())
    return eye_height({name: matrix.translation.z for name, matrix in reference.items()},
                      min(lo for lo, hi in bounds) * UNIT, 'The selected hero')


def fit_hero_height(rig, settings):
    height = hero_height(settings)
    value = avatar_scale(rig) * height / avatar_eye_height(rig)
    scale_avatar(rig, value)
    return height
