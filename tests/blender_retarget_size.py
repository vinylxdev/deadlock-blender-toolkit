"""Uniform size, eye-height fitting, grounding and runtime export agreement.

blender -b --factory-startup --python-exit-code 1 --python tests/blender_retarget_size.py
No game, SDK or SourceIO installation required.
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import bpy
from mathutils import Matrix, Vector

folder = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('size_test_addon', folder / '__init__.py')
addon = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = addon
spec.loader.exec_module(addon)
from size_test_addon import preview, retarget, retarget_export, retarget_size as size
from size_test_addon.retarget_placement import preserve_proportions
from size_test_addon.retarget_validation import evaluate_rules


def close(a, b, tolerance=1e-5):
    return max(abs(a[i][j] - b[i][j]) for i in range(4) for j in range(4)) < tolerance


addon.register()
settings = bpy.context.scene.deadlock_export
source_collection = bpy.data.collections.new('Source avatar')
bpy.context.scene.collection.children.link(source_collection)
settings.collection = source_collection
data = bpy.data.armatures.new('Source bones')
source = bpy.data.objects.new('Source rig', data)
source_collection.objects.link(source)
bpy.context.view_layer.objects.active = source
source.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
rows = [('root_motion', None, (0, 0, 0), (0, 0, 0.2)),
        ('pelvis', 'root_motion', (0, 0, 1), (0, 0, 1.2)),
        ('head', 'pelvis', (0, 0, 1.8), (0, 0, 2)),
        ('Eye_L', 'head', (0.05, -0.05, 1.9), (0.05, -0.1, 1.9)),
        ('Eye_R', 'head', (-0.05, -0.05, 1.94), (-0.05, -0.1, 1.94))]
for side, sign in (('L', 1), ('R', -1)):
    rows.extend([
        (f'arm_upper_{side}', 'pelvis', (sign * 0.2, 0, 1.5), (sign * 0.5, 0, 1.5)),
        (f'arm_lower_{side}', f'arm_upper_{side}', (sign * 0.5, 0, 1.5), (sign * 0.8, 0, 1.5)),
        (f'hand_{side}', f'arm_lower_{side}', (sign * 0.8, 0, 1.5), (sign * 0.8, -0.1, 1.5)),
        (f'leg_upper_{side}', 'pelvis', (sign * 0.1, 0, 1), (sign * 0.1, 0, 0.5)),
        (f'leg_lower_{side}', f'leg_upper_{side}', (sign * 0.1, 0, 0.5), (sign * 0.1, 0, 0.1)),
        (f'ankle_{side}', f'leg_lower_{side}', (sign * 0.1, 0, 0.1), (sign * 0.1, -0.1, 0.05)),
        (f'ball_{side}', f'ankle_{side}', (sign * 0.1, -0.1, 0.05), (sign * 0.1, -0.2, 0.05)),
        (f'ball_end_{side}', f'ball_{side}', (sign * 0.1, -0.2, 0.05), (sign * 0.1, -0.3, 0.05))])
for name, parent, head, tail in rows:
    bone = data.edit_bones.new(name)
    bone.head, bone.tail = head, tail
    if parent:
        bone.parent = data.edit_bones[parent]
bpy.ops.object.mode_set(mode='OBJECT')
source.matrix_world = Matrix.Translation((3, 4, 1)) @ Matrix.Rotation(0.3, 4, 'Z') @ Matrix.Scale(1.2, 4)
reference = {}
for name, parent, head, tail in rows:
    direction = (Vector(tail) - Vector(head)).normalized()
    reference[name.replace('Eye_', 'eye_0_')] = Matrix.Translation(Vector(head) * 1.1) @ Vector((1, 0, 0)).rotation_difference(direction).to_matrix().to_4x4()
native_bones = [SimpleNamespace(name=name.replace('Eye_', 'eye_0_'), parent=parent) for name, parent, _, _ in rows]
points = [(0, 0, 2)]
for sign in (1, -1):
    points.extend((sign * 0.1 + i * 0.003, -j * 0.025, 0.02) for i in range(4) for j in range(4))
mesh = bpy.data.meshes.new('Source mesh')
mesh.from_pydata(points, [], [])
body = bpy.data.objects.new('Source body', mesh)
source_collection.objects.link(body)
body.parent = source
body.matrix_world = source.matrix_world @ Matrix.Translation((0.01, -0.02, 0))
for name, indices in [('head', [0]), ('ankle_L', list(range(1, 17))), ('ankle_R', list(range(17, 33)))]:
    body.vertex_groups.new(name=name).add(indices, 1, 'REPLACE')
body.modifiers.new('Armature', 'ARMATURE').object = source
body.shape_key_add(name='Basis')
key = body.shape_key_add(name='Tall ears')
key.data[0].co.z += 0.4
key.value = 0.5
bpy.context.view_layer.update()
source_world, body_world = source.matrix_world.copy(), body.matrix_world.copy()
source_vertices = [v.co.copy() for v in mesh.vertices]


def idle(rig, path):
    return retarget.retarget_pose_matrices(rig, reference, reference, retarget.contact_offsets(rig, reference))


with patch.object(addon, 'reference_bones', return_value=native_bones), \
        patch.object(addon, 'bone_matrices', return_value=reference), \
        patch.object(addon, 'selected_hero', return_value=('NONE', 'Fixture', 'hero.vmdl_c')), \
        patch.object(preview, 'idle_source', return_value=('idle', Path('idle.dmx'))), \
        patch('size_test_addon.retarget_validation.native_idle', return_value=reference), \
        patch.object(retarget, 'retarget_idle_matrices', side_effect=idle):
    collection, report = retarget.create_retarget_preview(settings)
    rig = next(o for o in collection.objects if o.type == 'ARMATURE')
    skin = next(o for o in collection.objects if o.type == 'MESH')
    assert rig.deadlock_avatar_scale == 1
    baseline = size.avatar_height(rig)
    baseline_eyes = size.avatar_eye_height(rig)
    ground, _ = size.avatar_bounds(rig)
    expected_eyes = sum((rig.matrix_world @ rig.data.bones[f'Eye_{side}'].head_local).z
                        for side in ('L', 'R')) / 2 - ground
    assert abs(baseline_eyes - expected_eyes) < 1e-6
    assert abs(baseline_eyes - baseline) > 0.1
    baseline_world, skin_world = rig.matrix_world.copy(), skin.matrix_world.copy()
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    skin.select_set(True)
    bpy.context.view_layer.objects.active = skin
    preview.toggle_idle(bpy.context, settings)
    saved_pose = rig[preview.RESTORE_KEY]
    for value in (1.25, 1.5, 0.75, 1.0):
        rig.deadlock_avatar_scale = value
        assert abs(rig.deadlock_avatar_scale - value) < 1e-6, settings.status
        assert abs(size.avatar_height(rig) - baseline * value) < 1e-5
        assert abs(size.avatar_eye_height(rig) - baseline_eyes * value) < 1e-5
        assert rig.data.pose_position == 'POSE'
        assert rig[preview.RESTORE_KEY] == saved_pose
        assert abs(min(json.loads(rig['deadlock_retarget_standing_soles']).values())) < 1e-6
        assert close(skin.matrix_world.inverted() @ rig.matrix_world, skin_world.inverted() @ baseline_world)
        assert max(rig.matrix_world.to_scale()) - min(rig.matrix_world.to_scale()) < 1e-6
    assert close(rig.matrix_world, baseline_world)
    assert close(skin.matrix_world, skin_world)
    assert close(source.matrix_world, source_world) and close(body.matrix_world, body_world)
    assert all(a.co == b for a, b in zip(mesh.vertices, source_vertices))
    assert skin.data.shape_keys.key_blocks['Tall ears'].value == key.value
    assert [[(g.group, g.weight) for g in v.groups] for v in mesh.vertices] == [
        [(g.group, g.weight) for g in v.groups] for v in skin.data.vertices]
    for bone in source.data.bones:
        assert close(bone.matrix_local, rig.data.bones[bone.name].matrix_local)

    # Height uses evaluated shape keys and excludes render-disabled objects.
    hidden = skin.copy()
    hidden.data = skin.data.copy()
    collection.objects.link(hidden)
    hidden.data.shape_keys.key_blocks['Tall ears'].data[0].co.z = 100
    hidden.hide_render = True
    bpy.context.view_layer.update()
    assert abs(size.avatar_height(rig) - baseline) < 1e-5
    assert abs(size.avatar_eye_height(rig) - baseline_eyes) < 1e-5

    def block(**kwargs):
        if kwargs.get('block_name') == 'DATA':
            return {'m_refMeshes': [], 'm_refLODGroupMasks': [1, 1, 2, 1],
                    'm_refMeshGroupMasks': [1, 1, 1, 2], 'm_nDefaultMeshGroupMask': 1}
        if kwargs.get('block_name') == 'CTRL':
            return {'embedded_meshes': [dict(m_nMeshIndex=i, m_Name=name, m_nDataBlock=i)
                                       for i, name in enumerate(('body', 'primary_weapon_only', 'body_lod1', 'hidden_body'))]}
        index = kwargs['block_id']
        return {'m_sceneObjects': [dict(m_vMinBounds=(0, 0, 0), m_vMaxBounds=(0, 0, 80 if index == 0 else 1000))]}

    resource = SimpleNamespace(get_block=lambda *args, **kwargs: block(**kwargs), get_bones=lambda: native_bones)
    with patch.object(addon, 'archive', return_value=SimpleNamespace(read=lambda path: b'hero')), \
            patch.object(addon, 'read_resource', return_value=resource):
        target = sum(reference[f'eye_0_{side}'].translation.z for side in ('L', 'R')) / 2
        assert abs(size.hero_height(settings) - target) < 1e-6
        assert bpy.ops.deadlock.fit_avatar_height() == {'FINISHED'}, settings.status
        assert 'eye height' in settings.status
        assert abs(size.avatar_eye_height(rig) - target) < 1e-5
        assert abs(rig.deadlock_avatar_scale - target / baseline_eyes) < 1e-5
        first_scale = rig.deadlock_avatar_scale
        # Taller ears/hair affect the mesh top but never the fitted scale.
        tall_ears = skin.data.shape_keys.key_blocks['Tall ears']
        tall_ears.value = 1
        bpy.context.view_layer.update()
        assert size.avatar_height(rig) > baseline * first_scale + 0.1
        assert bpy.ops.deadlock.fit_avatar_height() == {'FINISHED'}
        assert abs(rig.deadlock_avatar_scale - first_scale) < 1e-6
        tall_ears.value = key.value
        bpy.context.view_layer.update()

        # A single available eye is sufficient; eye-end bones are never centers.
        mapping = rig['deadlock_retarget_mapping']
        one_eye = json.loads(mapping)
        del one_eye['Eye_R']
        rig['deadlock_retarget_mapping'] = json.dumps(one_eye)
        ground, _ = size.avatar_bounds(rig)
        assert abs(size.avatar_eye_height(rig) -
                   ((rig.matrix_world @ rig.data.bones['Eye_L'].head_local).z - ground)) < 1e-6
        del one_eye['Eye_L']
        one_eye['head'] = 'eye_end_L'
        rig['deadlock_retarget_mapping'] = json.dumps(one_eye)
        before = rig.matrix_world.copy()
        try:
            size.fit_hero_height(rig, settings)
            raise AssertionError('Missing avatar eyes must reject fitting')
        except ValueError as error:
            assert 'no mapped eye bones' in str(error)
        assert close(rig.matrix_world, before)
        assert rig.deadlock_avatar_scale == first_scale
        assert rig.data.pose_position == 'POSE'
        rig['deadlock_retarget_mapping'] = mapping

        with patch.object(addon, 'bone_matrices', return_value={'eye_end_L': Matrix.Identity(4)}):
            try:
                size.fit_hero_height(rig, settings)
                raise AssertionError('Missing hero eyes must reject fitting')
            except ValueError as error:
                assert 'selected hero has no mapped eye bones' in str(error)
        assert close(rig.matrix_world, before)

    # External mesh ground bounds use Source 2 units; their top is irrelevant.
    model = SimpleNamespace(get_block=lambda *args, **kw: (
        {'m_refMeshes': ['body.vmesh'], 'm_refLODGroupMasks': [1],
         'm_refMeshGroupMasks': [1], 'm_nDefaultMeshGroupMask': 1}
        if kw.get('block_name') == 'DATA' else {'embedded_meshes': []}), get_bones=lambda: native_bones)
    external = SimpleNamespace(get_block=lambda *args, **kw: {
        'm_sceneObjects': [dict(m_vMinBounds=(0, 0, -2), m_vMaxBounds=(0, 0, 1000))]})
    with patch.object(addon, 'archive', return_value=SimpleNamespace(read=lambda path: b'hero')), \
            patch.object(addon, 'read_resource', side_effect=lambda blob, path, **kw: model if path.endswith('.vmdl_c') else external):
        assert abs(size.hero_height(settings) - (target + 2 * addon.UNIT)) < 1e-6

    # Failed calibration rolls back the full placement, cached contacts and value.
    world_before, skin_before = rig.matrix_world.copy(), skin.matrix_world.copy()
    contacts = rig['deadlock_retarget_contact_offsets']
    with patch.object(retarget, 'contact_offsets', side_effect=ValueError('Calibration failed')):
        rig.deadlock_avatar_scale = 2
    assert 'Calibration failed' in settings.status
    assert rig.deadlock_avatar_scale == first_scale
    assert close(rig.matrix_world, world_before) and close(skin.matrix_world, skin_before)
    assert rig['deadlock_retarget_contact_offsets'] == contacts

    # Reapplying authored proportions retains the explicit size multiplier.
    rig.deadlock_avatar_scale = 1.4
    preserve_proportions(rig)
    assert abs(rig.deadlock_avatar_scale - 1.4) < 1e-6, settings.status
    assert abs(size.avatar_height(rig) - baseline * 1.4) < 1e-5

    # Real runtime constraint generation must keep the scaled preview placement.
    settings.collection = collection
    fake_hash = SimpleNamespace(murmur_hash2=lambda name, seed: 1)
    with patch.dict(sys.modules, {'SourceIO.library.source2.utils.murmurhash2': fake_hash}), \
            patch.object(retarget_export, 'runtime_root_reference', return_value=reference['root_motion']):
        exported, export_report, skeleton, rules = retarget_export.prepare_retarget_export(settings)
    assert abs(export_report['avatar_overall_size'] - 1.4) < 1e-6
    assert close(next(o for o in exported.objects if o.type == 'MESH' and not o.hide_render).matrix_world, skin.matrix_world)
    names = export_report['export_bone_names']
    rest = {name: bone['matrix'] for name, bone in skeleton.items()}
    expected = evaluate_rules(rest, reference, rules, skeleton)
    for name, matrix in idle(rig, None).items():
        rig.pose.bones[name].matrix_basis = matrix
    bpy.context.view_layer.update()
    for bone in rig.pose.bones:
        assert close(retarget_export.unit_matrix(rig.matrix_world @ bone.matrix), expected[names[bone.name]]), bone.name
    preview.restore_pose(rig)
    assert all(close(b.matrix_basis, Matrix(json.loads(saved_pose)['bones'][b.name])) for b in rig.pose.bones)

    # Each preview owns its value; Blender library snapshots preserve it for workers.
    second, _ = retarget.create_retarget_preview(SimpleNamespace(collection=source_collection, hero='NONE', proportions='PRESERVE'))
    second_rig = next(o for o in second.objects if o.type == 'ARMATURE')
    assert second_rig.deadlock_avatar_scale == 1
    assert abs(rig.deadlock_avatar_scale - 1.4) < 1e-6
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / 'snapshot.blend')
        bpy.data.libraries.write(path, {collection})
        with bpy.data.libraries.load(path) as (available, loaded):
            loaded.collections = [collection.name]
        bpy.context.scene.collection.children.link(loaded.collections[0])
        bpy.context.view_layer.update()
        restored = next(o for o in loaded.collections[0].objects if o.type == 'ARMATURE')
        assert abs(restored.deadlock_avatar_scale - 1.4) < 1e-6
        assert close(restored.matrix_world, rig.matrix_world)

addon.unregister()
print('PASS: uniform size, eye-height fit ignoring tall ears, missing-eye errors, grounding, rollback, idle restore and runtime export')
