"""Weighted body fitting, export selection, pose preservation and cached preview shapes.

blender -b --factory-startup --python-exit-code 1 --python tests/blender_physics_collision.py
Append -- --emit-compile-probe DIR for a standalone native compiler fixture.
"""
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile

import bpy
from mathutils import Matrix, Vector

folder = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('collision_test_addon', folder / '__init__.py')
addon = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = addon
spec.loader.exec_module(addon)
from collision_test_addon import physics, physics_collision
from collision_test_addon.physics_preview import Preview
from collision_test_addon.retarget_export import retarget_modeldoc, unit_matrix

addon.register()
collection = bpy.data.collections.new('Collision Fixture')
bpy.context.scene.collection.children.link(collection)
data = bpy.data.armatures.new('Collision Bones')
rig = bpy.data.objects.new('Collision Rig', data)
collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
rig.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
for name, parent, head, tail in [
        ('Hips', None, (0, 0, 0), (0, 0, 1)), ('Chest', 'Hips', (0, 0, 1), (0, 0, 2)),
        ('Head', 'Chest', (0, 0, 2), (0, 0, 3)),
        ('Tail', 'Hips', (0, -0.3, 0.7), (0, -0.55, 0.7)),
        ('TailTip', 'Tail', (0, -0.55, 0.7), (0, -0.8, 0.7)),
        ('Other', 'Hips', (1, 0, 0), (1, 0.3, 0)), ('OtherTip', 'Other', (1, 0.3, 0), (1, 0.6, 0))]:
    bone = data.edit_bones.new(name)
    bone.head, bone.tail = head, tail
    bone.parent = data.edit_bones.get(parent) if parent else None
bpy.ops.object.mode_set(mode='OBJECT')
rig.matrix_world = Matrix.Translation((0.3, -0.2, 0.1)) @ Matrix.Rotation(0.6, 4, 'Z') @ Matrix.Scale(2, 4)
rig['deadlock_retarget_mapping'] = json.dumps({'Hips':'pelvis', 'Chest':'spine_2', 'Head':'head'})
points, owners = [], []
for index, name in enumerate(('Hips', 'Chest', 'Head')):
    for z in range(11):
        for angle in range(24):
            a = angle*2*math.pi/24
            height = index+0.4+0.02*z if name == 'Head' else index+0.1+0.08*z
            points.append((0.2*math.cos(a), 0.2*math.sin(a), height))
            owners.append(name)
# Huge accessory geometry must not inflate the body's colliders.
points.extend([(40, 40, 40)]*20); owners.extend(['TailTip']*20)
mesh = bpy.data.meshes.new('Weighted Body')
mesh.from_pydata(points, [], [])
obj = bpy.data.objects.new('Body', mesh)
collection.objects.link(obj)
obj.matrix_world = rig.matrix_world.copy()
for name in set(owners):
    obj.vertex_groups.new(name=name).add([i for i, owner in enumerate(owners) if owner == name], 1, 'REPLACE')
modifier = obj.modifiers.new('Skin', 'ARMATURE'); modifier.object = rig
enabled = dict(root='Tail', stiffness=0.5, damping=0.3, gravity=0, auto_collision=True)
disabled = dict(root='Other', stiffness=0.2, damping=0.5, gravity=1)
rig[physics.KEY] = json.dumps([enabled, disabled])
rig.data.pose_position = 'POSE'
rig.pose.bones['Chest'].rotation_mode = 'XYZ'
rig.pose.bones['Chest'].rotation_euler.x = 0.2
bpy.context.view_layer.update()
before = {b.name:b.matrix_basis.copy() for b in rig.pose.bones}
settings_before = rig[physics.KEY]
names = {b.name:'export_' + b.name for b in data.bones}
nodes = physics.export_chains(rig, names)
colliders = physics_collision.fit_body_colliders(rig, names)
assert len(colliders) == 3 and all(0.1 < c['radius0']*addon.UNIT < 0.4 for c in colliders)
head_shape = next(c for c in colliders if c['parent_bone'] == names['Head'])
assert (Vector(head_shape['point0']) - Vector(head_shape['point1'])).length > 0.1, 'Short capsules must retain a native vertex-map filter'
assert all(row['vertex_map'] == physics_collision.VERTEX_MAP for row in nodes[0]['chain']['joints'])
assert not any('vertex_map' in row for row in nodes[1]['chain']['joints'])
assert rig.data.pose_position == 'POSE' and rig[physics.KEY] == settings_before
assert all(b.matrix_basis == before[b.name] for b in rig.pose.bones)
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'collision.physics.json'
    physics.save_preset(rig, path); physics.load_preset(rig, path)
    assert physics.chains(rig) == [enabled, disabled]
session = Preview(bpy.context, rig)
assert len(session.colliders) == 3
assert set(('Hips', 'Chest', 'Head')).issubset(session.goal_names)
goals = session.goals(bpy.context.evaluated_depsgraph_get())
capsules = physics_collision.world_capsules(session.colliders, goals, addon.UNIT)
assert capsules.shape == (3, 7)
session.apply(goals, 1/60)
assert len(session.solvers[0].colliding) and not len(session.solvers[1].colliding)
session.restore()
assert all(b.matrix_basis == before[b.name] for b in rig.pose.bones)
rig[physics.KEY] = json.dumps([dict(enabled, auto_collision=False), disabled])
assert physics_collision.fit_body_colliders(rig, names) == []
assert physics.export_chains(rig, names)[1] == nodes[1], 'An opted-out chain must remain byte-for-byte unchanged'
rig[physics.KEY] = settings_before

if '--emit-compile-probe' in sys.argv:
    destination = Path(sys.argv[sys.argv.index('--emit-compile-probe') + 1])
    destination.mkdir(parents=True, exist_ok=True)
    header = '<!-- kv3 encoding:text:version{e21c7f3c-8a33-41c5-9977-a76d3a32aa0d} format:modeldoc28:version{fb63b6ca-f435-4aa0-a2c7-c66ddc651dca} -->\n'
    template = header + '{ rootNode = { _class = "RootNode" children = [ { _class = "Skeleton" children = [] } ] } }'
    skeleton = {names[b.name]:dict(parent=names[b.parent.name] if b.parent else None,
                                  matrix=unit_matrix(rig.matrix_world @ b.matrix_local)) for b in data.bones}
    # The compiler only keeps a serialized model skeleton when a mesh imports it.
    # A tiny triangle per bone makes the native fixture independent of game assets.
    vertices, faces, weighted = [], [], []
    for bone in data.bones:
        start = len(vertices)
        vertices.extend([bone.head_local + Vector(offset) for offset in ((0, 0, 0), (0.01, 0, 0), (0, 0.01, 0))])
        faces.append((start, start+1, start+2))
        weighted.append((bone.name, [start, start+1, start+2]))
    render_data = bpy.data.meshes.new('Compiler Triangles'); render_data.from_pydata(vertices, [], faces)
    render = bpy.data.objects.new('Compiler Mesh', render_data); collection.objects.link(render)
    render.matrix_world = rig.matrix_world.copy()
    render.modifiers.new('Skin', 'ARMATURE').object = rig
    for name, indices in weighted: render.vertex_groups.new(name=name).add(indices, 1, 'REPLACE')
    for bone in data.bones: bone.name = names[bone.name]
    obj.hide_render = True
    bpy.context.view_layer.update()
    addon.export_fbx(type('Settings', (), {'collection':collection})(), destination/'body_collision.fbx', source2=True)
    doc = physics.append_physics_modeldoc(retarget_modeldoc(template, skeleton), nodes, colliders)
    doc = doc.replace('children = [', 'children = [ { _class = "RenderMeshList" children = [ '
                      '{ _class = "RenderMeshFile" name = "collision_fixture" filename = "body_collision.fbx" import_scale = 1.0 } ] },', 1)
    (destination / 'body_collision.vmdl').write_text(doc)
    (destination / 'expected.json').write_text(json.dumps({'nodes':nodes, 'colliders':colliders}))
addon.unregister()
print('PASS: weighted automatic body fitting, scale, accessory exclusion, presets, cached preview and per-chain selection')
