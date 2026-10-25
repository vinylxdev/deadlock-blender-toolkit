"""Motion-limit authoring, presets, UI lifecycle and native ModelDoc output.

blender -b --factory-startup --python-exit-code 1 --python tests/blender_physics_motion_limits.py
Append -- --emit-compile-probe DIR to emit standalone CSDK probe ModelDocs.
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import bpy
from mathutils import Matrix

folder = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('deadlock_blender_toolkit', folder / '__init__.py')
addon = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = addon
spec.loader.exec_module(addon)
from deadlock_blender_toolkit import physics, physics_curve
from deadlock_blender_toolkit.physics_solver import ChainSolver
from deadlock_blender_toolkit.retarget_export import retarget_modeldoc


def editors():
    return [tree for tree in bpy.data.node_groups if tree.get(physics_curve.EDITOR_KEY)]


addon.register()
settings = bpy.context.scene.deadlock_export
collection = bpy.data.collections.new('Motion Limit Test')
collection['deadlock_preview_only'] = True
collection['deadlock_hero'] = settings.hero
bpy.context.scene.collection.children.link(collection)
data = bpy.data.armatures.new('Limit Bones')
rig = bpy.data.objects.new('Limit Rig', data)
collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
rig.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
for name, parent, y in [('Root', None, 0), ('Middle', 'Root', 0.1), ('Tip', 'Middle', 0.4)]:
    bone = data.edit_bones.new(name)
    bone.head, bone.tail = (0, y, 0), (0, y + 0.1, 0)
    bone.parent = data.edit_bones.get(parent) if parent else None
bpy.ops.object.mode_set(mode='OBJECT')
rig.matrix_world = Matrix.Translation((0, 0, 1)) @ Matrix.Scale(2, 4)
names = {bone.name: bone.name for bone in data.bones}
base = dict(root='Root', stiffness=0, damping=0.5, gravity=1, lock_root_rotation=True)
radius_curve = {'enabled': True, 'points': [[0, 1, 'VECTOR'], [1, 0.25, 'VECTOR']]}
strength_curve = {'enabled': True, 'points': [[0, 1, 'VECTOR'], [1, 0.5, 'VECTOR']]}
limited = dict(base, use_motion_limits=True, motion_limit=0.04, motion_limit_strength=0.65,
               motion_limit_curve=radius_curve, motion_limit_strength_curve=strength_curve)


def export(entry):
    rig[physics.KEY] = json.dumps([entry])
    before = len(editors())
    node, = physics.export_chains(rig, names)
    assert len(editors()) == before
    return node


ordinary = export(base)
assert all(type(row['goal_strength']) is float and type(row['gravity_z']) is float
           for row in ordinary['chain']['joints']), 'ModelDoc numeric attributes require floats'
assert export(dict(limited, use_motion_limits=False)) == ordinary
node = export(limited)
expected = {'Root': (0.04, 0.65), 'Middle': (0.0325, 0.56875), 'Tip': (0.01, 0.325)}
for row in node['chain']['joints']:
    radius, strength = expected[row['joint_name']]
    assert abs(row['stray_radius'] * addon.UNIT - radius) < 1e-6
    assert abs(row['stray_radius_stretchiness'] - (1 - strength)) < 1e-6
    original = next(r for r in ordinary['chain']['joints'] if r['joint_name'] == row['joint_name'])
    assert {k: v for k, v in row.items() if not k.startswith('stray_radius')} == original
solver = ChainSolver(node, {bone.name: rig.matrix_world @ bone.matrix_local for bone in data.bones}, addon.UNIT)
for index, name in enumerate(solver.owners):
    assert abs(solver.limit_radius[index, 0] - expected[name][0]) < 1e-6
try:
    export(dict(limited, lock_twist=True))
    raise AssertionError('An unsupported jiggle limit was accepted')
except ValueError as error:
    assert 'turn off Lock twist' in str(error)
for entry in (dict(limited, motion_limit=0), dict(limited, motion_limit_strength=0)):
    assert all(row['stray_radius'] == 0 for row in export(entry)['chain']['joints'])

export(limited)
with tempfile.TemporaryDirectory() as folder_name:
    path = Path(folder_name) / 'limits.physics.json'
    physics.save_preset(rig, path)
    expected_settings = physics.validate_chains(rig)
    rig[physics.KEY] = '[]'
    physics.load_preset(rig, path)
    assert physics.chains(rig) == expected_settings
    text = path.read_text()
    for key, invalid in [('motion_limit', -0.1), ('motion_limit_strength', 1.5), ('lock_twist', True)]:
        document = json.loads(text)
        document['chains'][0][key] = invalid
        path.write_text(json.dumps(document))
        try:
            physics.load_preset(rig, path)
            raise AssertionError('Invalid settings loaded')
        except ValueError:
            assert physics.chains(rig) == expected_settings

op = SimpleNamespace(index=0, rig_name=rig.name, root='Root', stiffness=0, damping=0.5, gravity=1,
                     twist_resistance=0.2, lock_root_rotation=True, lock_twist=False,
                     anchor_child_roots=False, stable_rotation_frames=False, report=lambda *args: None)
context = SimpleNamespace(scene=bpy.context.scene, object=rig,
                          window_manager=SimpleNamespace(invoke_props_dialog=lambda *args, **kwargs: {'RUNNING_MODAL'}))
assert addon.DL_OT_physics_chain.invoke(op, context, None) == {'RUNNING_MODAL'}
assert op.use_motion_limits and op.use_motion_limit_curve and op.use_motion_limit_strength_curve
assert op.motion_limit == 0.04 and op.motion_limit_strength == 0.65
assert len(editors()) == len(physics_curve.CURVE_SETTINGS)
class Layout:
    def __init__(self): self.widgets = []
    def prop_search(self, *args, **kwargs): pass
    def label(self, **kwargs): pass
    def prop(self, *args, **kwargs): pass
    def box(self): return self
    def row(self, **kwargs): return self
    def column(self, **kwargs): return self
    def split(self, **kwargs): return self
    def separator(self): pass
    def template_curve_mapping(self, node, *args, **kwargs): self.widgets.append(node.id_data.name)
op.layout = Layout()
addon.DL_OT_physics_chain.draw(op, context)
assert not op.layout.widgets, 'Motion tab must stay compact with limits enabled'
op.settings_tab = 'CURVES'
for key in ('motion_limit', 'motion_limit_strength'):
    op.curve_setting = key
    op.layout.widgets.clear()
    addon.DL_OT_physics_chain.draw(op, context)
    assert op.layout.widgets == [getattr(op, key + '_curve_name')]
assert addon.DL_OT_physics_chain.execute(op, context) == {'FINISHED'}
assert not editors()
assert addon.DL_OT_physics_chain.execute(op, context) == {'FINISHED'}
assert physics.chains(rig)[0]['motion_limit_curve']['enabled']
assert addon.DL_OT_physics_chain.invoke(op, context, None) == {'RUNNING_MODAL'}
before = rig[physics.KEY]
addon.DL_OT_physics_chain.cancel(op, context)
assert not editors() and rig[physics.KEY] == before
op.index = -1
assert addon.DL_OT_physics_chain.invoke(op, context, None) == {'RUNNING_MODAL'}
assert not op.use_motion_limits and not op.use_motion_limit_curve
addon.DL_OT_physics_chain.cancel(op, context)
assert not editors()
properties = bpy.ops.deadlock.physics_chain.get_rna_type().properties
assert properties['motion_limit'].unit == 'LENGTH'

if '--emit-compile-probe' in sys.argv:
    destination = Path(sys.argv[sys.argv.index('--emit-compile-probe') + 1])
    destination.mkdir(parents=True, exist_ok=True)
    header = '<!-- kv3 encoding:text:version{e21c7f3c-8a33-41c5-9977-a76d3a32aa0d} format:modeldoc28:version{fb63b6ca-f435-4aa0-a2c7-c66ddc651dca} -->\n'
    template = header + '{ rootNode = { _class = "RootNode" children = [ { _class = "Skeleton" children = [] } ] } }'
    skeleton = {bone.name: dict(parent=bone.parent.name if bone.parent else None,
                               matrix=rig.matrix_world @ bone.matrix_local) for bone in data.bones}
    for mode, entry in [('soft', limited), ('hard', dict(limited, motion_limit_strength=1,
                                                       motion_limit_strength_curve=dict(strength_curve, enabled=False))),
                        ('off', dict(limited, use_motion_limits=False))]:
        node = export(entry)
        document = physics.append_physics_modeldoc(retarget_modeldoc(template, skeleton), [node])
        (destination / ('motion_limits_' + mode + '.vmdl')).write_text(document)
        (destination / ('motion_limits_' + mode + '.json')).write_text(json.dumps([node]))
addon.unregister()
print('Motion limit authoring passed: meter conversion, radius/strength curves, disabled compatibility, preview settings, presets, cloth-only validation, editor lifecycle and redo')
