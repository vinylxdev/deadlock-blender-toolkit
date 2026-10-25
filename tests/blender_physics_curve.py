"""Factory-startup proof of the editor, presets, preview settings and both export paths.

blender -b --factory-startup --python-exit-code 1 --python tests/blender_physics_curve.py
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


def editors():
    return {tree.name for tree in bpy.data.node_groups if tree.get(physics_curve.EDITOR_KEY)}


def close(actual, expected):
    assert len(actual) == len(expected)
    assert all(abs(a - b) < 2e-4 for a, b in zip(actual, expected)), (actual, expected)


addon.register()
settings = bpy.context.scene.deadlock_export
collection = bpy.data.collections.new('Curve Test')
collection['deadlock_preview_only'] = True
collection['deadlock_hero'] = settings.hero
bpy.context.scene.collection.children.link(collection)
data = bpy.data.armatures.new('Curve Bones')
rig = bpy.data.objects.new('Curve Rig', data)
collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
rig.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
specs = [('Root', None, (0, 0, 0)), ('Shared', 'Root', (0, 1, 0)),
         ('Short', 'Shared', (1, 1, 0)), ('Long', 'Shared', (0, 3, 0)),
         ('Tip', 'Long', (0, 4, 0))]
for name, parent, head in specs:
    bone = data.edit_bones.new(name)
    bone.head = head
    bone.tail = (head[0], head[1] + 0.25, head[2])
    bone.parent = data.edit_bones.get(parent) if parent else None
bpy.ops.object.mode_set(mode='OBJECT')
rig.matrix_world = Matrix.Translation((0.3, -0.4, 1)) @ Matrix.Rotation(0.7, 4, 'Z') @ Matrix.Scale(1.5, 4)
entry = dict(root='Root', stiffness=0.8, damping=0.3, gravity=0.5, lock_root_rotation=True)
names = {name: name for name, _, _ in specs}


def export(value):
    rig[physics.KEY] = json.dumps([value])
    before = editors()
    node, = physics.export_chains(rig, names)
    assert editors() == before, 'Export leaked a curve editor'
    return node


ordinary = export(entry)
flat = dict(physics_curve.default_curve(), enabled=True)
definition = {'enabled': True, 'points': [[0, 1, 'VECTOR'], [1, 0.25, 'VECTOR']]}
positions = dict(Root=0, Shared=0.25, Short=1, Long=0.75, Tip=1)
for key, (field, default) in physics_curve.SETTINGS.items():
    for lock_twist in (False, True):
        baseline = export(dict(entry, lock_twist=lock_twist))
        assert export(dict(entry, lock_twist=lock_twist, **{key + '_curve': flat})) == baseline
        disabled = dict(definition, enabled=False)
        assert export(dict(entry, lock_twist=lock_twist, **{key + '_curve': disabled})) == baseline
    curved = export(dict(entry, **{key + '_curve': definition}))
    for row, original in zip(curved['chain']['joints'], ordinary['chain']['joints']):
        expected = entry.get(key, default) * (1 - 0.75 * positions[row['joint_name']])
        close([row[field]], [expected])
        assert {k: v for k, v in row.items() if k != field} == {
            k: v for k, v in original.items() if k != field}, 'One curve changed another setting'
node = export(dict(entry, stiffness_curve=definition))
strengths = {row['joint_name']: row['goal_strength'] for row in node['chain']['joints']}
for name, expected in dict(Root=0.8, Shared=0.65, Short=0.2, Long=0.35, Tip=0.2).items():
    close([strengths[name]], [expected])
for row, original in zip(node['chain']['joints'], ordinary['chain']['joints']):
    assert {k: v for k, v in row.items() if k != 'goal_strength'} == {
        k: v for k, v in original.items() if k != 'goal_strength'}
bind = {bone.name: rig.matrix_world @ bone.matrix_local for bone in data.bones}
solver = ChainSolver(node, bind, addon.UNIT)
close([row['goal_strength'] for row in solver.rows], [strengths[row['joint_name']] for row in solver.rows])
uniform = ChainSolver(ordinary, bind, addon.UNIT)
for _ in range(60):
    solver.advance(bind, 1 / 60)
    uniform.advance(bind, 1 / 60)
assert solver.matrices(bind)['Tip'][2, 3] < uniform.matrices(bind)['Tip'][2, 3] - 1e-5, 'Curve did not loosen the tip in preview'
swing = export(dict(entry, stiffness_curve=definition, lock_twist=True))
for helper, jiggle in zip(swing['_swing_only']['helpers'], swing['_swing_only']['jigglebones']):
    close([jiggle['yaw_stiffness'], jiggle['pitch_stiffness']], [1000 * strengths[helper['target']]] * 2)

# Independent damping and gravity profiles must reach native jiggle controls,
# including gravity slider values above 1, without using the chain-wide value.
profiles = {
    'stiffness_curve': definition,
    'damping_curve': {'enabled': True, 'points': [[0, 0.25, 'VECTOR'], [1, 1, 'VECTOR']]},
    'gravity_curve': {'enabled': True, 'points': [[0, 0, 'VECTOR'], [1, 1, 'VECTOR']]},
    'twist_resistance_curve': {'enabled': True, 'points': [[0, 1, 'VECTOR'], [1, 0, 'VECTOR']]},
}
combined = export(dict(entry, gravity=2, lock_twist=True, **profiles))
by_name = {row['joint_name']: row for row in combined['chain']['joints']}
for row in by_name.values():
    t = positions[row['joint_name']]
    close([row['goal_strength'], row['goal_damping'], row['gravity_z'], row['torsion_spring']],
          [0.8 * (1 - 0.75 * t), 0.3 * (0.25 + 0.75 * t), 2 * t, 0.2 * (1 - t)])
for helper, jiggle in zip(combined['_swing_only']['helpers'], combined['_swing_only']['jigglebones']):
    row = by_name[helper['target']]
    close([jiggle['yaw_stiffness'], jiggle['pitch_stiffness']], [1000 * row['goal_strength']] * 2)
    close([jiggle['yaw_damping'], jiggle['pitch_damping']], [10 * row['goal_damping']] * 2)
    close([jiggle['tip_mass']], [row['gravity_z'] * 9.80665 / addon.UNIT])
# Twist resistance is a cloth-only spring. It must not change native jiggles.
without_torsion = export(dict(entry, gravity=2, lock_twist=True,
                              **{k: v for k, v in profiles.items() if k != 'twist_resistance_curve'}))
assert combined['_swing_only'] == without_torsion['_swing_only']

# A zero gravity curve must remove gravitational sag from the actual preview.
zero_gravity = export(dict(entry, gravity_curve={'enabled': True, 'points': [[0, 0, 'VECTOR'], [1, 0, 'VECTOR']]}))
floating = ChainSolver(zero_gravity, bind, addon.UNIT)
uniform = ChainSolver(ordinary, bind, addon.UNIT)
initial = floating.position.copy()
for _ in range(60):
    floating.advance(bind, 1 / 60)
    uniform.advance(bind, 1 / 60)
assert abs(floating.position - initial).max() < 1e-8
assert floating.matrices(bind)['Tip'][2, 3] > uniform.matrices(bind)['Tip'][2, 3] + 1e-5

# Curved handles are restored exactly and evaluated by Blender, rather than
# replacing the user's curve with straight interpolation or sampled points.
smooth = {'enabled': True, 'points': [[0, 1, 'AUTO_CLAMPED'], [0.4, 0.9, 'AUTO_CLAMPED'], [1, 0.1, 'VECTOR']]}
tree = physics_curve.new_editor(smooth)
restored = physics_curve.read_editor(tree, True)
for actual, expected in zip(restored['points'], smooth['points']):
    close(actual[:2], expected[:2])
    assert actual[2] == expected[2]
mapping = tree.nodes[0].mapping
sampled = [mapping.evaluate(mapping.curves[0], t) for t in (0, 0.2, 0.7, 1)]
physics_curve.remove_editor(tree.name)
close(physics_curve.multipliers(restored, (0, 0.2, 0.7, 1)), sampled)
extended = dict(restored, extend='EXTRAPOLATED')
tree = physics_curve.new_editor(extended)
assert physics_curve.read_editor(tree, True)['extend'] == 'EXTRAPOLATED'
physics_curve.remove_editor(tree.name)
export(dict(entry, **dict(profiles, stiffness_curve=restored)))
with tempfile.TemporaryDirectory() as temporary:
    path = Path(temporary) / 'curve.physics.json'
    physics.save_preset(rig, path)
    before = physics.validate_chains(rig)
    rig[physics.KEY] = '[]'
    physics.load_preset(rig, path)
    assert physics.chains(rig) == before
    valid_text = path.read_text()
    for key in physics_curve.SETTINGS:
        broken = json.loads(valid_text)
        broken['chains'][0][key + '_curve']['points'][0][1] = 2
        path.write_text(json.dumps(broken))
        try:
            physics.load_preset(rig, path)
            raise AssertionError('Invalid preset was accepted: ' + key)
        except ValueError:
            assert physics.chains(rig) == before

# Exercise the same operator methods used by the dialog, including reopen,
# disable/re-enable, cancellation, redo and direct EXEC_DEFAULT calls.
context = SimpleNamespace(scene=bpy.context.scene, object=rig,
                          window_manager=SimpleNamespace(invoke_props_dialog=lambda *args, **kwargs: {'RUNNING_MODAL'}))
op = SimpleNamespace(index=0, rig_name=rig.name, root='Root', stiffness=0.8, damping=0.3, gravity=0.5,
                     twist_resistance=0.2, lock_root_rotation=True, lock_twist=False,
                     anchor_child_roots=False, stable_rotation_frames=False, report=lambda *args: None)
assert addon.DL_OT_physics_chain.invoke(op, context, None) == {'RUNNING_MODAL'}
for key in physics_curve.SETTINGS:
    assert getattr(op, 'use_' + key + '_curve')
    assert physics_curve.editor(getattr(op, key + '_curve_name'))
assert len(editors()) == len(physics_curve.CURVE_SETTINGS)

class Layout:
    def __init__(self): self.curves = []
    def prop_search(self, *args, **kwargs): pass
    def label(self, **kwargs): pass
    def prop(self, *args, **kwargs): pass
    def box(self): return self
    def row(self, **kwargs): return self
    def column(self, **kwargs): return self
    def split(self, **kwargs): return self
    def separator(self): pass
    def template_curve_mapping(self, node, *args, **kwargs): self.curves.append(node.id_data.name)

op.layout = Layout()
addon.DL_OT_physics_chain.draw(op, context)
assert not op.layout.curves, 'Motion tab expanded curve editors'
op.settings_tab = 'CURVES'
for key in physics_curve.SETTINGS:
    op.curve_setting = key
    op.layout.curves.clear()
    addon.DL_OT_physics_chain.draw(op, context)
    assert op.layout.curves == [getattr(op, key + '_curve_name')], 'Curves tab must show only the selected editor'
op.settings_tab = 'ADVANCED'
op.layout.curves.clear()
addon.DL_OT_physics_chain.draw(op, context)
assert not op.layout.curves, 'Advanced tab expanded curve editors'
assert addon.DL_OT_physics_chain.execute(op, context) == {'FINISHED'}
assert not editors()
assert addon.DL_OT_physics_chain.execute(op, context) == {'FINISHED'}, 'Redo lost its curve'
before = physics.chains(rig)[0]
assert 'settings_tab' not in before and 'curve_setting' not in before
for key in physics_curve.SETTINGS:
    assert before[key + '_curve']['enabled']
assert addon.DL_OT_physics_chain.invoke(op, context, None) == {'RUNNING_MODAL'}
op.use_damping_curve = False
assert addon.DL_OT_physics_chain.execute(op, context) == {'FINISHED'}
disabled = physics.chains(rig)[0]['damping_curve']
assert not disabled['enabled']
for key in ('stiffness', 'gravity', 'twist_resistance'):
    assert physics.chains(rig)[0][key + '_curve'] == before[key + '_curve']
assert addon.DL_OT_physics_chain.invoke(op, context, None) == {'RUNNING_MODAL'}
assert not op.use_damping_curve
assert physics_curve.read_editor(physics_curve.editor(op.damping_curve_name), False) == disabled
before = rig[physics.KEY]
addon.DL_OT_physics_chain.cancel(op, context)
assert rig[physics.KEY] == before and not editors()
op.use_damping_curve = True
op.damping_curve_data = json.dumps(dict(disabled, enabled=True))
assert addon.DL_OT_physics_chain.execute(op, context) == {'FINISHED'}
assert physics.chains(rig)[0]['damping_curve']['enabled']

# Editing another chain must start with fresh shapes and disabled toggles.
op.index = -1
assert addon.DL_OT_physics_chain.invoke(op, context, None) == {'RUNNING_MODAL'}
for key in physics_curve.SETTINGS:
    assert not getattr(op, 'use_' + key + '_curve')
    tree = physics_curve.editor(getattr(op, key + '_curve_name'))
    assert physics_curve.read_editor(tree, False) == physics_curve.default_curve()
addon.DL_OT_physics_chain.cancel(op, context)
assert rig[physics.KEY] != '[]' and not editors()
kwargs = {prop: value for key in physics_curve.SETTINGS
          for prop, value in [('use_' + key + '_curve', True), (key + '_curve_data', json.dumps(profiles[key + '_curve']))]}
assert bpy.ops.deadlock.physics_chain(root='Root', rig_name=rig.name, **kwargs) == {'FINISHED'}
assert all(physics.chains(rig)[0][key + '_curve']['enabled'] for key in physics_curve.SETTINGS)
assert not editors()
addon.unregister()
print('Physics curve checks passed: all four sliders, distance, branches, independent values/editors, flat/disabled compatibility, preview motion, cloth/jiggle export, smooth handles, presets, dialog lifecycle and redo')
