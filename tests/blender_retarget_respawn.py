"""Static countdown channels must contain the retargeted pose, including graph-root correction.

blender -b --factory-startup --python-exit-code 1 --python tests/blender_retarget_respawn.py
"""
import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import bpy
from mathutils import Matrix

sys.path.insert(0, bpy.utils.user_resource('SCRIPTS', path='addons'))
try:
    importlib.import_module('SourceIO')
except ImportError:
    importlib.import_module('SourceIO-master')
folder = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('respawn_test_addon', folder / '__init__.py')
addon = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = addon
spec.loader.exec_module(addon)
from respawn_test_addon.retarget_respawn import write_pose_animation, bake_respawn_idle
from respawn_test_addon.retarget_validation import native_idle, evaluate_rules

native = [('root_motion', None, (0, 0, 0)), ('pelvis', 'root_motion', (0, 0, 1)),
          ('head', 'pelvis', (0, 0, 1.7)), ('hand_L', 'pelvis', (0.6, 0, 1.4)),
          ('hand_R', 'pelvis', (-0.6, 0, 1.4))]
skeleton = {name: {'parent': parent, 'matrix': Matrix.Translation(position)} for name, parent, position in native}
skeleton['avatar'] = {'parent': 'root_motion', 'matrix': Matrix.Translation((0, 0, 0.4))}
skeleton['tail'] = {'parent': 'avatar', 'matrix': Matrix.Translation((0, -0.4, 0.4))}
rest = {n: b['matrix'] for n, b in skeleton.items()}
rest_before = {n:m.copy() for n,m in rest.items()}
pose = {name: Matrix.Translation(m.translation)
        @ Matrix.Rotation(0.3, 4, 'X') for name, m in rest.items()}
pose['head'] = Matrix.Translation((0.1, 0.2, 1.6)) @ Matrix.Rotation(0.8, 4, 'Y')
bone_rows = [SimpleNamespace(name=name, parent=parent) for name, parent, _ in native]
reference = {b.name: rest[b.name] for b in bone_rows}
constraints = [{'_class': 'CPointConstraint', 'm_slaves': [{'m_sName': 'avatar', 'm_flWeight': 1.0}],
                'm_targets': [{'m_sName': 'root_motion', 'm_flWeight': 1.0, 'm_vOffset': [0, 0, 0.4 / addon.UNIT]}]}]


def error(actual, expected):
    return max(abs(actual[n][i][j] - expected[n][i][j]) for n in expected for i in range(4) for j in range(4))


with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'pose.dmx'
    write_pose_animation(path, skeleton, pose)
    all_bones = [SimpleNamespace(name=n, parent=b['parent']) for n, b in skeleton.items()]
    for time in (0, 1 / 30, 10):
        decoded = native_idle(rest, all_bones, path, time_seconds=time)
        assert error(decoded, pose) < 1e-6, 'Local channels must hold the same pose at every sample'
    # Bake from a native-only source, as happens for extracted hero animations.
    write_pose_animation(path, {n:skeleton[n] for n in reference}, {n:pose[n] for n in reference})
    output = Path(directory) / 'baked.dmx'
    runtime_root = Matrix.Rotation(-0.6, 4, 'Y')
    with patch('respawn_test_addon.reference_bones', return_value=bone_rows), \
            patch('respawn_test_addon.preview.idle_source', return_value=('idle', path)), \
            patch('respawn_test_addon.retarget_export.runtime_root_reference', return_value=runtime_root):
        report, baked = bake_respawn_idle(None, skeleton, constraints, output)
    native_pose = {n:pose[n].copy() for n in reference}
    model_root = native_pose['root_motion'].copy()
    native_pose['root_motion'] = native_pose['root_motion'] @ reference['root_motion'].inverted() @ runtime_root
    expected = evaluate_rules(rest, native_pose, constraints, skeleton)
    expected['root_motion'] = model_root
    assert error(baked, expected) < 1e-6
    assert error(native_idle(rest, all_bones, output), expected) < 1e-6
    assert report['idle_frame'] == 0 and report['bones'] == len(skeleton)
    assert error(rest, rest_before) == 0, 'Baking must preserve the bind skeleton'
print('PASS: static countdown channels, avatar hierarchy, frame 0 and graph-root correction')
