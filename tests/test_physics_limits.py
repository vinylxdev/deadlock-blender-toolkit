"""Numerical proof of cloth motion bounds and portable setting validation."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest

import numpy as np

folder = Path(__file__).parents[1]
package = types.ModuleType('motion_limit_tests')
package.__path__ = [str(folder)]
sys.modules[package.__name__] = package
for name in ('physics', 'physics_solver'):
    spec = importlib.util.spec_from_file_location(package.__name__ + '.' + name, folder / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    setattr(package, name, module)
ChainSolver = package.physics_solver.ChainSolver


def chain(radius=None, strength=1, unit=1):
    rows, bind = [], {}
    for i in range(4):
        name = 'bone' + str(i)
        matrix = np.eye(4)
        matrix[:3, 3] = (0, 0.2 * i, 1)
        bind[name] = matrix
        row = dict(joint_name=name, joint_parent='bone' + str(i - 1) if i else '',
                   simulate=i > 0, allow_rotation=i > 0, goal_strength=0, goal_damping=0.5,
                   gravity_z=1, mass=1, drag=0.2, extra_iterations=2, extrude_sides=2,
                   extrude_radius=0.02 / unit, stretch_spring=1, bend_spring=0.8,
                   torsion_spring=0.2, end_effector=0.2 / unit if i == 3 else 0)
        if radius is not None:
            row.update(stray_radius=radius / unit, stray_radius_stretchiness=1 - strength)
        rows.append(row)
    return ChainSolver(dict(chain=dict(joints=rows, attrs={})), bind, unit), bind


def simulate(solver, goals, fps=60):
    for _ in range(fps):
        solver.advance(goals, 1 / fps)


class MotionLimits(unittest.TestCase):
    def test_optional_settings_validate_without_blender(self):
        entry = dict(root='Tail', stiffness=0.5, damping=0.3, gravity=0.5)
        self.assertEqual(package.physics.validated_entries([entry]), [entry])
        enabled = dict(entry, use_motion_limits=True, motion_limit=0.009, motion_limit_strength=0.65)
        self.assertEqual(package.physics.validated_entries([enabled]), [enabled])
        for key, values in (('motion_limit', [-1, True, float('nan'), float('inf')]),
                            ('motion_limit_strength', [-1, 1.01, True, float('nan')]),
                            ('use_motion_limits', [1, 'true', None])):
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    package.physics.validated_entries([dict(enabled, **{key: value})])
        with self.assertRaisesRegex(ValueError, 'turn off Lock twist'):
            package.physics.validated_entries([dict(enabled, lock_twist=True)])
        package.physics.validated_entries([dict(enabled, use_motion_limits=False, lock_twist=True)])

    def test_hard_limits_bound_actual_motion_in_meters(self):
        limited, goals = chain(radius=0.009, unit=0.0254)
        free, _ = chain(unit=0.0254)
        simulate(limited, goals)
        simulate(free, goals)
        displacement = np.linalg.norm(limited.position - limited.targets(goals), axis=1)
        self.assertLessEqual(displacement.max(), 0.009 + 1e-10)
        self.assertGreater(np.linalg.norm(free.position - free.targets(goals), axis=1).max(), 0.1)
        np.testing.assert_allclose(limited.matrices(goals)['bone0'], goals['bone0'], atol=1e-10)

    def test_compiled_limits_reject_missing_or_changed_native_constraints(self):
        rows = [dict(joint_name='Bone', simulate=True, allow_rotation=True,
                     stray_radius=2.0, stray_radius_stretchiness=0.35)]
        fe = dict(m_CtrlName=['Anchor', 'Bone'], m_nStaticNodes=1,
                  m_AnimStrayRadii=[dict(nNode=[1, 1], flMaxDist=2.0, flRelaxationFactor=0.65)])
        checked = package.physics.validate_compiled_limits(fe, rows, 0.0254)
        self.assertAlmostEqual(checked[0]['radius_meters'], 0.0508)
        for records in ([], [dict(nNode=[1, 1], flMaxDist=3, flRelaxationFactor=0.65)],
                        [dict(nNode=[1, 1], flMaxDist=2, flRelaxationFactor=1)]):
            with self.assertRaisesRegex(ValueError, 'did not compile'):
                package.physics.validate_compiled_limits(dict(fe, m_AnimStrayRadii=records), rows, 0.0254)
        with self.assertRaisesRegex(ValueError, 'Disabled motion limit'):
            package.physics.validate_compiled_limits(fe, [dict(rows[0], stray_radius=0)], 0.0254)

    def test_rotating_static_anchor_does_not_require_a_motion_constraint(self):
        rows = [dict(joint_name='Root', simulate=False, allow_rotation=True,
                     stray_radius=4.0, stray_radius_stretchiness=0.0),
                dict(joint_name='Child', simulate=True, allow_rotation=True,
                     stray_radius=4.0, stray_radius_stretchiness=0.0)]
        fe = dict(m_CtrlName=['Root', '$ccRoot_0', '$ccRoot_1', 'Child', '$ccChild_0'],
                  m_nStaticNodes=3,
                  m_AnimStrayRadii=[dict(nNode=[3, 3], flMaxDist=4.0, flRelaxationFactor=1.0),
                                   dict(nNode=[4, 4], flMaxDist=4.0, flRelaxationFactor=1.0)])
        checked = package.physics.validate_compiled_limits(fe, rows, 0.0254)
        self.assertEqual([item['bone'] for item in checked], ['Child'])
        with self.assertRaisesRegex(ValueError, 'did not compile.*Child'):
            package.physics.validate_compiled_limits(dict(fe, m_AnimStrayRadii=[]), rows, 0.0254)

    def test_zero_radius_and_zero_strength_preserve_unlimited_motion(self):
        free, goals = chain()
        zero_radius, _ = chain(radius=0)
        zero_strength, _ = chain(radius=0.009, strength=0)
        for solver in (free, zero_radius, zero_strength):
            simulate(solver, goals)
        np.testing.assert_array_equal(zero_radius.position, free.position)
        np.testing.assert_array_equal(zero_strength.position, free.position)

    def test_soft_limit_restricts_motion_but_allows_overshoot(self):
        hard, goals = chain(radius=0.009)
        soft, _ = chain(radius=0.009, strength=0.25)
        free, _ = chain()
        for solver in (hard, soft, free):
            simulate(solver, goals)
        def displacement(solver):
            return np.linalg.norm(solver.position - solver.targets(goals), axis=1).max()
        self.assertGreater(displacement(soft), displacement(hard) + 1e-6)
        self.assertLess(displacement(soft), displacement(free) * 0.1)

    def test_limits_leave_motion_inside_the_radius_untouched(self):
        solver, goals = chain(radius=0.02)
        solver.position[~solver.fixed, 0] += 0.002
        before = solver.position.copy()
        solver._project_limits(solver.targets(goals))
        np.testing.assert_array_equal(solver.position, before)

    def test_hard_limits_follow_goals_between_fixed_steps(self):
        for elapsed in (1 / 200, 1 / 90):
            solver, goals = chain(radius=0.009)
            moved = {name: matrix.copy() for name, matrix in goals.items()}
            for matrix in moved.values():
                matrix[0, 3] += 0.1
            solver.advance(moved, elapsed)
            error = np.linalg.norm(solver.position - solver.targets(moved), axis=1)
            self.assertLessEqual(error.max(), 0.009 + 1e-10)

    def test_soft_limits_keep_fixed_step_frame_rate_invariance(self):
        positions = []
        for fps in (30, 60, 120):
            solver, goals = chain(radius=0.009, strength=0.65)
            simulate(solver, goals, fps)
            positions.append(solver.position)
        for position in positions[1:]:
            np.testing.assert_allclose(position, positions[0], atol=1e-10)


if __name__ == '__main__':
    unittest.main()
