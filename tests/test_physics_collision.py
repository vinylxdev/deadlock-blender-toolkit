"""Body fitting, collision projection and opt-in compatibility without Blender."""
import importlib
from pathlib import Path
import sys
import types
import unittest

import numpy as np

package = types.ModuleType('collision_tests')
package.__path__ = [str(Path(__file__).parents[1])]
sys.modules[package.__name__] = package
collision = importlib.import_module(package.__name__ + '.physics_collision')
physics = importlib.import_module(package.__name__ + '.physics')
ChainSolver = importlib.import_module(package.__name__ + '.physics_solver').ChainSolver


class BodyCollisions(unittest.TestCase):
    def test_fits_weighted_surface_and_ignores_outliers(self):
        points = np.array([(0.2*np.cos(a), y, 0.2*np.sin(a))
                           for y in np.linspace(0, 2, 21) for a in np.linspace(0, 2*np.pi, 24, endpoint=False)])
        first, last, radius = collision.fit_capsule(points, np.zeros(3), (0, 1, 0))
        self.assertAlmostEqual(radius, 0.18)
        self.assertTrue(np.allclose((first+last)/2, (0, 1, 0)))
        outliers = np.vstack((points, [100, 100, 100]))
        self.assertAlmostEqual(collision.fit_capsule(outliers, np.zeros(3), (0, 1, 0))[2], radius)
        shifted = points * 3 + (4, 5, 6)
        a, b, r = collision.fit_capsule(shifted, np.array((4, 5, 6)), (0, 1, 0))
        self.assertTrue(np.allclose(a, first*3+(4, 5, 6)))
        self.assertTrue(np.allclose(b, last*3+(4, 5, 6)))
        self.assertAlmostEqual(r, radius*3)

    def test_no_fabricated_fit_for_missing_or_degenerate_body(self):
        for points, direction in (([], (0, 1, 0)), (np.zeros((20, 3)), (0, 1, 0)),
                                  (np.ones((20, 3)), (0, 0, 0)), (np.full((20, 3), np.nan), (0, 1, 0))):
            self.assertIsNone(collision.fit_capsule(points, np.zeros(3), direction))

    def test_short_body_regions_keep_a_capsule_axis_for_native_filtering(self):
        points = np.array([(np.cos(a), y, np.sin(a)) for y in np.linspace(0, 0.2, 20)
                           for a in np.linspace(0, 2*np.pi, 24, endpoint=False)])
        first, last, radius = collision.fit_capsule(points, np.zeros(3), (0, 1, 0))
        self.assertGreater(np.linalg.norm(last-first), 0.01)
        self.assertGreater(radius, 0)

    def test_projection_handles_thickness_centerline_and_spheres(self):
        for capsule in ([0, -1, 0, 0, 1, 0, 0.2], [0, 0, 0, 0, 0, 0, 0.2]):
            points = np.array([[0, 0, 0], [0, 0, 0], [2, 0, 0]], dtype=float)
            collision.project_capsules(points, np.array([1, 2]), np.array([0.03, 0.03]), np.array([capsule]))
            self.assertTrue(np.isfinite(points).all())
            self.assertTrue(np.array_equal(points[0], [0, 0, 0]), 'Excluded particles must stay unchanged')
            self.assertAlmostEqual(np.linalg.norm(points[1]), 0.23)
            self.assertTrue(np.array_equal(points[2], [2, 0, 0]))

    def test_capsules_follow_body_pose_without_scaling_distances_twice(self):
        collider = dict(parent_bone='body', point0=[0, 0, 0], point1=[0, 2, 0], radius0=1)
        matrix = np.diag((3., 3., 3., 1.)); matrix[:3, 3] = (2, 3, 4)
        actual = collision.world_capsules([collider], {'body':matrix}, 0.1)
        self.assertTrue(np.allclose(actual, [[2, 3, 4, 2, 3.2, 4, 0.1]]))

    def test_old_presets_stay_opted_out_and_invalid_options_fail(self):
        entry = dict(root='Tail', stiffness=0.5, damping=0.3, gravity=0)
        self.assertEqual(physics.validated_entries([entry]), [entry])
        self.assertEqual(physics.validated_entries([dict(entry, auto_collision=True)])[0]['auto_collision'], True)
        for value in (0, 1, 'true', None):
            with self.assertRaisesRegex(ValueError, 'auto_collision'):
                physics.validated_entries([dict(entry, auto_collision=value)])
        with self.assertRaisesRegex(ValueError, 'turn off Lock twist'):
            physics.validated_entries([dict(entry, auto_collision=True, lock_twist=True)])

    def test_solver_only_collides_when_chain_is_enabled(self):
        bind = {}; rows = []
        for i, name in enumerate(('Root', 'Tip')):
            matrix = np.eye(4); matrix[:3, 3] = (0, i*0.4, 1)
            bind[name] = matrix
            rows.append(dict(joint_name=name, joint_parent='Root' if i else '', simulate=bool(i),
                             allow_rotation=bool(i), extrude_radius=0.01, extrude_sides=2, end_effector=0,
                             mass=1, goal_strength=0, goal_damping=0, gravity_z=0, drag=0,
                             stretch_spring=1, bend_spring=0, torsion_spring=0, extra_iterations=2,
                             collision_radius=0.01))
        node = dict(chain=dict(joints=rows, attrs={}))
        disabled = ChainSolver(node, bind, 1)
        for row in rows: row['vertex_map'] = collision.VERTEX_MAP
        enabled = ChainSolver(node, bind, 1)
        capsule = np.array([[0, 0.4, 0.8, 0, 0.4, 1.2, 0.1]])
        before = disabled.position.copy()
        for _ in range(10):
            disabled.advance(bind, 1/60, colliders=capsule)
            enabled.advance(bind, 1/60, colliders=capsule)
        self.assertTrue(np.allclose(disabled.position, before))
        self.assertTrue(np.array_equal(enabled.position[enabled.fixed], before[enabled.fixed]))
        delta = enabled.position[enabled.colliding] - np.array((0, 0.4, 1))
        self.assertTrue((np.linalg.norm(delta[:, :2], axis=1) >= 0.11-1e-8).all())
        # Even an infeasible collision cannot override a user-authored hard bound.
        for row in rows: row.update(stray_radius=0.05, stray_radius_stretchiness=0)
        limited = ChainSolver(node, bind, 1)
        for _ in range(10): limited.advance(bind, 1/60, colliders=capsule)
        self.assertTrue((np.linalg.norm(limited.position-limited.targets(bind), axis=1) <= 0.05+1e-8).all())


if __name__ == '__main__':
    unittest.main()
