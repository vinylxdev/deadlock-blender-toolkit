"""Portable preset validation and distance-based curve placement, without Blender."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location('physics_curve', Path(__file__).parents[1] / 'physics_curve.py')
curve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curve)


class PhysicsCurve(unittest.TestCase):
    def test_invalid_presets_are_rejected_before_blender_is_needed(self):
        valid = curve.default_curve()
        self.assertEqual(curve.validated_curve(valid), valid)
        invalid = [None, [], {}, dict(valid, enabled=1), dict(valid, extend='UNKNOWN'), dict(valid, points=[]),
                   dict(valid, points=[[0, 1, 'AUTO']])]
        for point in ([True, 1, 'AUTO'], [0, float('nan'), 'AUTO'], [0, float('inf'), 'AUTO'],
                      [-0.1, 1, 'AUTO'], [0, 1.1, 'AUTO'], [0, 1, 'UNKNOWN'], [0, 1]):
            changed = copy.deepcopy(valid)
            changed['points'][0] = point
            invalid.append(changed)
        invalid += [dict(valid, points=[[0.5, 1, 'AUTO'], [0.5, 0, 'AUTO']]),
                    dict(valid, points=[[1, 1, 'AUTO'], [0, 0, 'AUTO']])]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                curve.validated_curve(value)

    def test_distance_and_branch_tips_determine_curve_position(self):
        class Point:
            def __init__(self, x): self.x = x
            def __sub__(self, other): return SimpleNamespace(length=abs(self.x - other.x))
        class World:
            def __matmul__(self, point): return Point(3 * point.x + 10)
        def bone(name, x, parent=None):
            result = SimpleNamespace(name=name, head_local=Point(x), parent=parent, children=[])
            if parent: parent.children.append(result)
            return result
        root = bone('root', 0)
        shared = bone('shared', 1, root)
        short = bone('short', 2, shared)
        long = bone('long', 3, shared)
        tip = bone('tip', 4, long)
        # Input ordering is irrelevant, and shorter branches still reach 1.
        positions = curve.chain_positions([root, tip, long, short, shared], World())
        self.assertEqual(positions, dict(root=0, shared=0.25, short=1, long=0.75, tip=1))
        self.assertEqual(curve.chain_positions([root], World()), {'root': 0})


if __name__ == '__main__':
    unittest.main()
