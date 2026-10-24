"""Particle approximation of exported ClothChains, in meters and world space.

Topology follows ModelDoc's documented extrusion and ancestor springs. The
projection/integration below is our preview solver, not Valve's runtime solver.
No runtime parity is implied by sharing the authored ClothChain settings.
"""
import math

import numpy as np


def swing_only(reference, simulated):
    """Aim local +Y with the shortest bend, preserving the reference's twist."""
    direction = reference.T @ simulated[:, 1]
    sine = np.linalg.norm(direction[[0, 2]])
    cosine = np.clip(direction[1], -1, 1)
    if sine < 1e-12:
        # A reversed direction has no unique bend axis. Local X gives a
        # deterministic half-turn with no Y twist and no division by zero.
        bend = np.eye(3) if cosine >= 0 else np.diag((1, -1, -1))
    else:
        x, z = direction[2] / sine, -direction[0] / sine
        cross = np.array(((0, -z, 0), (z, 0, -x), (0, x, 0)))
        bend = np.eye(3) + sine * cross + (1 - cosine) * cross @ cross
    return reference @ bend


class ChainSolver:
    STEP = 1 / 120
    GRAVITY = 9.80665

    def __init__(self, node, bind, unit, *, lock_twist=False):
        self.lock_twist = lock_twist
        self.rows = node['chain']['joints']
        defaults = {key: attr['default'] for key, attr in node['chain']['attrs'].items()}
        self.rows = [dict(defaults, **row) for row in self.rows]
        self.bind = {name: np.array(matrix, dtype=float) for name, matrix in bind.items()}
        self.frames = []
        points, weights, owners, offsets = [], [], [], []
        edges = []
        by_name = {}

        def frame(row, parent, tip=False):
            name = row['joint_name']
            matrix = self.bind[name]
            rotation = matrix[:3, :3] / np.linalg.norm(matrix[:3, :3], axis=0)
            origin = np.array((0, row['end_effector'] * unit if tip else 0, 0))
            radius = row['extrude_radius'] * unit
            local = [origin] + [origin + radius * np.array((math.cos(angle), 0, math.sin(angle)))
                                for angle in np.linspace(0, 2 * math.pi, row['extrude_sides'], endpoint=False)]
            indices = []
            for i, offset in enumerate(local):
                indices.append(len(points))
                points.append(matrix[:3, 3] + rotation @ offset)
                # An unlocked static joint pins its center but may rotate.
                fixed = not tip and not row['simulate'] and (i == 0 or not row['allow_rotation'])
                weights.append(0.0 if fixed else 1.0 / row['mass'])
                owners.append(name)
                offsets.append(offset)
            result = dict(row=row, indices=indices, parent=parent, tip=tip)
            self.frames.append(result)
            # Rigid cross section, with independent motion between sections.
            for i, a in enumerate(indices):
                for b in indices[i + 1:]:
                    edges.append((a, b, 1.0, False))
            ancestor = parent
            for key in ('stretch_spring', 'bend_spring', 'torsion_spring'):
                if ancestor is None:
                    break
                for a, b in zip(indices, ancestor['indices']):
                    edges.append((a, b, row[key], key != 'stretch_spring'))
                ancestor = ancestor['parent']
            return result

        for row in self.rows:
            by_name[row['joint_name']] = frame(row, by_name.get(row['joint_parent']))
        for row in self.rows:
            if row['end_effector']:
                frame(row, by_name[row['joint_name']], tip=True)
        self.offsets = np.array(offsets)
        self.owners = owners
        self.position = np.array(points)
        self.velocity = np.zeros_like(self.position)
        self.weights = np.array(weights)
        self.fixed = self.weights == 0
        settings = {row['joint_name']: row for row in self.rows}
        self.strength = np.array([settings[name]['goal_strength'] for name in owners])[:, None]
        self.damping = np.array([settings[name]['goal_damping'] for name in owners])[:, None]
        self.gravity = np.array([settings[name]['gravity_z'] for name in owners])[:, None]
        self.drag = np.array([settings[name]['drag'] for name in owners])[:, None]
        self.iterations = 1 + max(row['extra_iterations'] for row in self.rows)
        # Color the graph so vectorized projection never writes an endpoint twice.
        groups = []
        for a, b, strength, unilateral in edges:
            if strength <= 0 or self.weights[a] + self.weights[b] == 0:
                continue
            for occupied, group in groups:
                if a not in occupied and b not in occupied:
                    break
            else:
                occupied, group = set(), []
                groups.append((occupied, group))
            occupied.update((a, b))
            group.append((a, b, np.linalg.norm(self.position[b] - self.position[a]), strength, unilateral))
        self.groups = [np.array(group) for _, group in groups]
        self.goals = self.position.copy()
        self.accumulator = 0.0

    def targets(self, matrices):
        result = []
        for owner, offset in zip(self.owners, self.offsets):
            matrix = np.asarray(matrices[owner])
            rotation = matrix[:3, :3] / np.linalg.norm(matrix[:3, :3], axis=0)
            result.append(matrix[:3, 3] + rotation @ offset)
        return np.array(result)

    def reset(self, matrices):
        self.goals = self.targets(matrices)
        self.position = self.goals.copy()
        self.velocity.fill(0)
        self.accumulator = 0.0

    def advance(self, matrices, seconds):
        target = self.targets(matrices)
        if seconds <= 0 or seconds > 0.25:
            self.reset(matrices)
            return
        elapsed = self.accumulator + seconds
        count = int((elapsed + 1e-10) / self.STEP)
        # Interpolate moving animation goals across fixed simulation steps.
        for step in range(count):
            fraction = min(1.0, ((step + 1) * self.STEP - self.accumulator) / seconds)
            goal = self.goals + (target - self.goals) * fraction
            self._step(goal)
        self.accumulator = max(0.0, elapsed - count * self.STEP)
        self.goals = target
        self.position[self.fixed] = target[self.fixed]
        if not np.isfinite(self.position).all():
            raise ValueError('Physics preview became unstable; reset the preview')

    def _step(self, goal):
        dt = self.STEP
        previous = self.position.copy()
        # Implicit goal spring: damping affects attraction, not free motion when
        # strength is zero. Air drag remains independent, as in ModelDoc.
        spring = 3600 * self.strength / np.maximum(0.001, 1 - self.strength)
        damping = 2 * np.sqrt(spring) * self.damping / np.maximum(0.05, 1 - self.damping)
        self.velocity *= np.power(1 - self.drag, 60 * dt)
        acceleration = np.zeros_like(self.position)
        acceleration[:, 2:3] = -self.GRAVITY * self.gravity
        self.velocity = (self.velocity + dt * (acceleration + spring * (goal - self.position))) / (
            1 + dt * damping + dt * dt * spring)
        self.position += dt * self.velocity
        self.position[self.fixed] = goal[self.fixed]
        for _ in range(self.iterations):
            for group in self.groups:
                a, b = group[:, 0].astype(int), group[:, 1].astype(int)
                delta = self.position[b] - self.position[a]
                length = np.linalg.norm(delta, axis=1)
                error = length - group[:, 2]
                # Source bend/torsion springs bound maximum bind distance;
                # they do not force a curved accessory to straighten out.
                error = np.where(group[:, 4] != 0, np.maximum(0, error), error)
                total = self.weights[a] + self.weights[b]
                correction = delta * (error * group[:, 3] / (np.maximum(length, 1e-12) * total))[:, None]
                self.position[a] += correction * self.weights[a, None]
                self.position[b] -= correction * self.weights[b, None]
        self.velocity = (self.position - previous) / dt
        self.velocity[self.fixed] = 0

    def matrices(self, goals):
        """Fit bone orientation to support points and neighboring joint centers."""
        result = {}
        for frame in self.frames:
            if frame['tip']:
                continue
            row, indices = frame['row'], list(frame['indices'])
            name = row['joint_name']
            if not row['simulate'] and not row['allow_rotation']:
                result[name] = np.array(goals[name])
                continue
            neighbors = [f for f in self.frames if f['parent'] is frame]
            if frame['parent']:
                neighbors.append(frame['parent'])
            indices.extend(f['indices'][0] for f in neighbors)
            reference = np.array([self.bind[self.owners[i]][:3, 3] +
                                  (self.bind[self.owners[i]][:3, :3] /
                                   np.linalg.norm(self.bind[self.owners[i]][:3, :3], axis=0)) @ self.offsets[i]
                                  for i in indices])
            actual = self.position[indices]
            reference -= reference[0]
            actual = actual - actual[0]
            u, _, vt = np.linalg.svd(reference.T @ actual)
            rotation = vt.T @ np.diag((1, 1, np.linalg.det(vt.T @ u.T))) @ u.T
            matrix = self.bind[name].copy()
            scale = np.linalg.norm(np.asarray(goals[name])[:3, :3], axis=0)
            orientation = rotation @ (matrix[:3, :3] / np.linalg.norm(matrix[:3, :3], axis=0))
            if self.lock_twist:
                reference = np.asarray(goals[name])[:3, :3] / scale
                parent = row['joint_parent']
                if parent in result:
                    parent_goal = np.asarray(goals[parent])[:3, :3]
                    parent_goal = parent_goal / np.linalg.norm(parent_goal, axis=0)
                    parent_pose = result[parent][:3, :3]
                    parent_pose = parent_pose / np.linalg.norm(parent_pose, axis=0)
                    reference = parent_pose @ parent_goal.T @ reference
                orientation = swing_only(reference, orientation)
            matrix[:3, :3] = orientation * scale
            matrix[:3, 3] = self.position[indices[0]]
            result[name] = matrix
        return result
