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
        self.names = [row['joint_name'] for row in self.rows]
        self.name_indices = {name: index for index, name in enumerate(self.names)}
        self.owner_indices = np.array([self.name_indices[name] for name in owners])
        bind_frames = np.array([self.bind[name] for name in self.names])
        self.bind_rotations = bind_frames[:, :3, :3] / np.linalg.norm(bind_frames[:, :3, :3], axis=1)[:, None, :]
        self.position = np.array(points)
        self.velocity = np.zeros_like(self.position)
        self.weights = np.array(weights)
        self.fixed = self.weights == 0
        settings = {row['joint_name']: row for row in self.rows}
        self.strength = np.array([settings[name]['goal_strength'] for name in owners])[:, None]
        self.damping = np.array([settings[name]['goal_damping'] for name in owners])[:, None]
        self.gravity = np.array([settings[name]['gravity_z'] for name in owners])[:, None]
        self.drag = np.array([settings[name]['drag'] for name in owners])[:, None]
        # Authored settings are immutable for this solver's lifetime. Editing a
        # chain recreates it, so these fixed-step coefficients need computing once.
        self.spring = 3600 * self.strength / np.maximum(0.001, 1 - self.strength)
        self.spring_damping = 2 * np.sqrt(self.spring) * self.damping / np.maximum(0.05, 1 - self.damping)
        self.air_drag = np.power(1 - self.drag, 60 * self.STEP)
        self.denominator = 1 + self.STEP * self.spring_damping + self.STEP * self.STEP * self.spring
        self.limit_radius = np.array([settings[name].get('stray_radius', 0.0) * unit for name in owners])[:, None]
        self.limit_strength = np.array([1.0 - settings[name].get('stray_radius_stretchiness', 0.0)
                                        for name in owners])[:, None]
        self.limited = np.flatnonzero((self.limit_radius[:, 0] > 0) &
                                     (self.limit_strength[:, 0] > 0) & ~self.fixed)
        self.colliding = np.array([i for i, name in enumerate(owners)
                                   if settings[name].get('vertex_map') == 'deadlock_auto_body_collision'
                                   and settings[name]['simulate'] and not self.fixed[i]], dtype=int)
        self.collision_thickness = np.array([settings[owners[i]].get('collision_radius', 0) * unit
                                             for i in self.colliding])
        self.capsules = np.empty((0, 7))
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
        self.groups = []
        for _, entries in groups:
            group = np.array(entries)
            a, b = group[:, 0].astype(int), group[:, 1].astype(int)
            self.groups.append((a, b, group[:, 2], group[:, 3], group[:, 4].astype(bool),
                                self.weights[a] + self.weights[b], self.weights[a, None], self.weights[b, None]))
        # The fitting topology and bind offsets never change during preview.
        for frame in self.frames:
            if frame['tip']:
                continue
            neighbors = [other for other in self.frames if other['parent'] is frame]
            if frame['parent']:
                neighbors.append(frame['parent'])
            indices = list(frame['indices']) + [other['indices'][0] for other in neighbors]
            frame['fit_indices'] = np.array(indices)
            reference = self.position[indices]
            frame['fit_reference'] = reference - reference[0]
        self.goals = self.position.copy()
        self.accumulator = 0.0

    def targets(self, matrices):
        frames = np.asarray([matrices[name] for name in self.names], dtype=float)
        rotations = frames[:, :3, :3] / np.linalg.norm(frames[:, :3, :3], axis=1)[:, None, :]
        return (frames[self.owner_indices, :3, 3] +
                (rotations[self.owner_indices] @ self.offsets[:, :, None])[:, :, 0])

    def reset(self, matrices):
        self.goals = self.targets(matrices)
        self.position = self.goals.copy()
        self.velocity.fill(0)
        self.accumulator = 0.0

    def advance(self, matrices, seconds, *, colliders=()):
        self.capsules = np.asarray(colliders, dtype=float).reshape(-1, 7)
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
        # Also enforce hard bounds against the displayed animation pose when
        # a frame ends between simulation steps. Soft limits stay on fixed steps.
        self._project_limits(target, hard_only=True)
        if len(self.capsules) and len(self.colliding):
            self._project_collisions()
            self._project_limits(target, hard_only=True)
        if not np.isfinite(self.position).all():
            raise ValueError('Physics preview became unstable; reset the preview')

    def _step(self, goal):
        dt = self.STEP
        previous = self.position.copy()
        # Implicit goal spring: damping affects attraction, not free motion when
        # strength is zero. Air drag remains independent, as in ModelDoc.
        self.velocity *= self.air_drag
        acceleration = np.zeros_like(self.position)
        acceleration[:, 2:3] = -self.GRAVITY * self.gravity
        self.velocity = (self.velocity + dt * (acceleration + self.spring * (goal - self.position))) / self.denominator
        self.position += dt * self.velocity
        self.position[self.fixed] = goal[self.fixed]
        for _ in range(self.iterations):
            for a, b, rest, strength, unilateral, total, weight_a, weight_b in self.groups:
                delta = self.position[b] - self.position[a]
                length = np.linalg.norm(delta, axis=1)
                error = length - rest
                # Source bend/torsion springs bound maximum bind distance;
                # they do not force a curved accessory to straighten out.
                error = np.where(unilateral, np.maximum(0, error), error)
                correction = delta * (error * strength / (np.maximum(length, 1e-12) * total))[:, None]
                self.position[a] += correction * weight_a
                self.position[b] -= correction * weight_b
            self._project_limits(goal)
            if len(self.capsules) and len(self.colliding):
                self._project_collisions()
                self._project_limits(goal, hard_only=True)
        self.velocity = (self.position - previous) / dt
        self.velocity[self.fixed] = 0

    def _project_collisions(self):
        from .physics_collision import project_capsules
        project_capsules(self.position, self.colliding, self.collision_thickness, self.capsules)

    def _project_limits(self, goal, *, hard_only=False):
        indices = self.limited
        if hard_only:
            indices = indices[self.limit_strength[indices, 0] >= 1.0]
        if not len(indices):
            return
        delta = self.position[indices] - goal[indices]
        distance = np.linalg.norm(delta, axis=1, keepdims=True)
        excess = np.maximum(0.0, 1.0 - self.limit_radius[indices] / np.maximum(distance, 1e-12))
        self.position[indices] -= delta * excess * self.limit_strength[indices]

    def matrices(self, goals):
        """Fit bone orientation to support points and neighboring joint centers."""
        goal_frames = np.asarray([goals[name] for name in self.names], dtype=float)
        scales = np.linalg.norm(goal_frames[:, :3, :3], axis=1)
        goal_rotations = goal_frames[:, :3, :3] / scales[:, None, :]
        result, orientations = {}, {}
        for frame in self.frames:
            if frame['tip']:
                continue
            row, indices = frame['row'], frame['fit_indices']
            name = row['joint_name']
            index = self.name_indices[name]
            if not row['simulate'] and not row['allow_rotation']:
                result[name] = goal_frames[index].copy()
                orientations[name] = goal_rotations[index]
                continue
            actual = self.position[indices]
            actual = actual - actual[0]
            u, _, vt = np.linalg.svd(frame['fit_reference'].T @ actual)
            rotation = vt.T @ np.diag((1, 1, np.linalg.det(vt.T @ u.T))) @ u.T
            matrix = self.bind[name].copy()
            orientation = rotation @ self.bind_rotations[index]
            if self.lock_twist:
                reference = goal_rotations[index]
                parent = row['joint_parent']
                if parent in result:
                    reference = orientations[parent] @ goal_rotations[self.name_indices[parent]].T @ reference
                orientation = swing_only(reference, orientation)
            orientations[name] = orientation
            matrix[:3, :3] = orientation * scales[index]
            matrix[:3, 3] = self.position[indices[0]]
            result[name] = matrix
        return result
