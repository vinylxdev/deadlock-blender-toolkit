"""Reversible viewport-only physics; never persist simulated pose in a build."""
import time

import bpy
from mathutils import Matrix

from .physics import KEY, export_chains, physics_rig
from .physics_solver import ChainSolver

_session = None


def active(rig=None):
    return _session is not None and (rig is None or _session.rig == rig)


class Preview:
    def __init__(self, context, rig):
        from . import UNIT
        if rig.mode == 'EDIT':
            raise ValueError('Leave Edit Mode before previewing physics')
        if rig.name not in context.view_layer.objects:
            raise ValueError('The physics rig must be in the current view layer')
        nodes = export_chains(rig, {bone.name: bone.name for bone in rig.data.bones})
        if not nodes:
            raise ValueError('Add a physics chain before previewing physics')
        self.rig, self.scene, self.view_layer = rig, context.scene, context.view_layer
        self.config = rig.get(KEY)
        self.names = [row['joint_name'] for node in nodes for row in node['chain']['joints']]
        if any(constraint for name in self.names for constraint in rig.pose.bones[name].constraints
               if not constraint.mute):
            raise ValueError('Remove accessory bone constraints on the preview copy before simulating')
        self.basis = {name: rig.pose.bones[name].matrix_basis.copy() for name in self.names}
        self.pose_position = rig.data.pose_position
        self.frame = self.scene.frame_current_final
        self.clock = time.monotonic()
        self.busy = False
        bind = {bone.name: rig.matrix_world @ bone.matrix_local for bone in rig.data.bones}
        self.solvers = [ChainSolver(node, bind, UNIT) for node in nodes]
        self.order = [bone.name for bone in sorted(rig.data.bones, key=lambda b: len(b.parent_recursive))
                      if bone.name in self.basis]

    def restore(self):
        for name, matrix in self.basis.items():
            bone = self.rig.pose.bones.get(name)
            if bone:
                bone.matrix_basis = matrix

    def check(self):
        if (bpy.context.scene != self.scene or self.rig.mode == 'EDIT' or
                self.rig.get(KEY) != self.config or
                any(name not in self.rig.pose.bones for name in self.names) or
                self.rig.name not in self.view_layer.objects):
            raise ValueError('Physics preview stopped because the scene, rig or chains changed')

    def goals(self, depsgraph):
        rig = self.rig.evaluated_get(depsgraph)
        return {name: rig.matrix_world @ rig.pose.bones[name].matrix for name in self.names}

    def apply(self, goals, seconds):
        matrices = {}
        for solver in self.solvers:
            solver.advance(goals, seconds)
            matrices.update(solver.matrices(goals))
        inverse = self.rig.matrix_world.inverted()
        # Parent-before-child conversion avoids reading stale pose matrices
        # between assignments; convert_local_to_pose respects inherit scale.
        pose = {}
        for name in self.order:
            bone = self.rig.pose.bones[name]
            matrix = inverse @ Matrix(matrices[name].tolist())
            parent = bone.parent
            kwargs = {} if parent is None else dict(
                parent_matrix=pose.get(parent.name, parent.matrix),
                parent_matrix_local=parent.bone.matrix_local)
            bone.matrix_basis = bone.bone.convert_local_to_pose(
                matrix, bone.bone.matrix_local, invert=True, **kwargs)
            pose[name] = matrix

    def tick(self, seconds=None, depsgraph=None):
        if self.busy:
            return
        self.busy = True
        try:
            self.check()
            if depsgraph is None:
                self.restore()
                self.view_layer.update()
                depsgraph = bpy.context.evaluated_depsgraph_get()
            else:
                # frame_change_pre restored channels; animation has now been
                # evaluated for this frame. Preserve these current base values.
                self.basis = {name: self.rig.pose.bones[name].matrix_basis.copy() for name in self.names}
            goals = self.goals(depsgraph)
            now = time.monotonic()
            if seconds is None:
                seconds = now - self.clock
            self.clock = now
            self.rig.data.pose_position = 'POSE'
            self.apply(goals, seconds)
            self.view_layer.update()
        finally:
            self.busy = False


def _failure(error):
    scene = _session.scene if _session else None
    stop()
    if scene and hasattr(scene, 'deadlock_export'):
        scene.deadlock_export.status = f'Physics preview stopped: {error}'


def _timer():
    if _session is None:
        return None
    try:
        # Playback uses scene time in frame_change_post, never wall time.
        playing = any(window.screen.is_animation_playing for window in bpy.context.window_manager.windows)
        if not playing:
            _session.tick()
        else:
            _session.clock = time.monotonic()
    except Exception as error:
        _failure(error)
        return None
    return 1 / 60


def _frame_pre(scene, *_):
    if _session and _session.scene == scene and not _session.busy:
        try:
            _session.restore()
        except Exception as error:
            _failure(error)


def _frame_post(scene, depsgraph):
    if _session and _session.scene == scene and not _session.busy:
        try:
            frame = scene.frame_current_final
            delta = frame - _session.frame
            _session.frame = frame
            fps = scene.render.fps / scene.render.fps_base
            _session.tick(delta / fps, depsgraph)
        except Exception as error:
            _failure(error)


def _stop_event(*_):
    stop()


def _handlers():
    return [(bpy.app.handlers.frame_change_pre, _frame_pre),
            (bpy.app.handlers.frame_change_post, _frame_post)] + [
                (getattr(bpy.app.handlers, name), _stop_event)
                for name in ('save_pre', 'load_pre', 'undo_pre', 'redo_pre', 'render_init')]


def stop():
    """Restore channels and remove every callback, including during reload."""
    global _session
    session, _session = _session, None
    if bpy.app.timers.is_registered(_timer):
        bpy.app.timers.unregister(_timer)
    for handlers, callback in _handlers():
        if callback in handlers:
            handlers.remove(callback)
    if session:
        try:
            session.restore()
            session.rig.data.pose_position = session.pose_position
            session.view_layer.update()
        except (ReferenceError, RuntimeError):
            pass  # A deleted object or a closing file has nothing to restore.


def start(context, rig):
    global _session
    stop()
    session = Preview(context, rig)
    _session = session
    try:
        session.rig.data.pose_position = 'POSE'
        session.view_layer.update()
        goals = session.goals(context.evaluated_depsgraph_get())
        for solver in session.solvers:
            solver.reset(goals)
        for handlers, callback in _handlers():
            handlers.append(callback)
        bpy.app.timers.register(_timer, first_interval=1 / 60)
    except Exception:
        stop()
        raise


def toggle(context, settings):
    if active():
        stop()
        return 'Physics preview stopped; animation pose restored'
    rig = physics_rig(context, settings)
    if rig is None:
        raise ValueError('Create Retarget Original Rig Preview first')
    start(context, rig)
    return 'Physics preview running (approximate Source 2); move the rig or play the timeline'


def reset(context):
    if _session is None:
        raise ValueError('Start the physics preview before resetting it')
    start(context, _session.rig)
    return 'Physics preview reset to the animation pose'
