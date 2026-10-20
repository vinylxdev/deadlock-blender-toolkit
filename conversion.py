"""Automatic humanoid skin conversion on disposable copies of the source model."""

import re
from collections import defaultdict

import bpy
from mathutils import Matrix, Vector


from .bone_mapping import normalized, semantic_mapping


def model_frame(pelvis, head, left, right):
    up = (head - pelvis).normalized()
    across = left - right
    across = (across - up * across.dot(up)).normalized()
    if up.length < 0.5 or across.length < 0.5:
        raise ValueError("Degenerate humanoid landmarks")
    forward = across.cross(up).normalized()
    matrix = Matrix((across, forward, up)).transposed().to_4x4()
    matrix.translation = pelvis
    return matrix


def joint_frame(direction, across):
    """Construct an anatomical orientation without relying on bone-roll conventions."""
    along = direction.normalized()
    side = (across - along * across.dot(along)).normalized()
    if along.length < 0.5 or side.length < 0.5:
        raise ValueError('Degenerate hand/head orientation')
    return Matrix((side, along, side.cross(along))).transposed().to_4x4()


def deform_matrices(rig, target_matrices, mapping, preserve):
    from . import UNIT
    source_heads = {b.name: rig.matrix_world @ b.head_local for b in rig.data.bones}
    reverse = {target: source for source, target in mapping.items()}
    source_frame = model_frame(*(source_heads[reverse[n]] for n in ("pelvis", "head", "arm_upper_L", "arm_upper_R")))
    target_frame = model_frame(*(target_matrices[n].translation for n in ("pelvis", "head", "arm_upper_L", "arm_upper_R")))
    source_height = (source_heads[reverse["head"]] - source_heads[reverse["pelvis"]]).length
    target_height = (target_matrices["head"].translation - target_matrices["pelvis"].translation).length
    if preserve:
        source_feet = (source_heads[reverse['ankle_L']] + source_heads[reverse['ankle_R']]) * 0.5
        target_feet = (target_matrices['ankle_L'].translation + target_matrices['ankle_R'].translation) * 0.5
        source_height = (source_heads[reverse['head']] - source_feet).length
        target_height = (target_matrices['head'].translation - target_feet).length
        source_frame.translation = source_feet
        target_frame.translation = target_feet
    scale = target_height / source_height
    fit = target_frame @ Matrix.Scale(scale, 4) @ source_frame.inverted()
    # Animation pivots must coincide even when preserving the body silhouette.
    # Palm orientation also needs its transverse axis: a single direction leaves
    # hand roll unconstrained, and the hero's finger rotations twist the surface.
    hand_rotations = {}
    for side in ('L', 'R'):
        names = [f'hand_{side}', f'finger_middle_0_{side}', f'finger_index_0_{side}', f'finger_pinky_0_{side}']
        if all(name in reverse for name in names):
            src = [source_heads[reverse[name]] for name in names]
            dst = [target_matrices[name].translation for name in names]
            hand_rotations[f'hand_{side}'] = joint_frame(dst[1]-dst[0], dst[2]-dst[3]) @ joint_frame(src[1]-src[0], src[2]-src[3]).inverted()
    children = defaultdict(list)
    for source in mapping:
        parent = rig.data.bones[source].parent
        while parent and parent.name not in mapping:
            parent = parent.parent
        children[parent.name if parent else None].append(source)
    result, positions = {}, {}
    queue = [(source, None) for source in children[None]]
    while queue:
        source, parent = queue.pop(0)
        target = mapping[source]
        base = result[parent] if parent else fit
        precision_joint = target in {'neck_0', 'head', 'hand_L', 'hand_R'} or target.startswith('finger_')
        pivot = base @ source_heads[source] if preserve and not precision_joint else target_matrices[target].translation
        descendants = children[source]
        # Choose an anatomical continuation, rather than a finger or accessory branch.
        priority = ("spine_0", "spine_1", "spine_3", "neck_0", "head", "arm_upper", "arm_lower", "hand",
                    "leg_lower", "ankle", "ball", "finger_middle_0")
        child = min(descendants, key=lambda n: next((i for i, term in enumerate(priority)
                                                    if mapping[n].startswith(term)), 100)) if descendants else None
        if target in hand_rotations:
            rotation = hand_rotations[target]
            result[source] = Matrix.Translation(pivot) @ rotation @ Matrix.Scale(scale, 4) @ Matrix.Translation(-source_heads[source])
            positions[source] = pivot
            queue.extend((child, source) for child in descendants)
            continue
        if target == 'head':
            source_axis = rig.matrix_world.to_3x3() @ (rig.data.bones[source].tail_local - rig.data.bones[source].head_local)
            target_axis = target_matrices[target].to_3x3() @ Vector((1, 0, 0))
            rotation = joint_frame(target_axis, target_frame.col[0].to_3d()) @ joint_frame(source_axis, source_frame.col[0].to_3d()).inverted()
            result[source] = Matrix.Translation(pivot) @ rotation @ Matrix.Scale(scale, 4) @ Matrix.Translation(-source_heads[source])
            positions[source] = pivot
            queue.extend((child, source) for child in descendants)
            continue
        if child or target.startswith('finger_'):
            source_endpoint = source_heads[child] if child else rig.matrix_world @ rig.data.bones[source].tail_local
            endpoint_name = target.rsplit('_', 2)[0] + '_end_' + target.rsplit('_', 1)[-1]
            target_endpoint = target_matrices[mapping[child]].translation if child else (
                target_matrices[endpoint_name].translation if endpoint_name in target_matrices else
                target_matrices[target].translation + target_matrices[target].to_3x3() @ Vector((UNIT, 0, 0)))
            source_direction = (base.to_3x3() @ (source_endpoint - source_heads[source])).normalized()
            target_direction = (target_endpoint - target_matrices[target].translation).normalized()
            rotation = source_direction.rotation_difference(target_direction).to_matrix().to_4x4() if (
                source_direction.length > 0.5 and target_direction.length > 0.5) else Matrix.Identity(4)
            stretch = 1.0
            if not preserve or target.startswith('finger_'):
                original_length = (base.to_3x3() @ (source_endpoint - source_heads[source])).length
                target_length = (target_endpoint - target_matrices[target].translation).length
                stretch = target_length / original_length if original_length > 1e-6 else 1.0
            # Preserve uses rotations and one uniform scale; fit can rescale individual segments.
            delta = rotation @ Matrix.Scale(stretch, 4)
        else:
            delta = Matrix.Identity(4)
        result[source] = Matrix.Translation(pivot) @ delta @ Matrix.Translation(-(base @ source_heads[source])) @ base
        positions[source] = pivot
        queue.extend((child, source) for child in descendants)
    # Avatar Chest/UpperChest can be adjacent while the hero has an intervening
    # spine segment. Blending their independently rebased transforms compresses
    # the pectorals. Preserve the upper torso as one rigid, uniformly scaled unit.
    if preserve and 'spine_1' in reverse and 'spine_3' in reverse:
        chest, upper_chest = reverse['spine_1'], reverse['spine_3']
        parent = rig.data.bones[upper_chest].parent
        if chest not in target_matrices and upper_chest not in target_matrices and parent and parent.name == chest:
            mapping[chest] = mapping[upper_chest]
            result[chest] = result[upper_chest].copy()
            positions[chest] = result[chest] @ source_heads[chest]
    offsets = {source: (positions[source] - target_matrices[target].translation).length
               for source, target in mapping.items()}
    return result, scale, offsets


def neck_transition(rig, mapping, warp, target_matrices):
    """Blend the collar into one head transform and skin it around the actual pivots."""
    reverse = {target: source for source, target in mapping.items()}
    if not all(name in reverse for name in ('pelvis', 'neck_0', 'head', 'clavicle_L', 'clavicle_R')):
        return None
    heads = {name: rig.matrix_world @ rig.data.bones[source].head_local for name,source in reverse.items()}
    up = (heads['head'] - heads['pelvis']).normalized()
    center = heads['neck_0']
    shoulder = (heads['clavicle_L'] + heads['clavicle_R']) * 0.5
    lower = (shoulder-center).dot(up)
    upper = 0.0
    head_height = (heads['head']-center).dot(up)
    radius = (heads['clavicle_L']-heads['clavicle_R']).length
    target_neck = target_matrices['neck_0'].translation
    target_axis = target_matrices['head'].translation-target_neck
    if upper-lower < 1e-6 or radius < 1e-6 or target_axis.length_squared < 1e-8:
        return None
    def smooth(value):
        value = max(0.0, min(1.0, value))
        return value*value*(3.0-2.0*value)
    def correction(world, position, eligibility=1.0):
        height = (world-center).dot(up)
        radial = ((world-center)-up*height).length
        mask = smooth((height-lower)/(upper-lower)) * (1.0-smooth((radial/radius-1.2)/0.8))
        mask *= 1.0-smooth((height-head_height)/max(head_height, 0.01))
        mask *= eligibility
        if mask <= 1e-8:
            return position, 0.0, 0.0
        # Keep the source neck's cross-section and surface detail instead of
        # blending divergent chest/neck/head transforms at every weighted vertex.
        corrected = position.lerp(warp[reverse['head']] @ world, mask)
        head_weight = max(0.0, min(1.0, (corrected-target_neck).dot(target_axis)/target_axis.length_squared))
        return corrected, mask, head_weight
    return correction


def create_converted(settings):
    from . import source_objects, reference_bones, bone_matrices, selected_hero, object_context, UNIT
    rig, meshes = source_objects(settings)
    bones = reference_bones(settings)
    target_matrices = bone_matrices(bones)
    mapping = semantic_mapping(rig.data.bones.keys(), target_matrices.keys())
    warp, scale, offsets = deform_matrices(rig, target_matrices, mapping, settings.proportions == "PRESERVE")
    correct_neck = neck_transition(rig, mapping, warp, target_matrices)
    resolved, collapsed = {}, []
    for bone in rig.data.bones:
        anchor = bone
        while anchor and anchor.name not in mapping:
            anchor = anchor.parent
        if anchor is None:
            raise ValueError(f"Source bone {bone.name} has no recognized humanoid ancestor")
        resolved[bone.name] = (mapping[anchor.name], anchor.name)
        if anchor != bone:
            collapsed.append(bone.name)
    _, hero_name, model = selected_hero(settings)
    collection = bpy.data.collections.new(f"Deadlock Converted - {hero_name}")
    collection["deadlock_converted"] = True
    collection["deadlock_hero"] = settings.hero
    bpy.context.scene.collection.children.link(collection)
    created, blocks = [], []
    unweighted = 0
    neck_vertices = 0
    with object_context():
        old_pose = rig.data.pose_position
        try:
            rig.data.pose_position = "REST"
            bpy.context.view_layer.update()
            depsgraph = bpy.context.evaluated_depsgraph_get()
            target = bpy.data.objects.new(f"Deadlock Toolkit {hero_name} Rig", bpy.data.armatures.new(f"{hero_name} Skeleton"))
            collection.objects.link(target)
            created.append(target)
            blocks.append(target.data)
            target.show_in_front = False
            target["deadlock_model"] = model
            for selected in bpy.context.selected_objects:
                selected.select_set(False)
            target.select_set(True)
            bpy.context.view_layer.objects.active = target
            bpy.ops.object.mode_set(mode="EDIT")
            for bone in bones:
                edit = target.data.edit_bones.new(bone.name)
                edit.tail = (0, UNIT, 0)
                edit.matrix = target_matrices[bone.name]
            for bone in bones:
                if bone.parent:
                    target.data.edit_bones[bone.name].parent = target.data.edit_bones[bone.parent]
            bpy.ops.object.mode_set(mode="OBJECT")
            for original in meshes:
                copy = original.copy()
                copy.data = bpy.data.meshes.new_from_object(original.evaluated_get(depsgraph),
                                                          preserve_all_data_layers=True, depsgraph=depsgraph)
                collection.objects.link(copy)
                created.append(copy)
                blocks.append(copy.data)
                copy.animation_data_clear()
                for modifier in list(copy.modifiers):
                    copy.modifiers.remove(modifier)
                for constraint in list(copy.constraints):
                    copy.constraints.remove(constraint)
                groups = {g.index: resolved[g.name] for g in original.vertex_groups if g.name in resolved}
                neck_mesh = correct_neck and any(groups[g.group][0] in {'neck_0', 'spine_3'} and g.weight > 0.01
                                                for vertex in copy.data.vertices for g in vertex.groups if g.group in groups)
                mapped_weights = []
                for vertex in copy.data.vertices:
                    weights = [(groups[g.group], g.weight) for g in vertex.groups if g.group in groups and g.weight > 1e-8]
                    if not weights:
                        weights = [(("pelvis", next(src for src, tgt in mapping.items() if tgt == "pelvis")), 1.0)]
                        unweighted += 1
                    world = original.matrix_world @ vertex.co
                    total = sum(weight for _, weight in weights)
                    vertex.co = sum(((warp[anchor] @ world) * (weight / total) for (_, anchor), weight in weights), Vector((0, 0, 0)))
                    combined = defaultdict(float)
                    for (name, _), weight in weights:
                        combined[name] += weight
                    if neck_mesh:
                        eligibility = sum(weight for (name,_),weight in weights if name.startswith('spine_') or name in {'neck_0','head'}) / total
                        # A collar-height mask alone reaches the upper pectorals.
                        # Respect the avatar's painted neck/head membership so
                        # torso-only vertices cannot acquire neck deformation.
                        neck_support = sum(weight for (name,_),weight in weights if name in {'neck_0','head'}) / total
                        support = max(0.0, min(1.0, (neck_support-0.02)/0.08))
                        eligibility *= support*support*(3.0-2.0*support)
                        corrected, mask, head_weight = correct_neck(world, vertex.co, eligibility)
                        if mask > 1e-8:
                            vertex.co = corrected
                            for name in combined:
                                combined[name] = combined[name]/total*(1.0-mask)
                            combined['neck_0'] += mask*(1.0-head_weight)
                            combined['head'] += mask*head_weight
                            neck_vertices += 1
                    strongest = sorted(combined.items(), key=lambda item: item[1], reverse=True)[:4]
                    total = sum(weight for _, weight in strongest)
                    mapped_weights.append([(name, weight / total) for name, weight in strongest])
                copy.vertex_groups.clear()
                groups = {name: copy.vertex_groups.new(name=name) for name in sorted({n for rows in mapped_weights for n, _ in rows})}
                for index, rows in enumerate(mapped_weights):
                    for name, weight in rows:
                        groups[name].add([index], weight, "REPLACE")
                copy.parent = target
                copy.matrix_parent_inverse = Matrix.Identity(4)
                copy.matrix_basis = Matrix.Identity(4)
                copy.hide_viewport = False
                copy.hide_select = False
                copy.hide_set(False)
                modifier = copy.modifiers.new("Deadlock Skin", "ARMATURE")
                modifier.object = target
                if copy.data.has_custom_normals:
                    copy.data.normals_split_custom_set([(0, 0, 0)] * len(copy.data.loops))
                copy.data.update()
            report = {"hero": hero_name, "model": model, "proportions": settings.proportions,
                      "uniform_scale": scale, "source_bones": len(rig.data.bones), "target_bones": len(bones),
                      "direct_bone_mapping": mapping, "accessory_bones_collapsed_to_ancestors": collapsed,
                      "unweighted_vertices_assigned_to_pelvis": unweighted,
                      "joint_offset_meters": offsets, "mesh_count": len(meshes),
                      "precision_joint_alignment": "Hands, fingers, neck and head use the hero's animation pivots; palm roll is aligned and finger lengths fit the hero rig.",
                      "neck_transition_vertices": neck_vertices,
                      "note": "Avatar silhouette is preserved approximately. Different hero proportions can require weight adjustment. Hair, clothing and unmapped accessory bones follow mapped ancestors; avatar physics and facial morph animation are not generated."}
            return collection, report
        except Exception:
            if bpy.context.object and bpy.context.object.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            for obj in reversed(created):
                bpy.data.objects.remove(obj, do_unlink=True)
            for block in reversed(blocks):
                (bpy.data.meshes if isinstance(block, bpy.types.Mesh) else bpy.data.armatures).remove(block)
            bpy.data.collections.remove(collection)
            raise
        finally:
            rig.data.pose_position = old_pose
            bpy.context.view_layer.update()
