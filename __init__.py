"""Blender UI and export pipeline for prepared Deadlock hero replacements."""

bl_info = {
    "name": "Deadlock Blender Toolkit",
    "author": "Vinyl",
    "version": (0, 3, 38),
    "blender": (5, 2, 0),
    "location": "3D View > Sidebar > Deadlock Toolkit",
    "description": "Choose a hero, inspect its rig, export FBX, compile and package a prepared replacement",
    "category": "Import-Export",
}

import importlib
import json
import os
import re
import shutil
import subprocess
import time
import zlib
from contextlib import contextmanager
from pathlib import Path

import bpy
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper
from mathutils import Matrix, Quaternion, Vector

from .core import VPK, hero_catalog, hero_labels, pack_vpk, patch_modeldoc, resource_path
from .physics_ui import CURVE_ITEMS

UNIT = 0.0254  # Source 2 inch -> Blender meter, as used by SourceIO.
_catalog = []  # Keep strings alive for Blender's dynamic EnumProperty.
_archive_cache = None


def local_path(value):
    if not value.strip():
        raise ValueError("A required file or directory path is empty")
    return Path(bpy.path.abspath(value)).expanduser().resolve()


def sourceio():
    try:
        utils = importlib.import_module("SourceIO.library.utils")
        resources = importlib.import_module("SourceIO.library.source2.compiled_resource")
        models = importlib.import_module("SourceIO.library.source2.resource_types.compiled_model_resource")
        return utils, resources, models
    except ImportError as error:
        raise ValueError("Enable the SourceIO add-on to read Deadlock's hero data and skeletons") from error


def archive(settings):
    global _archive_cache
    path = local_path(settings.game_dir) / "pak01_dir.vpk"
    stamp = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
    if _archive_cache is None or _archive_cache[0] != stamp:
        _archive_cache = (stamp, VPK(path))
    return _archive_cache[1]


def read_resource(data, path, model=False):
    utils, resources, models = sourceio()
    cls = models.CompiledModelResource if model else resources.CompiledResource
    return cls.from_buffer(utils.MemoryBuffer(data), utils.TinyPath(path))


def selected_hero(settings):
    for key, name, model in _catalog:
        if key == settings.hero:
            return key, name, model
    raise ValueError("Refresh the hero list and choose a hero")


def hero_items(self, context):
    return [(key, name, model, zlib.crc32(key.encode()) & 0x7FFFFFFF) for key, name, model in _catalog] or [
        ("NONE", "Refresh heroes first", "Choose the installed Deadlock game/citadel folder", 0)
    ]


def load_catalog(settings):
    global _catalog
    vpk = archive(settings)
    path = "scripts/heroes.vdata_c"
    metadata = read_resource(vpk.read(path), path).get_block(None, block_name="DATA")
    labels_path = local_path(settings.game_dir) / "resource/localization/citadel_gc_hero_names/citadel_gc_hero_names_english.txt"
    labels = hero_labels(labels_path.read_bytes()) if labels_path.is_file() else {}
    _catalog = hero_catalog(metadata, vpk.entries, labels)
    if not _catalog:
        raise ValueError("No available heroes found in this installation")
    if settings.hero not in {key for key, _, _ in _catalog}:
        settings.hero = _catalog[0][0]
    return _catalog


def reference_bones(settings):
    _, _, path = selected_hero(settings)
    resource = read_resource(archive(settings).read(path), path, model=True)
    return resource.get_bones()


def bone_matrices(bones):
    result = {}
    pending = list(bones)
    while pending:
        progress = False
        for bone in pending[:]:
            if bone.parent and bone.parent not in result:
                continue
            local = Matrix.Translation(Vector(bone.pos) * UNIT) @ Quaternion(bone.rot).to_matrix().to_4x4()
            result[bone.name] = result[bone.parent] @ local if bone.parent else local
            pending.remove(bone)
            progress = True
        if not progress:
            raise ValueError("Unsupported or cyclic hero bone hierarchy")
    return result


def source_objects(settings):
    collection = settings.collection
    if collection is None:
        raise ValueError("Choose the collection containing your replacement model")
    objects = list(collection.all_objects)
    meshes = [obj for obj in objects if obj.type == "MESH" and not obj.hide_render]
    if not meshes:
        raise ValueError("The export collection has no render-enabled mesh objects")
    rigs = {obj.find_armature() for obj in meshes}
    if None in rigs or len(rigs) != 1:
        raise ValueError("All exported meshes must be bound to the same armature")
    rig = next(iter(rigs))
    if any(obj.name not in bpy.context.view_layer.objects for obj in meshes + [rig]):
        raise ValueError("The source meshes and armature must be included in the active view layer")
    return rig, meshes


def material_report(meshes):
    from bpy_extras.node_shader_utils import PrincipledBSDFWrapper
    materials = {m.name: m for mesh in meshes for m in mesh.data.materials if m}
    missing, complex_materials = [], []
    channels = ("base_color_texture", "normalmap_texture", "roughness_texture",
                "metallic_texture", "alpha_texture", "emission_color_texture")
    for name, material in sorted(materials.items()):
        wrapper = PrincipledBSDFWrapper(material, is_readonly=True)
        for channel in channels:
            texture = getattr(wrapper, channel, None)
            if texture and texture.image:
                image = texture.image
                if not image.packed_file and not Path(bpy.path.abspath(image.filepath)).is_file():
                    missing.append({"material": name, "channel": channel, "image": image.name,
                                    "path": image.filepath})
        if material.node_tree and any(n.type in {"MIX", "MIX_RGB", "GROUP"} for n in material.node_tree.nodes):
            complex_materials.append(name)
    return {"missing_texture_references": missing, "materials_needing_baking_or_manual_conversion": complex_materials,
            "note": "FBX exports supported Principled channels. Source 2 materials must be authored separately; arbitrary Blender shader graphs are not converted."}


def check_rig(settings):
    rig, meshes = source_objects(settings)
    bones = reference_bones(settings)
    target_matrices = bone_matrices(bones)
    missing, parents, transforms = [], [], []
    for target in bones:
        current = rig.data.bones.get(target.name)
        if current is None:
            missing.append(target.name)
            continue
        if (current.parent.name if current.parent else None) != target.parent:
            parents.append(target.name)
        actual = rig.matrix_world @ current.matrix_local
        expected = target_matrices[target.name]
        if max(abs(actual[row][col] - expected[row][col]) for row in range(4) for col in range(4)) > 0.001:
            transforms.append(target.name)
    unweighted = {}
    for mesh in meshes:
        groups = {g.index for g in mesh.vertex_groups if g.name in rig.data.bones}
        count = sum(not any(g.group in groups and g.weight > 0 for g in v.groups) for v in mesh.data.vertices)
        if count:
            unweighted[mesh.name] = count
    return {"hero": selected_hero(settings)[1], "target_model": selected_hero(settings)[2],
            "source_rig": rig.name, "target_bones": len(bones), "missing_bones": missing,
            "parent_mismatches": parents, "rest_transform_mismatches": transforms,
            "unweighted_vertices": unweighted,
            "materials": material_report(meshes),
            "compatible": not (missing or parents or transforms or unweighted),
            "note": "Skeleton compatibility is checked; visual deformation, scale, materials and gameplay still require an in-game test."}


@contextmanager
def object_context():
    """Restore the user's selection, active object and mode after export helpers."""
    context = bpy.context
    selected = list(context.selected_objects)
    active = context.view_layer.objects.active
    mode = active.mode if active else "OBJECT"
    if active and active.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    try:
        yield
    finally:
        if context.object and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        for obj in list(context.selected_objects):
            obj.select_set(False)
        for obj in selected:
            if obj.name in context.view_layer.objects:
                obj.select_set(True)
        context.view_layer.objects.active = active
        if active and mode != "OBJECT":
            bpy.ops.object.mode_set(mode=mode)


def export_fbx(settings, output, source2=False):
    """Bake the current shape-key appearance and modifiers on disposable copies."""
    from .physics_preview import stop
    stop()
    rig, meshes = source_objects(settings)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    objects, data = [], []
    with object_context():
        previous_pose = rig.data.pose_position
        try:
            rig.data.pose_position = "REST"
            bpy.context.view_layer.update()
            depsgraph = bpy.context.evaluated_depsgraph_get()
            rig_copy = rig.copy()
            rig_copy.data = rig.data.copy()
            rig_copy.animation_data_clear()
            rig_copy.parent = None
            rig_copy.matrix_world = rig.matrix_world.copy()
            bpy.context.scene.collection.objects.link(rig_copy)
            objects.append(rig_copy)
            data.append(rig_copy.data)
            for constraint in list(rig_copy.constraints):
                rig_copy.constraints.remove(constraint)
            for bone in rig_copy.pose.bones:
                for constraint in list(bone.constraints):
                    bone.constraints.remove(constraint)
                bone.matrix_basis = Matrix.Identity(4)
            for mesh in meshes:
                copy = mesh.copy()
                copy.data = bpy.data.meshes.new_from_object(mesh.evaluated_get(depsgraph),
                                                           preserve_all_data_layers=True, depsgraph=depsgraph)
                copy.animation_data_clear()
                for modifier in list(copy.modifiers):
                    copy.modifiers.remove(modifier)
                for constraint in list(copy.constraints):
                    copy.constraints.remove(constraint)
                copy.parent = rig_copy
                copy.matrix_world = mesh.matrix_world.copy()
                modifier = copy.modifiers.new("Export Rig", "ARMATURE")
                modifier.object = rig_copy
                bpy.context.scene.collection.objects.link(copy)
                objects.append(copy)
                data.append(copy.data)
                if source2:
                    # The compiler drops tiny absolute FBX weights. Blender
                    # normalizes these when deforming; export the same ratios
                    # so hair with weights near 1e-20 keeps its authored bone.
                    groups={g.index:g for g in copy.vertex_groups if g.name in rig_copy.data.bones}
                    for vertex in copy.data.vertices:
                        influences=[(groups[g.group],g.weight) for g in vertex.groups
                                    if g.group in groups and g.weight>0]
                        total=sum(w for _,w in influences)
                        if total<=0:raise ValueError(f'{mesh.name}: vertex {vertex.index} has no bone weights')
                        if abs(total-1)>1e-6:
                            for group,weight in influences:group.add([vertex.index],weight/total,'REPLACE')
            for obj in bpy.context.selected_objects:
                obj.select_set(False)
            for obj in objects:
                obj.hide_viewport = False
                obj.hide_set(False)
                obj.hide_select = False
                obj.select_set(True)
            bpy.context.view_layer.objects.active = rig_copy
            result = bpy.ops.export_scene.fbx(
                filepath=str(output), check_existing=False, use_selection=True,
                object_types={"MESH", "ARMATURE"}, add_leaf_bones=False,
                use_armature_deform_only=False, bake_anim=False, use_mesh_modifiers=False,
                # Source 2 rebases the imported mesh onto the authored skeleton.
                # Bake inch coordinates into node transforms; FBX unit metadata
                # alone leaves skin offsets in meters and collapses the surface.
                mesh_smooth_type="FACE", use_triangles=True,
                global_scale=1.0 / (100.0 * UNIT * bpy.context.scene.unit_settings.scale_length) if source2 else 1.0,
                apply_unit_scale=True, apply_scale_options="FBX_SCALE_NONE" if source2 else "FBX_SCALE_UNITS",
                axis_forward="-Y", axis_up="Z", path_mode="COPY", embed_textures=True,
            )
            if result != {"FINISHED"}:
                raise ValueError("Blender's FBX exporter did not finish")
        finally:
            rig.data.pose_position = previous_pose
            for obj in reversed(objects):
                bpy.data.objects.remove(obj, do_unlink=True)
            for block in reversed(data):
                if isinstance(block, bpy.types.Mesh):
                    bpy.data.meshes.remove(block)
                else:
                    bpy.data.armatures.remove(block)
            bpy.context.view_layer.update()


class DL_Material(bpy.types.PropertyGroup):
    material: PointerProperty(type=bpy.types.Material)
    resource: StringProperty(name="Source 2 material", description="Existing game or authored .vmat resource path, such as materials/deadlock_toolkit/body.vmat")


class DL_Settings(bpy.types.PropertyGroup):
    game_dir: StringProperty(name="Deadlock game/citadel", subtype="DIR_PATH")
    hero: EnumProperty(name="Replace hero", items=hero_items)
    collection: PointerProperty(name="Model collection", type=bpy.types.Collection)
    output_dir: StringProperty(name="Export folder", subtype="DIR_PATH")
    compiler: StringProperty(name="Resource compiler", subtype="FILE_PATH")
    content_dir: StringProperty(name="SDK content folder", subtype="DIR_PATH")
    sdk_game_dir: StringProperty(name="SDK game folder", subtype="DIR_PATH")
    template: StringProperty(name="Hero ModelDoc template", subtype="FILE_PATH")
    sdk_root: StringProperty(name="Reduced CSDK folder", subtype="DIR_PATH")
    viewer_cli: StringProperty(name="Source 2 Viewer CLI", subtype="FILE_PATH")
    proportions: EnumProperty(name="Proportions", items=[
        ("PRESERVE", "Preserve avatar", "Uniform scale and pose alignment; preserve avatar limb lengths as closely as possible"),
        ("FIT", "Fit to hero", "Also fit individual body segments to the hero proportions"),
    ], default="PRESERVE")
    texture_size: IntProperty(name="Texture resolution", default=1024, min=256, max=4096)
    keep_native_gun: BoolProperty(name='Keep native gun',default=True,
                                 description='Include the hero’s separate gun or weapon mesh in automatic builds; turn off to hide it while keeping animation bones and attachments')
    install: BoolProperty(name='Install into Deadlock',default=False,
                         description='After a successful build, place the VPK in an exporter-owned addons slot without replacing another mod')
    bindings: CollectionProperty(type=DL_Material)
    status: StringProperty(default="Refresh heroes to begin")


def report_error(operator, settings, error):
    settings.status = str(error)
    operator.report({"ERROR"}, str(error))
    return {"CANCELLED"}


class DL_OT_refresh(bpy.types.Operator):
    bl_idname = "deadlock.refresh_heroes"
    bl_label = "Refresh Heroes"

    def execute(self, context):
        settings = context.scene.deadlock_export
        try:
            heroes = load_catalog(settings)
            settings.status = f"{len(heroes)} heroes loaded from Deadlock"
            return {"FINISHED"}
        except Exception as error:
            return report_error(self, settings, error)


class DL_OT_reference(bpy.types.Operator):
    bl_idname = "deadlock.load_reference_rig"
    bl_label = "Load Hero Reference Skeleton"
    bl_description = "Add the exact hero skeleton in a separate collection; this does not rebind your model"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.deadlock_export
        obj = None
        collection = None
        try:
            key, name, model = selected_hero(settings)
            bones = reference_bones(settings)
            matrices = bone_matrices(bones)
            with object_context():
                collection = bpy.data.collections.new(f"Deadlock Reference - {name}")
                context.scene.collection.children.link(collection)
                obj = bpy.data.objects.new(f"{name} Reference Rig", bpy.data.armatures.new(f"{name} Skeleton"))
                collection.objects.link(obj)
                obj.show_in_front = True
                obj["deadlock_model"] = model
                for selected in context.selected_objects:
                    selected.select_set(False)
                obj.select_set(True)
                context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode="EDIT")
                for bone in bones:
                    edit = obj.data.edit_bones.new(bone.name)
                    edit.tail = (0, UNIT, 0)
                    edit.matrix = matrices[bone.name]
                for bone in bones:
                    if bone.parent:
                        obj.data.edit_bones[bone.name].parent = obj.data.edit_bones[bone.parent]
                bpy.ops.object.mode_set(mode="OBJECT")
            settings.status = f"Loaded {len(bones)} bones for {name}; bind a model copy to this rig"
            return {"FINISHED"}
        except Exception as error:
            if obj is not None:
                if obj.mode != "OBJECT":
                    bpy.ops.object.mode_set(mode="OBJECT")
                armature = obj.data
                bpy.data.objects.remove(obj, do_unlink=True)
                bpy.data.armatures.remove(armature)
            if collection is not None:
                bpy.data.collections.remove(collection)
            return report_error(self, settings, error)


class DL_OT_check(bpy.types.Operator):
    bl_idname = "deadlock.check_rig"
    bl_label = "Check Rig Compatibility"

    def execute(self, context):
        settings = context.scene.deadlock_export
        try:
            report = check_rig(settings)
            text = bpy.data.texts.get("Deadlock Rig Report") or bpy.data.texts.new("Deadlock Rig Report")
            text.clear()
            text.write(json.dumps(report, indent=2))
            settings.status = ("Skeleton matches; review materials and test in game" if report["compatible"] else
                               f"Rig needs preparation: {len(report['missing_bones'])} missing bones; see Deadlock Rig Report")
            self.report({"INFO"}, settings.status)
            return {"FINISHED"}
        except Exception as error:
            return report_error(self, settings, error)


class OutputDirectorySelector:
    directory: StringProperty(name='Output folder', subtype='DIR_PATH', options={'SKIP_SAVE'})
    filter_folder: BoolProperty(default=True, options={'HIDDEN'})

    def invoke(self, context, event):
        settings = context.scene.deadlock_export
        try:
            default = settings.output_dir if settings.output_dir.strip() else str(Path.home() / 'DeadlockExports')
            self.directory = str(local_path(default)) + os.sep
            context.window_manager.fileselect_add(self)
            return {'RUNNING_MODAL'}
        except Exception as error:
            return report_error(self, settings, error)

    def draw(self, context):
        self.layout.label(text="Files are grouped in the selected hero's subfolder")

    def apply_output_directory(self, settings):
        directory = local_path(self.directory or settings.output_dir)
        if directory.exists() and not directory.is_dir():
            raise ValueError('Choose an output folder, rather than a file')
        settings.output_dir = str(directory)


class DL_OT_export(bpy.types.Operator, ExportHelper, OutputDirectorySelector):
    bl_idname = "deadlock.export_source"
    bl_label = "Export Source FBX"
    bl_description = "Export the current model appearance and native rig; this is an authoring file, not a playable mod"
    filename_ext = '.fbx'
    filter_glob: StringProperty(default='*.fbx', options={'HIDDEN'})

    def invoke(self, context, event):
        settings = context.scene.deadlock_export
        try:
            if not self.filepath:
                default = settings.output_dir if settings.output_dir.strip() else str(Path.home() / 'DeadlockExports')
                self.filepath = str(local_path(default) / 'deadlock_toolkit.fbx')
            return ExportHelper.invoke(self, context, event)
        except Exception as error:
            return report_error(self, settings, error)

    def draw(self, context):
        self.layout.label(text='Writes the FBX and a matching JSON report')

    def execute(self, context):
        settings = context.scene.deadlock_export
        try:
            key, name, model = selected_hero(settings)
            if self.filepath:
                output = local_path(bpy.path.ensure_ext(self.filepath, self.filename_ext))
                settings.output_dir = str(output.parent)
            else:
                self.apply_output_directory(settings)
                output = local_path(settings.output_dir) / key / "deadlock_toolkit.fbx"
            export_fbx(settings, output)
            _, meshes = source_objects(settings)
            materials = material_report(meshes)
            manifest = {"hero": name, "hero_key": key, "replacement_path": model,
                        "blend_file": bpy.data.filepath, "fbx": str(output),
                        "shape_keys": "Current appearance baked; animation and inactive facial morphs excluded",
                        "materials": materials,
                        "ready_for_game": False}
            output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            missing = len(materials["missing_texture_references"])
            settings.status = f"Exported {output}; {missing} missing texture references. See the adjacent JSON report."
            self.report({"WARNING"} if missing else {"INFO"}, f"Source FBX exported for {name}; {missing} missing texture references")
            return {"FINISHED"}
        except Exception as error:
            return report_error(self, settings, error)


class DL_OT_materials(bpy.types.Operator):
    bl_idname = "deadlock.sync_materials"
    bl_label = "List Model Materials"

    def execute(self, context):
        settings = context.scene.deadlock_export
        try:
            _, meshes = source_objects(settings)
            previous = {row.material.name: row.resource for row in settings.bindings if row.material}
            used = {m.name: m for obj in meshes for m in obj.data.materials if m}
            settings.bindings.clear()
            for name, material in sorted(used.items()):
                row = settings.bindings.add()
                row.material = material
                row.resource = previous.get(name, "")
            settings.status = "Assign a Source 2 .vmat resource to each Blender material before building"
            return {"FINISHED"}
        except Exception as error:
            return report_error(self, settings, error)


class DL_OT_retarget_preview(bpy.types.Operator):
    bl_idname = 'deadlock.retarget_original_preview'
    bl_label = 'Retarget Original Rig Preview'
    bl_description = 'Create a comparison copy that retains the original bones, weights and proportions; preview the hero idle pose without mesh corrections'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings=context.scene.deadlock_export
        try:
            from .retarget import create_retarget_preview
            collection,report=create_retarget_preview(settings)
            text=bpy.data.texts.get('Deadlock Retarget Report') or bpy.data.texts.new('Deadlock Retarget Report')
            text.clear();text.write(json.dumps(report,indent=2))
            settings.status=f'Original rig retarget in {collection.name}; inspect Preview Idle Pose, then use Build Retargeted VPK.'
            self.report({'INFO'},settings.status)
            return {'FINISHED'}
        except Exception as error:
            return report_error(self,settings,error)


class DL_OT_fit_retarget_legs(bpy.types.Operator):
    bl_idname = 'deadlock.fit_retarget_legs'
    bl_label = 'Fit Leg Proportions to Hero'
    bl_description = 'Fit the thighs and shins on the retarget copy to the hero contacts, preserving vertex weights and foot shape'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        from .physics import physics_rig
        rig=physics_rig(context,context.scene.deadlock_export)
        return rig is not None and not rig.get('deadlock_retarget_leg_fit') and not rig.get('deadlock_retarget_preserve_proportions')

    def execute(self, context):
        settings=context.scene.deadlock_export
        try:
            from .physics import physics_rig
            from .retarget_legs import fit_legs
            from .physics_preview import stop
            stop()
            rig=physics_rig(context,settings)
            if rig is None: raise ValueError('Create Retarget Original Rig Preview first')
            hidden=rig.hide_get()
            try:report=fit_legs(rig)
            finally:rig.hide_set(hidden)
            text=bpy.data.texts.get('Deadlock Leg Fit Report') or bpy.data.texts.new('Deadlock Leg Fit Report')
            text.clear();text.write(json.dumps(report,indent=2))
            settings.status='Leg proportions fitted on the retarget copy; original project data and vertex weights are preserved. Build Retargeted VPK to export.'
            return {'FINISHED'}
        except Exception as error: return report_error(self,settings,error)


class DL_OT_preserve_retarget_proportions(bpy.types.Operator):
    bl_idname = 'deadlock.preserve_retarget_proportions'
    bl_label = 'Preserve Proportions & Ground Pose'
    bl_description = 'Restore authored proportions, uniformly size the entire avatar to the hero leg reach, and ground the whole pose; this can make the avatar taller'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        from .physics import physics_rig
        return (not context.window_manager.get('deadlock_build_running',False)
                and physics_rig(context,context.scene.deadlock_export) is not None)

    def execute(self, context):
        settings=context.scene.deadlock_export
        try:
            from .physics import physics_rig
            from .retarget_placement import preserve_proportions
            from .physics_preview import stop
            stop()
            rig=physics_rig(context,settings)
            report=preserve_proportions(rig)
            text=bpy.data.texts.get('Deadlock Placement Report') or bpy.data.texts.new('Deadlock Placement Report')
            text.clear();text.write(json.dumps(report,indent=2))
            settings.status='Original proportions restored with uniform sizing and whole-pose ground placement. Build Retargeted VPK to export.'
            return {'FINISHED'}
        except Exception as error:return report_error(self,settings,error)


class DL_OT_fit_avatar_height(bpy.types.Operator):
    bl_idname = 'deadlock.fit_avatar_height'
    bl_label = 'Fit Height to Hero'
    bl_description = 'Uniformly scale the entire preview to match the base hero ground-to-eye height using rest eye bones, ignoring hats and hair above the eyes'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        from .physics import physics_rig
        rig = physics_rig(context, context.scene.deadlock_export)
        return (not context.window_manager.get('deadlock_build_running', False)
                and rig is not None and bool(rig.get('deadlock_retarget_mapping')))

    def execute(self, context):
        settings = context.scene.deadlock_export
        try:
            from .physics import physics_rig
            from .retarget_size import fit_hero_height
            height = fit_hero_height(physics_rig(context, settings), settings)
            settings.status = f'Avatar fitted to the base hero eye height ({height:.3f} m); Build Retargeted VPK to export.'
            return {'FINISHED'}
        except Exception as error:
            return report_error(self, settings, error)


class DL_OT_idle_preview(bpy.types.Operator):
    bl_idname = "deadlock.toggle_idle_preview"
    bl_label = "Preview Idle Pose"
    bl_description = "Toggle the native hero idle pose on a selected preview copy, or the latest copy; restore its previous pose when switched off"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        from .preview import preview_rig
        return not context.window_manager.get('deadlock_build_running', False) and preview_rig(context, context.scene.deadlock_export) is not None

    def execute(self, context):
        settings = context.scene.deadlock_export
        try:
            from .preview import toggle_idle
            settings.status = toggle_idle(context, settings)
            self.report({'INFO'}, settings.status)
            return {'FINISHED'}
        except Exception as error:
            return report_error(self, settings, error)


class DL_OT_physics_preview(bpy.types.Operator):
    bl_idname = 'deadlock.physics_preview'
    bl_label = 'Preview Physics'
    bl_description = 'Preview accessory physics using exported chain settings; approximate Source 2 motion, without body or world collisions. Stop to restore the animation pose'
    reset: BoolProperty(default=False, options={'HIDDEN', 'SKIP_SAVE'})

    @classmethod
    def poll(cls, context):
        from .physics import physics_rig, chains
        from .physics_preview import active
        return not context.window_manager.get('deadlock_build_running', False) and (
            active() or bool(chains(physics_rig(context, context.scene.deadlock_export))))

    def execute(self, context):
        from .physics_preview import toggle, reset
        settings = context.scene.deadlock_export
        try:
            settings.status = reset(context) if self.reset else toggle(context, settings)
            return {'FINISHED'}
        except Exception as error:
            return report_error(self, settings, error)


class DL_OT_physics_chain(bpy.types.Operator):
    bl_idname = 'deadlock.physics_chain'
    bl_label = 'Add Physics Chain'
    bl_description = 'Choose an accessory root; it stays attached while its descendants simulate in the exported VPK'
    bl_options = {'REGISTER', 'UNDO'}
    settings_tab: EnumProperty(name='Section', default='MOTION', options={'HIDDEN', 'SKIP_SAVE'},
                               items=(('MOTION', 'Motion', 'Movement and limits'),
                                      ('CURVES', 'Curves', 'Edit a profile along the chain'),
                                      ('ADVANCED', 'Advanced', 'Attachment and rotation stability')))
    curve_setting: EnumProperty(name='Setting', items=CURVE_ITEMS, default='stiffness',
                                options={'HIDDEN', 'SKIP_SAVE'})
    root: StringProperty(name='Root bone')
    stiffness: FloatProperty(name='Return to pose', default=0.5, min=0, max=1,
                             description='Animation attraction: higher values retain the original accessory shape')
    use_stiffness_curve: BoolProperty(name='Pose attraction curve', default=False,
                                     description='Scale Return to pose along each branch: left is the root, right is the tip; height is a multiplier from 0 to 1')
    stiffness_curve_name: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    stiffness_curve_data: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    damping: FloatProperty(name='Damping', default=0.5, min=0, max=1)
    use_damping_curve: BoolProperty(name='Damping curve', default=False,
                                   description='Multiply Damping by a curve from the base to the tip of each branch')
    damping_curve_name: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    damping_curve_data: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    gravity: FloatProperty(name='Gravity', default=0.3, min=0, max=2)
    use_gravity_curve: BoolProperty(name='Gravity curve', default=False,
                                   description='Multiply Gravity by a curve from the base to the tip of each branch')
    gravity_curve_name: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    gravity_curve_data: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    twist_resistance: FloatProperty(name='Twist resistance',default=0.2,min=0,max=1,
                                    description='Torsion spring strength on longer chains: higher values resist twisting; this is not a hard angle limit')
    use_twist_resistance_curve: BoolProperty(name='Twist resistance curve', default=False,
                                           description='Multiply the cloth torsion spring by a curve from base to tip; native jiggle exports do not use this spring')
    twist_resistance_curve_name: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    twist_resistance_curve_data: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    use_motion_limits: BoolProperty(name='Limit motion', default=False,
                                   description='Bound cloth motion around each bone’s animated pose')
    auto_collision: BoolProperty(name='Automatic collision fitting', default=False,
                                 description='Fit capsules to the weighted body mesh and make this chain collide with them in preview and game export; requires Lock twist off')
    motion_limit: FloatProperty(name='Motion radius', default=0.01, min=0, soft_max=1,
                               subtype='DISTANCE', unit='LENGTH', precision=4,
                               description='Maximum distance from the animated pose, stored in meters; zero disables the limit for that bone')
    use_motion_limit_curve: BoolProperty(name='Motion radius curve', default=False,
                                        description='Multiply the motion radius along each branch; a zero multiplier disables that bone’s limit')
    motion_limit_curve_name: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    motion_limit_curve_data: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    motion_limit_strength: FloatProperty(name='Limit strength', default=1, min=0, max=1,
                                        description='How firmly to enforce the radius: 1 is a hard limit, lower values allow some overshoot, and 0 disables it')
    use_motion_limit_strength_curve: BoolProperty(name='Limit strength curve', default=False,
                                                 description='Multiply the limit strength along each branch')
    motion_limit_strength_curve_name: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    motion_limit_strength_curve_data: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    lock_root_rotation: BoolProperty(name='Lock root rotation',default=True,
                                    description='Keep the pinned root’s animated orientation instead of letting simulation roll it; descendants still bend')
    lock_twist: BoolProperty(name='Lock twist (swing only)', default=False,
                            description='Allow bending without simulated roll around the bone’s length (local Y). Game export uses native pitch/yaw jiggle bones; spring response can differ from preview')
    anchor_child_roots: BoolProperty(name='Anchor child roots',default=False,
                                    description='Keep each direct child attached in its animated pose; useful when one hair root contains several strands. Lower bones still simulate')
    stable_rotation_frames: BoolProperty(name='Stable rotation frames',default=False,
                                         description='Use three support points around each bone instead of two to strengthen the simulated orientation frame; adds physics nodes')
    index: IntProperty(default=-1, options={'HIDDEN', 'SKIP_SAVE'})
    rig_name: StringProperty(options={'HIDDEN'})

    def invoke(self, context, event):
        from .physics import physics_rig, chains
        from .physics_curve import prepare_editors, clear_editors
        rig=physics_rig(context,context.scene.deadlock_export)
        if rig is None:
            self.report({'ERROR'},'Create Retarget Original Rig Preview first')
            return {'CANCELLED'}
        self.rig_name=rig.name
        entries=chains(rig)
        if self.index>=0:
            if self.index>=len(entries): return {'CANCELLED'}
            for key in ('root','stiffness','damping','gravity'): setattr(self,key,entries[self.index][key])
            self.twist_resistance=entries[self.index].get('twist_resistance',0.2)
            self.lock_root_rotation=entries[self.index].get('lock_root_rotation',False)
            self.lock_twist=entries[self.index].get('lock_twist',False)
            self.anchor_child_roots=entries[self.index].get('anchor_child_roots',False)
            self.stable_rotation_frames=entries[self.index].get('stable_rotation_frames',False)
        elif context.object and context.object.type=='ARMATURE' and context.object.data.bones.active:
            candidate=context.object.data.bones.active.name
            if candidate in rig.data.bones: self.root=candidate
        stored = entries[self.index] if self.index >= 0 else {}
        self.use_motion_limits = stored.get('use_motion_limits', False)
        self.auto_collision = stored.get('auto_collision', False)
        self.motion_limit = stored.get('motion_limit', 0.01)
        self.motion_limit_strength = stored.get('motion_limit_strength', 1.0)
        prepare_editors(self, entries[self.index] if self.index >= 0 else {})
        self.settings_tab = 'MOTION'
        self.curve_setting = 'stiffness'
        try:
            return context.window_manager.invoke_props_dialog(
                self, width=520, title='Edit Physics Chain' if self.index >= 0 else 'Add Physics Chain',
                confirm_text='Apply')
        except Exception:
            clear_editors(self)
            raise

    def draw(self, context):
        from .physics_ui import draw_dialog
        draw_dialog(self.layout, self, bpy.data.objects.get(self.rig_name))

    def check(self, context):
        # Tabs and optional controls change the dialog's visible layout.
        return True

    def execute(self, context):
        from .physics import physics_rig, chains, validate_chains, KEY
        from .physics_curve import store_curves, clear_editors
        settings=context.scene.deadlock_export
        try:
            rig=bpy.data.objects.get(self.rig_name) if self.rig_name else physics_rig(context,settings)
            if rig is None: raise ValueError('Create Retarget Original Rig Preview first')
            entries=chains(rig)
            entry={key:getattr(self,key) for key in ('root','stiffness','damping','gravity','twist_resistance','lock_root_rotation','anchor_child_roots','stable_rotation_frames','lock_twist')}
            entry.update(use_motion_limits=self.use_motion_limits, motion_limit=self.motion_limit,
                         motion_limit_strength=self.motion_limit_strength, auto_collision=self.auto_collision)
            store_curves(self, entry)
            index=self.index
            if index<0: index=next((i for i,e in enumerate(entries) if e['root']==self.root),-1)
            if index>=len(entries): raise ValueError('Physics chain no longer exists')
            if index>=0: entries[index]=entry
            else: entries.append(entry)
            validate_chains(rig,entries)
            from .physics_preview import stop
            stop()
            rig[KEY]=json.dumps(entries)
            settings.status=f'Physics chain {self.root} configured. Preview Physics to test; Build Retargeted VPK to apply.'
            return {'FINISHED'}
        except Exception as error: return report_error(self,settings,error)
        finally:
            clear_editors(self)

    def cancel(self, context):
        from .physics_curve import clear_editors
        clear_editors(self)


class DL_OT_physics_remove(bpy.types.Operator):
    bl_idname = 'deadlock.remove_physics_chain'
    bl_label = 'Remove Physics Chain'
    bl_options = {'REGISTER', 'UNDO'}
    index: IntProperty(default=-1, options={'HIDDEN'})

    def execute(self, context):
        from .physics import physics_rig, chains, KEY
        rig=physics_rig(context,context.scene.deadlock_export)
        entries=chains(rig)
        if rig is None or not 0<=self.index<len(entries): return {'CANCELLED'}
        from .physics_preview import stop
        stop()
        entries.pop(self.index)
        rig[KEY]=json.dumps(entries)
        return {'FINISHED'}


class DL_OT_physics_save(bpy.types.Operator, ExportHelper):
    bl_idname = 'deadlock.save_physics_preset'
    bl_label = 'Save Physics Preset'
    bl_description = 'Save chain root names and settings to a JSON file for another version of this avatar'
    filename_ext = '.json'
    filter_glob: StringProperty(default='*.json', options={'HIDDEN'})
    rig_name: StringProperty(options={'HIDDEN'})

    def invoke(self, context, event):
        from .physics import physics_rig
        settings=context.scene.deadlock_export
        rig=physics_rig(context,settings)
        if rig is None:
            return report_error(self,settings,ValueError('Create Retarget Original Rig Preview first'))
        self.rig_name=rig.name
        if not self.filepath:
            source=rig.users_collection[0].get('deadlock_source_blend',bpy.data.filepath)
            name=(Path(source).stem if source else 'avatar')+'.physics.json'
            self.filepath=str(local_path(settings.output_dir)/name)
        return ExportHelper.invoke(self,context,event)

    def execute(self, context):
        from .physics import physics_rig, save_preset
        settings=context.scene.deadlock_export
        try:
            rig=bpy.data.objects.get(self.rig_name) if self.rig_name else physics_rig(context,settings)
            if rig is None: raise ValueError('Create Retarget Original Rig Preview first')
            count=save_preset(rig,bpy.path.abspath(self.filepath))
            settings.status=f'Saved {count} physics chains to {self.filepath}'
            self.report({'INFO'},f'Saved {count} physics chains')
            return {'FINISHED'}
        except Exception as error: return report_error(self,settings,error)


class DL_OT_physics_load(bpy.types.Operator, ImportHelper):
    bl_idname = 'deadlock.load_physics_preset'
    bl_label = 'Load Physics Preset'
    bl_description = 'Replace this retarget rig\'s physics chains with a saved preset; all root bones must be compatible'
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = '.json'
    filter_glob: StringProperty(default='*.json', options={'HIDDEN'})
    rig_name: StringProperty(options={'HIDDEN'})

    def invoke(self, context, event):
        from .physics import physics_rig
        settings=context.scene.deadlock_export
        rig=physics_rig(context,settings)
        if rig is None:
            return report_error(self,settings,ValueError('Create Retarget Original Rig Preview first'))
        self.rig_name=rig.name
        if not self.filepath: self.filepath=str(local_path(settings.output_dir))+os.sep
        return ImportHelper.invoke(self,context,event)

    def execute(self, context):
        from .physics import physics_rig, load_preset
        settings=context.scene.deadlock_export
        try:
            rig=bpy.data.objects.get(self.rig_name) if self.rig_name else physics_rig(context,settings)
            if rig is None: raise ValueError('Create Retarget Original Rig Preview first')
            from .physics_preview import stop
            stop()
            count=load_preset(rig,bpy.path.abspath(self.filepath))
            settings.status=f'Loaded {count} physics chains from {Path(self.filepath).name}. Build Retargeted VPK to apply them.'
            self.report({'INFO'},f'Loaded {count} physics chains')
            return {'FINISHED'}
        except Exception as error: return report_error(self,settings,error)


class DL_OT_automatic(bpy.types.Operator, ExportHelper, OutputDirectorySelector):
    bl_idname = "deadlock.automatic_conversion"
    bl_label = "Build Retargeted VPK"
    bl_description = "Build the original-rig retarget preview, bake materials and compile its runtime constraints and physics into a replacement VPK"
    compile: BoolProperty(default=True, options={'HIDDEN'})
    retarget: BoolProperty(default=True, options={'HIDDEN'})
    filename_ext = '.vpk'
    filter_glob: StringProperty(default='*.vpk', options={'HIDDEN'})
    _process = None
    _timer = None
    _log = None

    @classmethod
    def poll(cls, context):
        return not context.window_manager.get('deadlock_build_running',False)

    def invoke(self, context, event):
        if not (self.compile and self.retarget):
            return OutputDirectorySelector.invoke(self, context, event)
        settings = context.scene.deadlock_export
        try:
            if not self.filepath:
                default = settings.output_dir if settings.output_dir.strip() else str(Path.home() / 'DeadlockExports')
                self.filepath = str(local_path(default) / f'deadlock_toolkit_retarget_{settings.hero}.vpk')
            return ExportHelper.invoke(self, context, event)
        except Exception as error:
            return report_error(self, settings, error)

    def draw(self, context):
        if self.compile and self.retarget:
            self.layout.label(text='Reports and logs go in a hero subfolder')
        else:
            OutputDirectorySelector.draw(self, context)

    def execute(self, context):
        settings=context.scene.deadlock_export
        try:
            vpk_path = None
            if self.compile and self.retarget and self.filepath:
                vpk_path = local_path(bpy.path.ensure_ext(self.filepath, self.filename_ext))
                settings.output_dir = str(vpk_path.parent)
            else:
                self.apply_output_directory(settings)
            from .conversion import semantic_mapping
            import uuid
            collection=settings.collection
            if self.retarget:
                candidates=[c for c in context.scene.collection.children_recursive
                            if c.get('deadlock_preview_only') and c.get('deadlock_hero')==settings.hero]
                selected=[c for c in candidates if context.object and context.object in c.all_objects[:]]
                collection=(selected or candidates)[-1] if (selected or candidates) else None
                if collection is None:
                    raise ValueError('Create Retarget Original Rig Preview for the selected hero first')
                from .physics import validate_chains
                validate_chains(next(o for o in collection.all_objects if o.type=='ARMATURE'))
            else:
                if collection and collection.get('deadlock_preview_only'):
                    raise ValueError('Use Build Retargeted VPK to export the original-rig retarget preview')
                rig,meshes=source_objects(settings)
                semantic_mapping(rig.data.bones.keys(),[bone.name for bone in reference_bones(settings)])
            sdk=local_path(settings.sdk_root)
            cli=local_path(settings.viewer_cli)
            if not cli.is_file():
                raise ValueError('Choose Source2Viewer-CLI in the automatic conversion tool settings')
            if not (sdk/'game/citadel/gameinfo.gi').is_file():
                raise ValueError('Choose the Reduced_CSDK_12 folder in the automatic conversion tool settings')
            if self.compile and not any((sdk/f'game/{folder}/win64/resourcecompiler.exe').is_file() for folder in ('bin_cs2','bin')):
                raise ValueError('The selected CSDK does not contain a model compiler')
            job=local_path(settings.output_dir)/'.runs'/uuid.uuid4().hex
            job.mkdir(parents=True)
            self._snapshot=job/'snapshot.blend'
            self._status_path=job/'status.json'
            config={key:getattr(settings,key) for key in ('game_dir','hero','output_dir','sdk_root','viewer_cli','proportions','texture_size','install','keep_native_gun')}
            config.update({'collection':collection.name,'source_blend':collection.get('deadlock_source_blend',bpy.data.filepath),
                           'status_path':str(self._status_path),'compile':self.compile,'retarget':self.retarget,
                           'scene_unit_scale':context.scene.unit_settings.scale_length})
            if vpk_path is not None:
                config['vpk_path'] = str(vpk_path)
            (job/'job.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
            settings.status='Saving an isolated snapshot for automatic conversion'
            from .physics_preview import stop
            stop()
            if self.retarget:
                bpy.data.libraries.write(str(self._snapshot),{collection},path_remap='ABSOLUTE',fake_user=True,compress=True)
            else:
                bpy.ops.wm.save_as_mainfile(filepath=str(self._snapshot),copy=True,check_existing=False)
            self._log_path=job/'worker.log'
            self._log=self._log_path.open('w',encoding='utf-8')
            worker=Path(__file__).with_name('worker.py')
            command=[bpy.app.binary_path,'--background','--disable-autoexec',str(self._snapshot),
                     '--python',str(worker),'--',str(job/'job.json')]
            environment=os.environ.copy()
            environment['ALSOFT_DRIVERS']='null'
            self._process=subprocess.Popen(command,stdout=self._log,stderr=subprocess.STDOUT,env=environment,
                                            start_new_session=os.name!='nt')
            self._timer=context.window_manager.event_timer_add(0.5,window=context.window)
            context.window_manager.modal_handler_add(self)
            context.window_manager['deadlock_build_running']=True
            settings.status='Automatic conversion started; Esc cancels. '+str(self._log_path)
            return {'RUNNING_MODAL'}
        except Exception as error:
            self.finish(context,False)
            return report_error(self,settings,error)

    def finish(self,context,success):
        if self._process and self._process.poll() is None and os.name!='nt':
            import signal
            os.killpg(self._process.pid,signal.SIGTERM)
        DL_OT_build.finish(self,context,success)
        if hasattr(self,'_snapshot') and self._snapshot.exists():
            self._snapshot.unlink()

    def modal(self,context,event):
        settings=context.scene.deadlock_export
        if event.type=='ESC':
            self.finish(context,False)
            settings.status='Automatic conversion cancelled; the source model is intact'
            return {'CANCELLED'}
        if event.type!='TIMER':
            return {'PASS_THROUGH'}
        state={}
        if self._status_path.is_file():
            try:
                state=json.loads(self._status_path.read_text(encoding='utf-8'))
                settings.status=state.get('stage','Converting')
            except (OSError,ValueError):
                pass
        if self._process.poll() is None:
            return {'PASS_THROUGH'}
        try:
            if self._process.returncode or state.get('state')!='complete':
                raise ValueError(state.get('stage',f'Conversion worker failed; see {self._log_path}'))
            if state.get('preview') and not self.retarget:
                with bpy.data.libraries.load(state['preview'],link=False) as (source,destination):
                    destination.collections=source.collections
                for collection in destination.collections:
                    if collection:
                        context.scene.collection.children.link(collection)
            if state.get('report'):
                text=bpy.data.texts.get('Deadlock Conversion Report') or bpy.data.texts.new('Deadlock Conversion Report')
                text.clear();text.write(Path(state['report']).read_text(encoding='utf-8'))
            self.finish(context,True)
            settings.status=(f"Built {state['vpk']}"+(f"; installed into {state['installed']}" if state.get('installed') else '') if state.get('vpk') else
                             'Rig, materials and ModelDoc converted; sources ready and preview added to the scene')
            self.report({'INFO'},settings.status)
            return {'FINISHED'}
        except Exception as error:
            self.finish(context,False)
            return report_error(self,settings,error)


def compiler_command(compiler, game_dir, model, *, environment=None, shallow=False):
    arguments = ["-i", str(model), "-game", str(game_dir), "-nop4"]
    arguments += ["-fshallow", "-noassert", "-dxlevel", "110"] if shallow else ["-f"]
    if os.name != "nt" and compiler.suffix.lower() == ".exe":
        wine = shutil.which("wine")
        winepath = shutil.which("winepath")
        if not wine or not winepath:
            raise ValueError("Windows Source 2 tools on Linux require Wine and winepath")
        for index in (1, 3):
            arguments[index] = subprocess.check_output([winepath, "-w", arguments[index]],
                                                       env=environment, text=True, timeout=30).strip()
        return [wine, str(compiler), *arguments]
    return [str(compiler), *arguments]


def collect_compiled(game_dir, main, replacement, retail, retail_staged=()):
    """Include the compiled dependency closure; leave mounted retail resources alone."""
    files, visited, pending = {replacement: main}, set(), [main]
    game_dir = game_dir.resolve()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        resource = read_resource(current.read_bytes(), current.relative_to(game_dir).as_posix())
        for dependency in resource.get_dependencies():
            name = resource_path(str(dependency))
            if name in retail_staged:
                continue
            source = (game_dir / name).resolve()
            if not source.is_relative_to(game_dir):
                raise ValueError("Compiled dependency escaped the SDK game folder")
            if source.is_file():
                files[name] = source
                pending.append(source)
            elif name not in retail.entries:
                raise ValueError(f"Missing compiled dependency: {name}. Compile or mount it in the SDK first.")
    # Keep the generated main path when a dependency references it, in addition to the hero override.
    return files


class DL_OT_build(bpy.types.Operator, OutputDirectorySelector):
    bl_idname = "deadlock.build_mod"
    bl_label = "Export and Build VPK"
    bl_description = "Requires a matching hero rig, prepared ModelDoc template, material bindings and Source 2 compiler"
    _process = None
    _timer = None
    _log = None

    @classmethod
    def poll(cls, context):
        return not context.window_manager.get("deadlock_build_running", False)

    def execute(self, context):
        settings = context.scene.deadlock_export
        try:
            self.apply_output_directory(settings)
            key, name, replacement = selected_hero(settings)
            compiler = local_path(settings.compiler)
            if not compiler.is_file():
                raise ValueError("Choose an installed Source 2 resourcecompiler executable")
            content = local_path(settings.content_dir)
            game_dir = local_path(settings.sdk_game_dir)
            if not (game_dir / "gameinfo.gi").is_file():
                raise ValueError("The SDK game folder must contain gameinfo.gi")
            if (content.parent.name != "content" or game_dir.parent.name != "game" or
                content.name != game_dir.name or content.parent.parent != game_dir.parent.parent):
                raise ValueError("Choose matching SDK content/<mod> and game/<mod> folders")
            template = local_path(settings.template)
            if not template.is_file() or template.suffix != ".vmdl":
                raise ValueError("Choose a prepared .vmdl template for the selected hero")
            # Template names must match the game's selected model to prevent accidental cross-hero builds.
            if template.name != Path(replacement).name.removesuffix("_c"):
                raise ValueError(f"The selected hero needs a {Path(replacement).name.removesuffix('_c')} template")
            report = check_rig(settings)
            if not report["compatible"]:
                raise ValueError("Rig does not match this hero. Use Check Rig Compatibility and prepare a bound model copy first.")
            rig, meshes = source_objects(settings)
            used = {m.name for obj in meshes for m in obj.data.materials if m}
            if any(not obj.data.materials or any(m is None for m in obj.data.materials) for obj in meshes):
                raise ValueError("Every exported mesh needs populated material slots")
            bindings = {row.material.name: resource_path(row.resource) for row in settings.bindings
                        if row.material and row.material.name in used}
            if used != set(bindings):
                raise ValueError("List Model Materials, then assign a .vmat path to every material")
            for path in bindings.values():
                if not (content / path).is_file() and path + "_c" not in archive(settings).entries:
                    raise ValueError(f"Material not found in SDK content or Deadlock: {path}")
            mesh_resource = f"models/deadlock_toolkit/{key}/deadlock_toolkit.fbx"
            model_resource = f"models/deadlock_toolkit/{key}/deadlock_toolkit.vmdl"
            model = content / model_resource
            if model.resolve() == template:
                raise ValueError("Keep the authored template separate from the generated model")
            patched = patch_modeldoc(template.read_text(encoding="utf-8-sig"), mesh_resource, bindings)
            output = local_path(settings.output_dir) / key
            output.mkdir(parents=True, exist_ok=True)
            self._compiled = game_dir / (model_resource + "_c")
            self._backup = self._compiled.with_suffix(self._compiled.suffix + ".previous")
            self._game_dir, self._replacement = game_dir, replacement
            self._retail = archive(settings)
            self._output = output / f"deadlock_toolkit_{key}.vpk"
            command = compiler_command(compiler, game_dir, model)
            export_fbx(settings, content / mesh_resource, source2=True)
            model.write_text(patched, encoding="utf-8")
            if self._compiled.exists():
                self._compiled.replace(self._backup)
            self._log_path = output / "compile.log"
            self._log = self._log_path.open("w", encoding="utf-8")
            self._process = subprocess.Popen(command, cwd=compiler.parent, stdout=self._log,
                                              stderr=subprocess.STDOUT)
            self._started = time.monotonic()
            self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
            context.window_manager.modal_handler_add(self)
            context.window_manager["deadlock_build_running"] = True
            settings.status = f"Compiling {name}; Esc cancels. Log: {self._log_path}"
            return {"RUNNING_MODAL"}
        except Exception as error:
            self.finish(context, success=False)
            return report_error(self, settings, error)

    def finish(self, context, success):
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)
        if self._log:
            self._log.close()
            self._log = None
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        if hasattr(self, "_backup") and self._backup.exists():
            if success:
                self._backup.unlink()
            else:
                self._backup.replace(self._compiled)
        context.window_manager["deadlock_build_running"] = False

    def modal(self, context, event):
        settings = context.scene.deadlock_export
        if event.type == "ESC" or time.monotonic() - self._started > 600:
            self.finish(context, success=False)
            settings.status = "Build cancelled or timed out; see compile.log"
            return {"CANCELLED"}
        if event.type != "TIMER" or self._process.poll() is None:
            return {"PASS_THROUGH"}
        try:
            if self._process.returncode or not self._compiled.is_file():
                raise ValueError(f"Source 2 compile failed; see {self._log_path}")
            files = collect_compiled(self._game_dir, self._compiled, self._replacement, self._retail)
            pack_vpk(files, self._output)
            self.finish(context, success=True)
            settings.status = f"Built {self._output}. Install through your mod manager and test in game."
            self.report({"INFO"}, f"Built {self._output.name}")
            return {"FINISHED"}
        except Exception as error:
            self.finish(context, success=False)
            return report_error(self, settings, error)


class DL_OT_copy_status(bpy.types.Operator):
    bl_idname = "deadlock.copy_status"
    bl_label = "Copy Status"
    bl_description = "Copy the full status message to the clipboard"

    def execute(self, context):
        context.window_manager.clipboard = context.scene.deadlock_export.status
        return {'FINISHED'}


def status_lines(context, message):
    """Wrap status text to the sidebar width using Blender's UI font."""
    import blf

    scale = context.preferences.system.ui_scale
    blf.size(0, context.preferences.ui_styles[0].widget.points * scale)
    region = context.region
    view_width = (region.view2d.region_to_view(region.width, 0)[0]
                  - region.view2d.region_to_view(0, 0)[0])
    # Leave room for panel padding, the scrollbar and sidebar category tabs.
    width = max(1, view_width - 64 * scale)
    for paragraph in message.split('\n'):
        line = ''
        for word in paragraph.split():
            candidate = f'{line} {word}' if line else word
            if line and blf.dimensions(0, candidate)[0] > width:
                yield line
                line = ''
            if blf.dimensions(0, word)[0] > width:
                for character in word:
                    if line and blf.dimensions(0, line + character)[0] > width:
                        yield line
                        line = ''
                    line += character
            else:
                line = f'{line} {word}' if line else word
        yield line


class DL_PT_export(bpy.types.Panel):
    bl_label = "Deadlock Blender Toolkit"
    bl_idname = "DL_PT_export"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Deadlock Toolkit"

    def draw(self, context):
        settings = context.scene.deadlock_export
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        header, body = layout.panel('deadlock_setup', default_closed=False)
        header.label(text='Setup', icon='PREFERENCES')
        if body:
            header, directories = body.panel('deadlock_automatic_tools', default_closed=True)
            header.label(text='Directories', icon='FILE_FOLDER')
            if directories:
                fields = directories.split(factor=0.38)
                labels = fields.column()
                labels.alignment = 'RIGHT'
                controls = fields.column()
                controls.use_property_split = False
                for label, prop in (('Game folder', 'game_dir'), ('Export folder', 'output_dir'),
                                    ('Reduced CSDK', 'sdk_root'), ('Viewer CLI', 'viewer_cli')):
                    labels.label(text=label)
                    controls.prop(settings, prop, text='')
            body.separator()
            fields = body.split(factor=0.38)
            labels = fields.column()
            labels.alignment = 'RIGHT'
            controls = fields.column()
            controls.use_property_split = False
            labels.label(text='')
            controls.operator('deadlock.refresh_heroes', icon='FILE_REFRESH')
            labels.label(text='Replace hero')
            controls.prop(settings, 'hero', text='')
            if _catalog:
                try:
                    resource = selected_hero(settings)[2]
                    labels.label(text='Hero resource')
                    controls.label(text=resource, icon='FILE_3D')
                except ValueError:
                    pass
            labels.separator()
            controls.separator()
            labels.label(text='Model collection')
            controls.prop(settings, 'collection', text='')

        header, body = layout.panel('deadlock_preview', default_closed=True)
        header.label(text='Preview & Retarget', icon='ARMATURE_DATA')
        if body:
            body.prop(settings, 'proportions')
            body.operator('deadlock.retarget_original_preview', icon='CONSTRAINT_BONE')
            from .preview import preview_rig, RESTORE_KEY
            rig = preview_rig(context, settings)
            from .physics import physics_rig
            size_rig = physics_rig(context, settings)
            if size_rig is not None and size_rig.get('deadlock_retarget_mapping'):
                body.prop(size_rig, 'deadlock_avatar_scale')
            body.operator('deadlock.fit_avatar_height', icon='FULLSCREEN_ENTER')
            body.separator()
            idle_active = rig is not None and RESTORE_KEY in rig
            body.operator('deadlock.toggle_idle_preview',
                          text='Restore Previous Pose' if idle_active else 'Preview Idle Pose',
                          icon='POSE_HLT', depress=idle_active)
            from .physics_preview import active
            running = active()
            row = body.row(align=True)
            row.operator('deadlock.physics_preview',
                         text='Stop Physics Preview' if running else 'Preview Physics',
                         icon='PAUSE' if running else 'PLAY').reset = False
            if running:
                row.operator('deadlock.physics_preview', text='', icon='FILE_REFRESH').reset = True
            if running:
                body.label(text='Move the rig or play the timeline')
            header, adjustments = body.panel('deadlock_pose_adjustments', default_closed=True)
            header.label(text='Pose adjustments')
            if adjustments:
                adjustments.operator('deadlock.preserve_retarget_proportions', icon='CON_TRANSFORM')
                adjustments.operator('deadlock.fit_retarget_legs', icon='MOD_ARMATURE')

        header, body = layout.panel('deadlock_physics', default_closed=True)
        header.label(text='Physics', icon='PHYSICS')
        if body:
            from .physics import physics_rig, chains
            physics_target = physics_rig(context, settings)
            if physics_target:
                body.label(text=physics_target.name, icon='ARMATURE_DATA')
                entries = chains(physics_target)
                if not entries:
                    body.label(text='No physics chains configured')
                for index, entry in enumerate(entries):
                    row = body.row(align=True)
                    row.label(text=entry['root'])
                    edit = row.operator('deadlock.physics_chain', text='', icon='PREFERENCES')
                    edit.index = index
                    remove = row.operator('deadlock.remove_physics_chain', text='', icon='X')
                    remove.index = index
                if any(entry.get('lock_twist') for entry in entries):
                    body.label(text='Twist lock exports native jiggle bones', icon='INFO')
                body.operator('deadlock.physics_chain', icon='ADD').index = -1
                row = body.row(align=True)
                row.operator('deadlock.save_physics_preset', text='Save Preset', icon='FILE_TICK')
                row.operator('deadlock.load_physics_preset', text='Load Preset', icon='FILE_FOLDER')
                body.label(text='Compiled with Build Retargeted VPK')
            else:
                body.label(text='Create an original rig retarget preview first', icon='INFO')

        header, body = layout.panel('deadlock_build_export', default_closed=True)
        header.label(text='Build & Export', icon='PACKAGE')
        if body:
            body.prop(settings, 'texture_size')
            body.prop(settings, 'keep_native_gun')
            body.prop(settings, 'install')
            body.separator()
            body.operator('deadlock.automatic_conversion', icon='PACKAGE').retarget = True

        header, body = layout.panel('deadlock_manual_workflow', default_closed=True)
        header.label(text='Manual Workflow', icon='MODIFIER')
        if body:
            body.operator('deadlock.load_reference_rig', icon='ARMATURE_DATA')
            body.operator('deadlock.check_rig', icon='CHECKMARK')
            body.operator('deadlock.export_source', icon='EXPORT')
            body.label(text='FBX bakes the current shape-key appearance')
            header, prepared = body.panel('deadlock_build_setup', default_closed=True)
            header.label(text='Compile a prepared hero replacement', icon='PACKAGE')
            if prepared:
                prepared.prop(settings, 'compiler')
                prepared.prop(settings, 'content_dir')
                prepared.prop(settings, 'sdk_game_dir')
                prepared.prop(settings, 'template')
                header, materials = prepared.panel('deadlock_material_bindings', default_closed=True)
                header.label(text='Material bindings', icon='MATERIAL')
                if materials:
                    materials.operator('deadlock.sync_materials')
                    for binding in settings.bindings:
                        if binding.material:
                            materials.label(text=binding.material.name)
                            materials.prop(binding, 'resource', text='VMAT')
                prepared.operator('deadlock.build_mod', icon='PACKAGE')
                prepared.label(text='Requires hero rig preparation and Source 2 tools')

        layout.separator()
        header = layout.row(align=True)
        header.label(text='Status', icon='INFO')
        header.operator('deadlock.copy_status', text='Copy', icon='COPYDOWN')
        status = layout.box()
        for line in status_lines(context, settings.status):
            status.label(text=line)


def find_game():
    candidates = []
    for steam in [Path.home() / ".local/share/Steam", Path.home() / ".steam/steam"]:
        candidates.append(steam / "steamapps/common/Deadlock/game/citadel")
        libraries = steam / "steamapps/libraryfolders.vdf"
        if libraries.is_file():
            text = libraries.read_text(encoding="utf-8", errors="replace")
            for path in re.findall(r'"path"\s+"([^"]+)"', text):
                candidates.append(Path(path.replace("\\\\", "\\")) / "steamapps/common/Deadlock/game/citadel")
    if os.name == "nt":
        candidates.append(Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) /
                          "Steam/steamapps/common/Deadlock/game/citadel")
    return next((str(path) for path in candidates if (path / "pak01_dir.vpk").is_file()), "")


_classes = (DL_Material, DL_Settings, DL_OT_refresh, DL_OT_reference, DL_OT_check,
            DL_OT_export, DL_OT_materials, DL_OT_retarget_preview, DL_OT_fit_retarget_legs, DL_OT_preserve_retarget_proportions,
            DL_OT_fit_avatar_height, DL_OT_idle_preview,
            DL_OT_physics_preview, DL_OT_physics_chain, DL_OT_physics_remove, DL_OT_physics_save, DL_OT_physics_load,
            DL_OT_automatic, DL_OT_build, DL_OT_copy_status, DL_PT_export)


def initialize_settings():
    # Blender restricts bpy.context/data while enabling add-ons. Initialize after registration.
    if not getattr(bpy.context, "scene", None):
        return 0.2
    settings = bpy.context.scene.deadlock_export
    if not settings.game_dir:
        settings.game_dir = find_game()
    if not settings.output_dir:
        settings.output_dir = str(Path.home() / "DeadlockExports")
    if settings.collection is None:
        settings.collection = bpy.data.collections.get("Collection")
    tools = Path.home() / "Tools/DeadlockExporter"
    if not settings.sdk_root and (tools / "sdk/Reduced_CSDK_12").is_dir():
        settings.sdk_root = str(tools / "sdk/Reduced_CSDK_12")
    viewer = tools / "source2viewer" / ("Source2Viewer-CLI.exe" if os.name == "nt" else "Source2Viewer-CLI")
    if not settings.viewer_cli and viewer.is_file():
        settings.viewer_cli = str(viewer)
    if settings.game_dir:
        try:
            load_catalog(settings)
            settings.status = f"{len(_catalog)} heroes loaded. Choose a hero and check your rig."
        except Exception as error:
            settings.status = str(error)
    return None


@bpy.app.handlers.persistent
def on_file_loaded(_):
    if not bpy.app.timers.is_registered(initialize_settings):
        bpy.app.timers.register(initialize_settings, first_interval=0.1)


def register():
    from .retarget_size import avatar_scale, set_avatar_scale
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.deadlock_export = PointerProperty(type=DL_Settings)
    bpy.types.Object.deadlock_avatar_scale = FloatProperty(
        name='Overall size', default=1.0, min=0.01, soft_min=0.1, soft_max=3.0, precision=3,
        description='Uniform size multiplier for this retarget preview; 1 is its initial size. Also affects VPK export',
        get=avatar_scale, set=set_avatar_scale)
    bpy.app.handlers.load_post.append(on_file_loaded)
    bpy.app.timers.register(initialize_settings, first_interval=0.1)


def unregister():
    manager = getattr(bpy.context, "window_manager", None)
    if manager and manager.get("deadlock_build_running", False):
        raise RuntimeError("Cancel the active build before disabling the add-on")
    from .physics_preview import stop
    stop()
    if bpy.app.timers.is_registered(initialize_settings):
        bpy.app.timers.unregister(initialize_settings)
    if on_file_loaded in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_file_loaded)
    del bpy.types.Scene.deadlock_export
    del bpy.types.Object.deadlock_avatar_scale
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
    _catalog.clear()


if __name__ == "__main__":
    register()
