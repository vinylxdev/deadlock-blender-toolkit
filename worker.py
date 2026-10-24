"""Run costly conversion outside the interactive Blender process."""

import json
import os
import sys
import time
import traceback
from pathlib import Path

import bpy


def load_addon():
    """Load the package beside this worker, including renamed GitHub downloads."""
    import importlib
    folder = Path(__file__).resolve().parent
    package = __package__ or folder.name
    existing = sys.modules.get(package)
    if existing:
        try:
            existing.unregister()
        except Exception:
            pass
        for name in list(sys.modules):
            if name == package or name.startswith(package + '.'):
                del sys.modules[name]
    sys.path.insert(0, str(folder.parent))
    importlib.invalidate_caches()
    return importlib.import_module(package)


def run(config):
    global __package__
    if bpy.context.scene is None:
        scene=bpy.data.scenes.new('Deadlock Export')
        bpy.context.window.scene=scene
    addon = load_addon()
    __package__ = addon.__name__
    addon.register()
    addon.initialize_settings()
    settings=bpy.context.scene.deadlock_export
    for key in ('game_dir','hero','output_dir','sdk_root','viewer_cli','proportions','texture_size'):
        setattr(settings,key,config[key])
    settings.keep_native_gun=config.get('keep_native_gun',True)
    settings.collection=bpy.data.collections.get(config['collection'])
    if settings.collection and settings.collection.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(settings.collection)
    bpy.context.scene.unit_settings.scale_length=config.get('scene_unit_scale',bpy.context.scene.unit_settings.scale_length)
    bpy.context.view_layer.update()
    from .conversion import create_converted
    from .materials import bake_materials
    from .pipeline import extract_hero,run_tool,compiler_environment,repair_runtime_bindings,prepare_sdk_workspace,stage_retail_materials,validate_compiled_model,validate_compiled_materials,validate_compiled_geometry,validate_native_weapon_meshes,install_vpk
    from .core import automatic_modeldoc,native_weapon_meshes,pack_vpk
    status_path=Path(config['status_path'])
    def progress(stage, state='running', **fields):
        state_data={'state':state,'stage':stage,'updated':time.time(),**fields}
        temporary=status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state_data,indent=2),encoding='utf-8');temporary.replace(status_path)
        print(stage,flush=True)
    try:
        output=Path(settings.output_dir)/settings.hero
        output.mkdir(parents=True,exist_ok=True)
        sdk=Path(settings.sdk_root)
        content,game=prepare_sdk_workspace(sdk,settings.hero)
        key,name,replacement=addon.selected_hero(settings)
        retail=addon.archive(settings)
        retail_bytes=retail.read(replacement)
        retarget = config.get('retarget', False)
        constraints = None
        skeleton = None
        if retarget:
            from .retarget_export import prepare_retarget_export,retarget_modeldoc
            progress(f'Preparing original rig retarget for {name}')
            collection,report,skeleton,constraints=prepare_retarget_export(settings)
        else:
            progress(f'Converting avatar rig for {name}')
            collection,report=create_converted(settings)
        progress('Baking Blender shader graphs to Source 2 textures')
        bindings,materials=bake_materials(collection,content,key,settings.texture_size,progress,config['source_blend'])
        report['materials']=materials
        native=extract_hero(Path(settings.viewer_cli),retail_bytes,replacement,Path(settings.game_dir),content,output,progress)
        stem='deadlock_toolkit_retarget' if retarget else 'deadlock_toolkit'
        mesh_resource=f'models/deadlock_toolkit/{key}/{stem}.fbx'
        model_resource=f'models/deadlock_toolkit/{key}/{stem}.vmdl'
        progress('Exporting converted mesh and hero skeleton')
        settings.collection=collection
        addon.export_fbx(settings,content/mesh_resource,source2=True)
        model=content/model_resource
        template=native.read_text(encoding='utf-8-sig')
        native_weapons=native_weapon_meshes(template)
        document=automatic_modeldoc(template,mesh_resource,bindings,settings.keep_native_gun)
        report['keep_native_gun']=settings.keep_native_gun
        report['native_weapon_meshes']=[mesh['name'] for mesh in native_weapons] if settings.keep_native_gun else []
        if retarget:
            document=retarget_modeldoc(document,skeleton)
            from .physics import append_physics_modeldoc
            export_rig=next(o for o in collection.objects if o.type=='ARMATURE')
            physics_nodes=json.loads(export_rig.get('deadlock_physics_nodes','[]'))
            document=append_physics_modeldoc(document,physics_nodes)
        model.write_text(document,encoding='utf-8')
        preview=output/('retarget_export_preview.blend' if retarget else 'converted_preview.blend')
        bpy.data.libraries.write(str(preview),{collection},path_remap='ABSOLUTE',fake_user=True)
        report.update({'preview':str(preview),'modeldoc':str(model),'fbx':str(content/mesh_resource),
                       'compiler_verified':False,'in_game_verified':False})
        report_path=output/('retarget_export_report.json' if retarget else 'conversion_report.json')
        report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
        if not config.get('compile',True):
            progress('Converted rig, materials and ModelDoc sources ready','complete',preview=str(preview),report=str(report_path))
            return 0
        compiler=sdk/'game/bin_cs2/win64/resourcecompiler.exe'
        if not compiler.is_file():
            compiler=sdk/'game/bin/win64/resourcecompiler.exe'
        if not compiler.is_file():
            raise ValueError('CSDK resourcecompiler executable is missing')
        progress('Staging native weapon material dependencies for the compiler')
        native_resource=addon.read_resource(retail_bytes,replacement,model=True)
        retail_staged=stage_retail_materials(native_resource,retail,game)
        report['staged_retail_material_resources']=len(retail_staged)
        progress('Compiling converted model and materials')
        environment=compiler_environment(sdk)
        command=addon.compiler_command(compiler,game,model,environment=environment,shallow=True)
        compiled=game/(model_resource+'_c')
        if compiled.exists():compiled.unlink()
        run_tool(command,compiler.parent,output/('retarget_compile.log' if retarget else 'compile.log'),environment)
        if not compiled.is_file():
            raise ValueError('Compiler did not create the new model in the addon game folder')
        progress('Preserving retail animation bindings and gameplay metadata')
        compiled.write_bytes(repair_runtime_bindings(compiled.read_bytes(),retail_bytes,replacement,constraints))
        report['validation']=validate_compiled_model(compiled.read_bytes(),retail_bytes,replacement,bindings.values(),constraints)
        report['native_weapon_mesh_validation']=validate_native_weapon_meshes(compiled.read_bytes(),replacement,
                                                                            [mesh['name'] for mesh in native_weapons],
                                                                            settings.keep_native_gun)
        report['validated_geometry']=validate_compiled_geometry(compiled.read_bytes(),replacement,collection)
        report['validated_material_textures']=validate_compiled_materials(game,materials['baked_materials'])
        if retarget:
            from .retarget_validation import validate_retarget_export
            progress('Verifying compiled original skeleton and retargeted idle pose')
            report['retarget_validation']=validate_retarget_export(compiled.read_bytes(),replacement,collection,skeleton,constraints,settings)
            from .physics import validate_compiled_physics
            report['physics_validation']=validate_compiled_physics(compiled.read_bytes(),replacement,physics_nodes)
        progress('Validating compiled dependency closure and packaging VPK')
        files=addon.collect_compiled(game,compiled,replacement,retail,retail_staged)
        vpk=Path(config['vpk_path']) if config.get('vpk_path') else output/f'{stem}_{key}.vpk'
        pack_vpk(files,vpk)
        report.update({'compiler_verified':True,'vpk':str(vpk),'packaged_files':sorted(files)})
        installed=None
        if config.get('install'):
            progress('Installing the converted VPK into an exporter-owned addon slot')
            installed=install_vpk(vpk,settings.game_dir,Path(settings.output_dir)/'install_manifest.json')
            report['installed_slot']=str(installed)
        report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
        progress('Converted and compiled VPK ready','complete',vpk=str(vpk),preview=str(preview),report=str(report_path),installed=str(installed) if installed else None)
        return 0
    except Exception as error:
        traceback.print_exc()
        progress(str(error),'failed')
        return 1


if __name__=='__main__':
    try:
        config=json.loads(Path(sys.argv[sys.argv.index('--')+1]).read_text(encoding='utf-8'))
        code=run(config)
    except Exception:
        traceback.print_exc();code=1
    sys.stdout.flush();sys.stderr.flush()
    # Avoid background Blender waiting on desktop audio servers during process shutdown.
    os._exit(code)
