"""Extract, compile and preserve the selected hero's authoritative runtime bindings."""

import copy
import os
import subprocess
from pathlib import Path

from .core import replace_resource_blocks, retarget_model_keys


def model_attachments(resource):
    """Collect authoritative attachment rows across meshes and LODs."""
    rows = {}
    for mesh in resource.get_blocks(None, 'MDAT'):
        for row in mesh.get('m_attachments', []):
            name = str(row['key'])
            if name in rows and repr(rows[name]) != repr(row):
                raise ValueError('Conflicting attachment definitions across model meshes: ' + name)
            rows[name] = row
    return rows


def validate_model_attachments(resource, reference):
    """Names alone do not prove that camera pivots retain their retail frame."""
    expected = model_attachments(reference)
    actual = model_attachments(resource)
    missing = sorted(expected.keys() - actual.keys())
    if missing:
        raise ValueError('Compiled model is missing retail attachments: ' + ', '.join(missing[:12]))
    changed = sorted(name for name, row in expected.items() if repr(actual[name]) != repr(row))
    if changed:
        raise ValueError('Compiled attachment transforms or bindings differ from retail: ' + ', '.join(changed[:12]))
    return {'attachment_count': len(expected), 'retail_attachments_match': True}


def repair_model_attachments(compiled, original, resource_path):
    """Retain exact retail bone-local attachments after ModelDoc compilation.

    The decompiler/compiler path can express root attachments in the NM root
    frame instead of the model bone frame. Camera offsets then rotate sideways
    even though the native skeleton and attachment names still match.
    """
    from . import read_resource, sourceio
    incoming = read_resource(compiled, resource_path, model=True)
    retail = read_resource(original, resource_path, model=True)
    expected = model_attachments(retail)
    if not expected:
        return compiled
    utils, _, _ = sourceio()
    replacements = {}
    for index, info in enumerate(incoming._header.blocks):
        if info.name != 'MDAT':
            continue
        mesh = incoming.get_block(None, block_id=index)
        existing = mesh.get('m_attachments')
        if existing is None:
            raise ValueError('Compiled mesh has no attachment table')
        # Keep compiler-added rows, preserving the serialized KV3 array type.
        rows = copy.deepcopy(existing)
        rows.clear()
        for row in expected.values():
            rows.append(copy.deepcopy(row))
        for row in existing:
            if str(row['key']) not in expected:
                rows.append(copy.deepcopy(row))
        mesh['m_attachments'] = rows
        buffer = utils.WritableMemoryBuffer()
        mesh.to_buffer(buffer)
        replacements[index] = bytes(buffer.data)
    repaired = replace_resource_blocks(compiled, replacements)
    validate_model_attachments(read_resource(repaired, resource_path, model=True), retail)
    return repaired


def repair_runtime_bindings(compiled, original, resource_path, retarget_constraints=None):
    from . import read_resource, sourceio
    incoming = read_resource(compiled, resource_path, model=True)
    retail = read_resource(original, resource_path, model=True)
    data = incoming.get_block(None, block_name='DATA')
    retail_data = retail.get_block(None, block_name='DATA')
    fields = ('m_animGraph2Refs', 'm_vecNmSkeletonRefs')
    for name in fields:
        if name in retail_data:
            data[name] = copy.deepcopy(retail_data[name])
        else:
            data.pop(name, None)
    # The CSDK's model-keyvalues schema differs from retail. Preserve the retail object shape.
    retail_keys = retail_data['m_modelInfo']['m_keyValueText']
    expected_keys = retarget_model_keys(retail_keys, retarget_constraints)
    if retarget_constraints:
        data['m_modelInfo']['m_keyValueText'] = type(retail_keys)(expected_keys)
    else:
        data['m_modelInfo']['m_keyValueText'] = copy.deepcopy(retail_keys)
    target_bones = {b.name for b in retail.get_bones()}
    current_bones = {b.name for b in incoming.get_bones()}
    aliases={name.replace('$cloth','_cloth',1):name for name in target_bones-current_bones
             if name.startswith('$cloth') and name.replace('$cloth','_cloth',1) in current_bones}
    unresolved=target_bones-current_bones-set(aliases.values())
    if unresolved:
        raise ValueError('Compiler removed required hero bones: ' + ', '.join(sorted(unresolved)[:12]))
    def restore_names(value):
        if isinstance(value,dict):
            for key,child in value.items():
                value[key]=restore_names(child)
        elif isinstance(value,list):
            for index,child in enumerate(value):
                value[index]=restore_names(child)
        elif isinstance(value,str) and value in aliases:
            replacement=type(value)(aliases[value])
            if hasattr(value,'specifier'):replacement.specifier=value.specifier
            return replacement
        return value
    restore_names(data)
    utils, _, _ = sourceio()
    replacements = {}
    buffer = utils.WritableMemoryBuffer()
    data.to_buffer(buffer)
    index = next(i for i,b in enumerate(incoming._header.blocks) if b.name=='DATA')
    replacements[index] = bytes(buffer.data)
    if aliases:
        for index,info in enumerate(incoming._header.blocks):
            if info.name=='MDAT':
                mesh=incoming.get_block(None,block_id=index)
                restore_names(mesh)
                buffer=utils.WritableMemoryBuffer();mesh.to_buffer(buffer)
                replacements[index]=bytes(buffer.data)
    refs = incoming.get_block(None, block_name='RERL')
    native_refs = retail.get_block(None, block_name='RERL')
    binding_paths = {str(x) for x in retail_data.get('m_vecNmSkeletonRefs',[])}
    binding_paths.update(str(x['m_hGraph']) for x in retail_data.get('m_animGraph2Refs',[]))
    names = {ref.name for ref in refs}
    for ref in native_refs:
        if ref.name in binding_paths and ref.name not in names:
            refs.append(ref)
            names.add(ref.name)
    if refs:
        buffer = utils.WritableMemoryBuffer()
        refs.to_buffer(buffer)
        index = next(i for i,b in enumerate(incoming._header.blocks) if b.name=='RERL')
        replacements[index] = bytes(buffer.data)
    repaired = replace_resource_blocks(compiled, replacements)
    repaired = repair_model_attachments(repaired, original, resource_path)
    verified_resource=read_resource(repaired, resource_path, model=True)
    verified = verified_resource.get_block(None, block_name='DATA')
    if target_bones-{bone.name for bone in verified_resource.get_bones()}:
        raise ValueError('Hero bone names did not survive serialization')
    for field in fields:
        if repr(verified.get(field)) != repr(retail_data.get(field)):
            raise ValueError(f'Runtime {field} did not survive serialization')
    if verified['m_modelInfo']['m_keyValueText'] != expected_keys:
        raise ValueError('Model gameplay metadata and UI physics settings did not survive serialization')
    return repaired


def stage_retail_materials(resource,retail,game_dir):
    """Let the compiler inspect retained native prop materials without changing retail files."""
    from . import read_resource
    # RERL lists material dependencies without decoding retail mesh KV3.
    # Newer retail MDAT types may be unsupported by the installed SourceIO,
    # and staging only needs resource paths, not mesh or material contents.
    paths={str(path) for path in resource.get_dependencies() if str(path).endswith('.vmat_c')}
    visited=set();queue=list(paths)
    while queue:
        name=queue.pop()
        if name in visited or name not in retail.entries:continue
        visited.add(name)
        data=retail.read(name)
        destination=Path(game_dir)/name
        destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data)
        queue.extend(str(path) for path in read_resource(data,name).get_dependencies())
    return visited


def validate_native_weapon_meshes(compiled,path,candidates,keep):
    """Fail a keep/hide build when the compiled render meshes contradict the toggle."""
    from . import read_resource
    resource=read_resource(compiled,path,model=True)
    ctrl=resource.get_block(None,block_name='CTRL')
    actual={str(mesh['m_Name']):mesh for mesh in ctrl['embedded_meshes']}
    requested=set(candidates) if keep else set()
    missing=requested-actual.keys()
    unexpected=set(candidates)&actual.keys() if not keep else set()
    if missing:raise ValueError('Compiler removed requested native weapon mesh: '+', '.join(sorted(missing)))
    if unexpected:raise ValueError('Compiler retained a hidden native weapon mesh: '+', '.join(sorted(unexpected)))
    vertices={name:sum(int(row['m_nElementCount']) for row in actual[name]['m_vertexBuffers']) for name in requested}
    if any(count<3 for count in vertices.values()):raise ValueError('Compiled native weapon mesh has no renderable vertices')
    return {'keep_native_gun':keep,'compiled_weapon_meshes':sorted(requested),
            'compiled_weapon_vertex_counts':vertices,'compiled_render_meshes':sorted(actual)}


def validate_compiled_model(compiled,original,path,expected_materials,retarget_constraints=None):
    from . import read_resource,bone_matrices
    resource=read_resource(compiled,path,model=True)
    reference=read_resource(original,path,model=True)
    actual_data=resource.get_block(None,block_name='DATA')
    expected_data=reference.get_block(None,block_name='DATA')
    for field in ('m_animGraph2Refs','m_vecNmSkeletonRefs'):
        if repr(actual_data.get(field))!=repr(expected_data.get(field)):
            raise ValueError(f'Compiled {field} does not match the selected retail hero')
    expected_keys=expected_data['m_modelInfo']['m_keyValueText']
    if actual_data['m_modelInfo']['m_keyValueText']!=retarget_model_keys(expected_keys,retarget_constraints):
        raise ValueError('Compiled model keys do not match the expected gameplay metadata and UI physics settings')
    expected_bones=reference.get_bones()
    actual_bones={bone.name:bone for bone in resource.get_bones()}
    expected_matrices=bone_matrices(expected_bones)
    actual_matrices=bone_matrices(actual_bones.values())
    maximum=0.0
    for bone in expected_bones:
        if bone.name not in actual_bones or actual_bones[bone.name].parent!=bone.parent:
            raise ValueError(f'Compiled hero bone or parent mismatch: {bone.name}')
        error=max(abs(expected_matrices[bone.name][row][column]-actual_matrices[bone.name][row][column])
                  for row in range(4) for column in range(4))
        maximum=max(maximum,error)
    if maximum>0.005:
        raise ValueError(f'Compiled hero skeleton rest transforms differ (maximum error {maximum:.6f})')
    attachments = validate_model_attachments(resource, reference)
    materials=set()
    def walk(value):
        if isinstance(value,dict):
            for child in value.values():walk(child)
        elif isinstance(value,list):
            for child in value:walk(child)
        elif isinstance(value,str) and value.endswith('.vmat'):
            materials.add(str(value))
    for block in resource.get_blocks(None,'MDAT'):walk(block)
    if '.vmat' in materials or set(expected_materials)-materials:
        raise ValueError('Compiled mesh has unresolved material bindings: '+str(sorted(set(expected_materials)-materials)))
    return {'bone_count':len(actual_bones),'max_rest_matrix_error':maximum,
            'material_references':sorted(materials),'animation_bindings_match_retail':True,
            'gameplay_metadata_matches_retail':True,
            'attachment_validation':attachments,
            'selection_cloth_presettle_disabled':bool(retarget_constraints),
            'selection_cloth_freeze_disabled':bool(retarget_constraints)}


def validate_compiled_materials(game_dir, baked_materials):
    """Reject successful compiles that silently discarded the authored textures."""
    from . import read_resource
    verified = {}
    shader_inputs = {'color': 'g_tColor', 'normal': 'g_tNormalRoughness',
                     'roughness': 'g_tNormalRoughness', 'metallic': 'g_tMetalness',
                     'emission': 'g_tSelfIllumMask', 'alpha': 'g_tColor'}
    for material in baked_materials:
        path = material['vmat'] + '_c'
        resource = read_resource((Path(game_dir) / path).read_bytes(), path)
        data = resource.get_block(None, block_name='DATA')
        textures = {str(row['m_name']): str(row['m_pValue']) for row in data['m_textureParams']}
        edit_info = resource.get_block(None, block_name='RED2')
        inputs = {str(row.relative_name) for row in edit_info.inputs}
        for channel, source in material['textures'].items():
            if channel not in shader_inputs:
                continue
            # Uniform maps can be folded into shader constants (and have no
            # texture binding). RED2 proves the compiler consumed the source.
            if source not in inputs:
                raise ValueError(f"Compiled material {material['source']} lost its {channel} texture: {source}")
        verified[material['vmat']] = textures
    return verified


def validate_compiled_geometry(compiled, path, collection):
    """Compare the compiler's vertex positions with the actual converted avatar."""
    import numpy as np
    from mathutils.kdtree import KDTree
    from SourceIO.library.source2.blocks.vertex_index_buffer.vertex_buffer import VertexBuffer
    from . import read_resource, UNIT
    expected = [obj.matrix_world @ vertex.co for obj in collection.objects if obj.type == 'MESH'
                for vertex in obj.data.vertices]
    resource = read_resource(compiled, path, model=True)
    control = resource.get_block(None, block_name='CTRL')
    mesh = next((row for row in control['embedded_meshes'] if row.get('m_Name') in ('deadlock_toolkit', 'vinylrex')), None)
    if mesh is None:
        raise ValueError('Compiled avatar mesh is missing')
    buffers = [VertexBuffer.from_kv(row).get_vertices(resource)['POSITION'] * UNIT
               for row in mesh['m_vertexBuffers']
               if any(field['m_pSemanticName'].upper() == 'POSITION' for field in row['m_inputLayoutFields'])]
    positions = np.concatenate(buffers) if buffers else None
    if positions is None or not len(positions) or not np.isfinite(positions).all():
        raise ValueError('Compiled avatar has invalid vertex positions')
    expected_array = np.asarray(expected)
    actual_bounds = np.array([positions.min(0), positions.max(0)])
    expected_bounds = np.array([expected_array.min(0), expected_array.max(0)])
    bounds_error = float(np.max(np.abs(actual_bounds - expected_bounds)))
    tree = KDTree(len(expected))
    for index, point in enumerate(expected):
        tree.insert(point, index)
    tree.balance()
    sample = np.linspace(0, len(positions)-1, min(2048, len(positions)), dtype=int)
    vertex_error = max(tree.find(positions[index])[2] for index in sample)
    # Long accessory meshes can shift slightly when the compiler rebases FBX
    # skinning onto the authored skeleton. Keep the measured error in the report.
    vertex_tolerance = 0.002
    if bounds_error > 0.001 or vertex_error > vertex_tolerance:
        raise ValueError(f'Compiler changed avatar geometry (bounds error {bounds_error:.6f} m, sampled vertex error {vertex_error:.6f} m)')
    return {'bounds_meters': actual_bounds.tolist(), 'max_bounds_error_meters': bounds_error,
            'max_sampled_vertex_error_meters': vertex_error, 'vertex_tolerance_meters': vertex_tolerance,
            'sampled_vertices': len(sample)}


def install_vpk(vpk_path,game_dir,manifest_path):
    """Own one explicitly requested loose-addon slot; never overwrite another mod."""
    import hashlib,json,shutil
    vpk_path=Path(vpk_path)
    addon_dir=Path(game_dir)/'addons'
    addon_dir.mkdir(parents=True,exist_ok=True)
    manifest_path=Path(manifest_path)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {}
    import re
    if manifest.get('slot') and not re.fullmatch(r'pak[0-9]{2}_dir\.vpk',manifest['slot']):
        raise ValueError('Invalid exporter addon slot in install manifest')
    owned=addon_dir/manifest.get('slot','invalid')
    if not owned.resolve().is_relative_to(addon_dir.resolve()):
        raise ValueError('Exporter addon slot escaped the addons directory')
    if owned.is_file():
        digest=hashlib.sha256(owned.read_bytes()).hexdigest()
        if digest!=manifest.get('sha256'):
            raise ValueError('The exporter addon slot was changed by another mod; choose a new slot by removing its install manifest')
    else:
        owned=next((addon_dir/f'pak{index:02d}_dir.vpk' for index in range(99,0,-1)
                    if not (addon_dir/f'pak{index:02d}_dir.vpk').exists()),None)
        if owned is None:raise ValueError('No free pak01..pak99 addon slot is available')
    temporary=owned.with_suffix('.vpk.tmp')
    shutil.copy2(vpk_path,temporary);temporary.replace(owned)
    manifest_path.write_text(json.dumps({'slot':owned.name,'sha256':hashlib.sha256(owned.read_bytes()).hexdigest(),
                                         'source':str(vpk_path)},indent=2),encoding='utf-8')
    return owned


def compiler_environment(sdk_root):
    environment = os.environ.copy()
    if os.name != 'nt':
        sdk=Path(sdk_root)
        tools=sdk.parents[1] if (sdk.parents[1]/'wineprefix').is_dir() else sdk.parent
        environment['WINEPREFIX'] = str(tools/'wineprefix')
        environment['WINEDEBUG'] = '-all'
        if (tools/'wineprefix/drive_c/windows/system32/dxgi.dll.wine-original').is_file():
            environment['WINEDLLOVERRIDES'] = 'dxgi,d3d11=n'
    return environment


def prepare_sdk_workspace(sdk_root, hero_key):
    """Use common ModelDoc tools without loading incompatible Citadel modtools schemas."""
    sdk=Path(sdk_root)
    name=f'deadlock_toolkit_{hero_key}'
    content=sdk/'content/citadel_addons'/name
    game=sdk/'game/citadel_addons'/name
    content.mkdir(parents=True,exist_ok=True);game.mkdir(parents=True,exist_ok=True)
    info=(sdk/'game/core/gameinfo.gi').read_text(encoding='utf-8-sig')
    start=info.index('SearchPaths');begin=info.index('{',start);end=info.index('}',begin)
    search='\n Game "|gameinfo_path|."\n Mod "|gameinfo_path|."\n Write "|gameinfo_path|."\n Game "citadel"\n Game "core"\n'
    info=info[:begin+1]+search+info[end:]
    if '"HasModAppSystems" "0"' not in info:
        raise ValueError('SDK base gameinfo unexpectedly enables incompatible mod app systems')
    (game/'gameinfo.gi').write_text(info,encoding='utf-8')
    return content,game


def run_tool(command, cwd, log, environment=None, timeout=900):
    with Path(log).open('a', encoding='utf-8') as output:
        output.write('\nCOMMAND: '+repr(command)+'\n');output.flush()
        completed = subprocess.run(command, cwd=cwd, env=environment, stdout=output, stderr=subprocess.STDOUT, timeout=timeout)
    if completed.returncode:
        raise ValueError(f'Tool exited with code {completed.returncode}; see {log}')


def embedded_animation_model(retail_bytes, replacement):
    """Omit AG2 traversal from an extraction copy; runtime bindings use retail bytes."""
    from . import read_resource, sourceio
    model = read_resource(retail_bytes, replacement, model=True)
    data = model.get_block(None, block_name='DATA')
    if not data.pop('m_animGraph2Refs', None):
        raise ValueError('Hero has no animation graph references to omit during extraction')
    utils, _, _ = sourceio()
    buffer = utils.WritableMemoryBuffer()
    data.to_buffer(buffer)
    index = next(i for i, block in enumerate(model._header.blocks) if block.name == 'DATA')
    return replace_resource_blocks(retail_bytes, {index: bytes(buffer.data)})


def extract_hero(cli, retail_bytes, replacement, game_dir, content_root, output, progress):
    output=Path(output)
    original=output/'retail'/Path(replacement).name
    original.parent.mkdir(parents=True,exist_ok=True)
    original.write_bytes(retail_bytes)
    search_info=original.parent/'gameinfo.gi'
    import json
    search_info.write_text('"GameInfo" { FileSystem { SteamAppId 1422450 SearchPaths { '
                          'Game '+json.dumps(str(game_dir))+' Game '+json.dumps(str(Path(game_dir).parent/'core'))+' } } }',encoding='utf-8')
    destination=Path(content_root)/replacement.removesuffix('_c')
    destination.parent.mkdir(parents=True,exist_ok=True)
    progress('Extracting hero ModelDoc and native weapon/animation sources')
    log = output/'extract.log'
    start = log.stat().st_size if log.exists() else 0
    error = None
    try:
        run_tool([str(cli),'-i',str(original),'-o',str(destination),'--game',str(search_info)],
                 original.parent,log)
    except ValueError as failure:
        error = failure
    # Some Viewer versions report decompiler exceptions but still exit with zero.
    with log.open('rb') as stream:
        stream.seek(start)
        diagnostics = stream.read().decode('utf-8', errors='replace')
    if 'Unknown KVType' in diagnostics and 'AnimationGraphLoader.' in diagnostics:
        progress('Viewer cannot read the hero animation graph; extracting embedded animations')
        fallback = original.parent/'embedded'/original.name
        fallback.parent.mkdir(exist_ok=True)
        fallback.write_bytes(embedded_animation_model(retail_bytes, replacement))
        # A previous extraction must not disguise a zero-exit retry failure.
        destination.unlink(missing_ok=True)
        run_tool([str(cli),'-i',str(fallback),'-o',str(destination),'--game',str(search_info)],
                 fallback.parent,log)
    elif error:
        raise error
    if not destination.is_file():
        raise ValueError(f'Source 2 Viewer did not write the requested hero ModelDoc; see {log}')
    return destination
