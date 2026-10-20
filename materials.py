"""Bake supported Blender Principled materials to Source 2 PBR textures."""

import hashlib
import json
import re
from pathlib import Path

import bpy
import numpy as np


def material_slug(name):
    safe = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40] or "material"
    return safe + "_" + hashlib.sha1(name.encode()).hexdigest()[:6]


def new_export_material(slug):
    """Avoid Blender's .001 suffix, which Source 2 treats as a file extension."""
    base = f"dl_{slug}"
    name = base
    suffix = 2
    while name in bpy.data.materials:
        name = f"{base}_{suffix}"
        suffix += 1
    return bpy.data.materials.new(name=name)


def repair_missing_images(material, project_folder):
    repaired, fallback = [], []
    for node in material.node_tree.nodes:
        if node.type != "TEX_IMAGE" or not node.image:
            continue
        image = node.image
        if image.packed_file or Path(bpy.path.abspath(image.filepath)).is_file():
            continue
        suffix = image.filepath.replace("\\", "/").split("/Assets/", 1)
        exact = project_folder / "Assets" / suffix[1] if len(suffix) == 2 else None
        if exact is not None and exact.is_file():
            node.image = bpy.data.images.load(str(exact), check_existing=False)
            repaired.append({"image": image.name, "resolved": str(exact)})
            continue
        # Missing scalar/normal maps use the existing input defaults. Missing albedo is explicit.
        links = [link for link in material.node_tree.links if link.from_node == node]
        for link in links:
            destination = link.to_socket
            if link.to_node.type == "NORMAL_MAP" and destination.name == "Color":
                destination.default_value = (0.5, 0.5, 1.0, 1.0)
            if destination.name == "Base Color":
                raise ValueError(f"Missing required color texture: {image.filepath}")
            material.node_tree.links.remove(link)
        if links:
            fallback.append({"image": image.name, "path": image.filepath,
                             "action": "Disconnected missing auxiliary map on the converted material; used shader input default"})
    return repaired, fallback


def write_vmat(path, textures, values):
    properties = {"shader": "pbr.vfx", "F_USE_NPR_LIGHTING": 1, "F_USE_STATUS_EFFECTS_PROXY": 1,
                  "g_vColorTint1": "[1 1 1 0]"}
    # Citadel's PBR shader uses numbered layer inputs. Unnumbered inputs compile
    # successfully but are ignored, silently substituting the engine defaults.
    for channel, name in {"color": "TextureColor1", "normal": "TextureNormal1", "roughness": "TextureRoughness1",
                          "metallic": "TextureMetalness1", "emission": "TextureSelfIllumMask1"}.items():
        if channel in textures:
            properties[name] = textures[channel]
    if "metallic" in textures:
        properties["F_METALNESS_TEXTURE"] = 1
    if "emission" in textures:
        properties.update({"F_SELF_ILLUM": 1, "g_flSelfIllumScale1": 1.0, "g_vSelfIllumTint1": "[1 1 1 0]"})
    if values.get("transparent"):
        properties.update({"F_TRANSLUCENT": 1, "g_flOpacityScale1": 1.0})
        # Citadel packs color alpha from this separate input; it ignores alpha
        # embedded in TextureColor1 when its translucency input is absent.
        properties['TextureTranslucency1'] = textures['alpha']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('Layer0\n{\n' + "\n".join(f"\t{key} {json.dumps(str(value))}" for key, value in properties.items()) + '\n}\n', encoding="utf-8")


def bake_materials(collection, content_root, hero_key, resolution, progress, project_folder=None):
    meshes = [obj for obj in collection.objects if obj.type == "MESH"]
    originals = {}
    for obj in meshes:
        for index in {polygon.material_index for polygon in obj.data.polygons}:
            if index >= len(obj.data.materials) or obj.data.materials[index] is None:
                raise ValueError(f'Every rendered face needs a material: {obj.name}, slot {index}')
            material=obj.data.materials[index]
            originals[material.name]=material
    window = bpy.context.window
    previous_scene = window.scene if window else bpy.context.scene
    bake_scene = bpy.data.scenes.new("Deadlock Material Bake")
    bake_scene.render.engine = "CYCLES"
    bake_scene.cycles.device = "CPU"
    bake_scene.cycles.samples = 1
    bake_scene.render.bake.margin = 8
    bake_scene.render.bake.use_clear = False
    bake_scene.render.bake.use_selected_to_active = False
    for obj in meshes:
        bake_scene.collection.objects.link(obj)
    bindings, report = {}, {"baked_materials": [], "repaired_images": [], "missing_map_fallbacks": [],
                             "note": "Principled color, normal, roughness, metallic, alpha and emission are baked. Transmission is approximated as transparency; avatar-specific realtime shader effects and physics are excluded."}
    folder_resource = f"materials/deadlock_toolkit/{hero_key}"
    folder = Path(content_root) / folder_resource
    folder.mkdir(parents=True, exist_ok=True)
    dummy = bpy.data.materials.new('Deadlock Bake Other Faces')
    dummy.use_nodes = True
    dummy_image = bpy.data.images.new('Deadlock Bake Scratch', width=1, height=1)
    dummy_node = dummy.node_tree.nodes.new('ShaderNodeTexImage')
    dummy_node.image = dummy_image
    dummy.node_tree.nodes.active = dummy_node
    slot_state = []
    try:
        if window:
            window.scene = bake_scene
        for name, original in sorted(originals.items()):
            progress(f"Baking material {name}")
            material = original.copy()
            material.use_nodes = True
            repaired, fallbacks = repair_missing_images(material, Path(project_folder or bpy.data.filepath).parent)
            report["repaired_images"].extend(repaired)
            report["missing_map_fallbacks"].extend(fallbacks)
            nodes, links = material.node_tree.nodes, material.node_tree.links
            output = next((node for node in nodes if node.type == "OUTPUT_MATERIAL" and node.is_active_output), None)
            if output is None or not output.inputs['Surface'].is_linked:
                raise ValueError(f"Material {name} has no active surface shader")
            shader = output.inputs['Surface'].links[0].from_node
            if shader.type != "BSDF_PRINCIPLED":
                raise ValueError(f"Material {name} needs a Principled BSDF surface for automatic PBR conversion")
            surface = output.inputs['Surface'].links[0].from_socket
            selected = [obj for obj in meshes if original in obj.data.materials[:]]
            slot_state = [(obj,index,slot) for obj in selected for index,slot in enumerate(obj.data.materials)]
            for obj in selected:
                for index, slot in enumerate(obj.data.materials):
                    obj.data.materials[index] = material if slot == original else dummy
                if not obj.data.uv_layers:
                    raise ValueError(f"{obj.name} has no UV map for automatic material baking")
            emission = nodes.new("ShaderNodeEmission")
            image_node = nodes.new("ShaderNodeTexImage")
            nodes.active = image_node
            slug = material_slug(name)
            images, texture_refs = {}, {}
            values = {"roughness": float(shader.inputs['Roughness'].default_value),
                      "metallic": float(shader.inputs['Metallic'].default_value),
                      "opacity": float(shader.inputs['Alpha'].default_value)}
            transmission = float(shader.inputs['Transmission Weight'].default_value)
            if transmission > 0:
                values['opacity'] *= max(0.05, 1.0 - transmission)
            values['transparent'] = shader.inputs['Alpha'].is_linked or values['opacity'] < 0.999
            for channel, socket_name in (('color','Base Color'), ('normal','Normal'), ('roughness','Roughness'),
                                         ('metallic','Metallic'), ('alpha','Alpha'), ('emission','Emission Color')):
                socket = shader.inputs[socket_name]
                strength = shader.inputs['Emission Strength']
                if channel == 'normal' and not socket.is_linked:
                    continue
                if channel in {'roughness','metallic','alpha'} and not socket.is_linked:
                    continue
                if channel == 'emission' and not strength.is_linked and strength.default_value <= 0:
                    continue
                progress(f"Baking {name}: {channel}")
                image = bpy.data.images.new(f"{slug}_{channel}", width=resolution, height=resolution, alpha=True)
                image.colorspace_settings.name = 'sRGB' if channel in {'color','emission'} else 'Non-Color'
                images[channel] = image
                image_node.image = image
                nodes.active = image_node
                for link in list(output.inputs['Surface'].links):
                    links.remove(link)
                for link in list(emission.inputs['Color'].links):
                    links.remove(link)
                if channel == 'normal':
                    links.new(surface, output.inputs['Surface'])
                else:
                    if socket.is_linked:
                        source = socket.links[0].from_socket
                        links.new(source, emission.inputs['Color'])
                    else:
                        value = socket.default_value
                        emission.inputs['Color'].default_value = tuple(value) if hasattr(value,'__len__') else (value,value,value,1)
                    if channel == 'emission':
                        multiply = nodes.new('ShaderNodeMixRGB')
                        multiply.blend_type = 'MULTIPLY'
                        multiply.inputs[0].default_value = 1
                        if socket.is_linked:
                            links.new(socket.links[0].from_socket,multiply.inputs[1])
                        else:
                            multiply.inputs[1].default_value = socket.default_value
                        if strength.is_linked:
                            links.new(strength.links[0].from_socket,multiply.inputs[2])
                        else:
                            value = strength.default_value
                            multiply.inputs[2].default_value = (value,value,value,1)
                        links.new(multiply.outputs[0],emission.inputs['Color'])
                    links.new(emission.outputs[0],output.inputs['Surface'])
                for obj in meshes:
                    obj.select_set(False)
                for obj in selected:
                    obj.select_set(True)
                    bake_scene.view_layers[0].objects.active = obj
                    with bpy.context.temp_override(scene=bake_scene, view_layer=bake_scene.view_layers[0],
                                                   active_object=obj, object=obj, selected_objects=[obj], selected_editable_objects=[obj]):
                        status = bpy.ops.object.bake(type='NORMAL' if channel=='normal' else 'EMIT', use_clear=False)
                    obj.select_set(False)
                    if status != {'FINISHED'}:
                        raise ValueError(f"Texture bake failed for {name}: {channel}")
                image.filepath_raw = str(folder / f"{slug}_{channel}.png")
                image.file_format = 'PNG'
                image.save()
                texture_refs[channel] = f"{folder_resource}/{slug}_{channel}.png"
            if 'alpha' in images:
                pixels = np.empty(resolution*resolution*4,dtype=np.float32)
                alpha = np.empty_like(pixels)
                images['color'].pixels.foreach_get(pixels)
                images['alpha'].pixels.foreach_get(alpha)
                pixels[3::4] = alpha[0::4] * values['opacity']
                images['color'].pixels.foreach_set(pixels)
                images['color'].save()
            elif values['opacity'] < 0.999:
                pixels = np.empty(resolution*resolution*4,dtype=np.float32)
                images['color'].pixels.foreach_get(pixels)
                pixels[3::4] = values['opacity']
                images['color'].pixels.foreach_set(pixels)
                images['color'].save()
            constant_channels = ['roughness', 'metallic']
            if values['transparent']:
                constant_channels.append('alpha')
            for channel in constant_channels:
                if channel not in images:
                    constant=bpy.data.images.new(f'{slug}_{channel}',width=2,height=2)
                    constant.colorspace_settings.name='Non-Color'
                    value=values['opacity'] if channel=='alpha' else values[channel]
                    constant.pixels[:]=[value,value,value,1]*4
                    constant.filepath_raw=str(folder/f'{slug}_{channel}.png')
                    constant.file_format='PNG';constant.save()
                    images[channel]=constant
                    texture_refs[channel]=f'{folder_resource}/{slug}_{channel}.png'
            vmat = f"{folder_resource}/{slug}.vmat"
            write_vmat(Path(content_root)/vmat,texture_refs,values)
            preview = new_export_material(slug)
            preview.use_nodes = True
            preview.diffuse_color = original.diffuse_color
            preview_shader = next(node for node in preview.node_tree.nodes if node.type == 'BSDF_PRINCIPLED')
            for channel,socket_name in (('color','Base Color'),('roughness','Roughness'),('metallic','Metallic'),('emission','Emission Color')):
                if channel in images:
                    tex = preview.node_tree.nodes.new('ShaderNodeTexImage');tex.image=images[channel]
                    preview.node_tree.links.new(tex.outputs['Color'],preview_shader.inputs[socket_name])
                    if channel=='color' and values['transparent']:
                        preview.node_tree.links.new(tex.outputs['Alpha'],preview_shader.inputs['Alpha'])
            if 'normal' in images:
                tex=preview.node_tree.nodes.new('ShaderNodeTexImage');tex.image=images['normal']
                normal=preview.node_tree.nodes.new('ShaderNodeNormalMap')
                preview.node_tree.links.new(tex.outputs['Color'],normal.inputs['Color'])
                preview.node_tree.links.new(normal.outputs['Normal'],preview_shader.inputs['Normal'])
            preview_shader.inputs['Roughness'].default_value=values['roughness']
            preview_shader.inputs['Metallic'].default_value=values['metallic']
            preview_shader.inputs['Emission Strength'].default_value=1 if 'emission' in images else 0
            bindings[preview.name] = vmat
            for obj,index,slot in slot_state:
                obj.data.materials[index] = preview if slot == original else slot
            slot_state = []
            bpy.data.materials.remove(material)
            report['baked_materials'].append({'source':name,'converted':preview.name,'vmat':vmat,'textures':texture_refs})
        return bindings, report
    finally:
        for obj,index,slot in slot_state:
            obj.data.materials[index] = slot
        if window:
            window.scene = previous_scene
        bpy.data.scenes.remove(bake_scene)
        bpy.data.materials.remove(dummy)
        bpy.data.images.remove(dummy_image)
