"""Source 2 file handling, independent of Blender and third-party packages."""

import io
import re
import struct
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class Entry:
    crc: int
    archive: int
    offset: int
    length: int
    preload: bytes


def resource_path(value):
    value = value.replace("\\", "/")
    path = PurePosixPath(value)
    if not value or value.startswith("/") or ":" in value or any(
        part in ("", ".", "..") for part in value.split("/")
    ):
        raise ValueError("Expected a relative Source 2 resource path")
    return path.as_posix()


class VPK:
    """Read Valve VPK v1/v2 directory entries and split archive data."""

    def __init__(self, filename):
        self.path = Path(filename)
        self.entries = {}
        with self.path.open("rb") as file:
            signature, version, tree_size = struct.unpack("<III", file.read(12))
            if signature != 0x55AA1234 or version not in (1, 2):
                raise ValueError("Not a supported Valve VPK (v1 or v2)")
            if version == 2:
                if len(file.read(16)) != 16:
                    raise ValueError("Truncated VPK header")
            self.data_start = file.tell() + tree_size
            tree = io.BytesIO(file.read(tree_size))
            if tree.getbuffer().nbytes != tree_size:
                raise ValueError("Truncated VPK directory")

        def string():
            chars = bytearray()
            while True:
                char = tree.read(1)
                if not char:
                    raise ValueError("Unterminated VPK directory string")
                if char == b"\0":
                    return chars.decode("utf-8")
                chars.extend(char)

        while extension := string():
            while directory := string():
                while name := string():
                    data = tree.read(18)
                    if len(data) != 18:
                        raise ValueError("Truncated VPK directory entry")
                    crc, size, archive, offset, length, end = struct.unpack("<IHHIIH", data)
                    if end != 0xFFFF:
                        raise ValueError("Invalid VPK entry terminator")
                    preload = tree.read(size)
                    if len(preload) != size:
                        raise ValueError("Truncated VPK preload")
                    prefix = "" if directory == " " else directory + "/"
                    suffix = "" if extension == " " else "." + extension
                    key = resource_path(prefix + name + suffix)
                    self.entries[key] = Entry(crc, archive, offset, length, preload)

    def read(self, name):
        entry = self.entries[resource_path(name)]
        if entry.archive == 0x7FFF:
            filename, offset = self.path, self.data_start + entry.offset
        else:
            if not self.path.name.endswith("_dir.vpk"):
                raise ValueError("Split VPK must have a *_dir.vpk filename")
            filename = self.path.with_name(self.path.name[:-8] + f"_{entry.archive:03d}.vpk")
            offset = entry.offset
        data = b""
        if entry.length:
            with filename.open("rb") as file:
                file.seek(offset)
                data = file.read(entry.length)
            if len(data) != entry.length:
                raise ValueError(f"Truncated VPK data: {name}")
        data = entry.preload + data
        if zlib.crc32(data) & 0xFFFFFFFF != entry.crc:
            raise ValueError(f"VPK checksum mismatch: {name}")
        return data


def hero_catalog(metadata, entries, labels):
    """Use the installed roster's explicit model references, not filename heuristics."""
    heroes = []
    for key, data in metadata.items():
        if not key.startswith("hero_") or key == "hero_base" or not isinstance(data, dict):
            continue
        if data.get("m_bDisabled") or data.get("m_eHeroDevelopmentState") not in (
            "EHeroDevState_Release", "EHeroDevState_PreRelease"
        ):
            continue
        model = str(data.get("m_strModelName", ""))
        if not model.endswith(".vmdl"):
            continue
        model = resource_path(model + "_c")
        if model not in entries:
            continue
        name = labels.get(key, key.removeprefix("hero_").replace("_", " ").title())
        if data.get("m_eHeroDevelopmentState") == "EHeroDevState_PreRelease":
            name += " (pre-release)"
        heroes.append((key, name, model))
    return sorted(heroes, key=lambda item: item[1])


def hero_labels(data):
    text = data.decode("utf-16") if data.startswith((b"\xff\xfe", b"\xfe\xff")) else data.decode("utf-8-sig")
    return dict(re.findall(r'"(hero_[a-z0-9_]+):n"\s+"([^"\n]+)"', text))


def pack_vpk(files, output):
    """Write a single-file v1 VPK using inline file data and CRC32 checksums."""
    groups = defaultdict(lambda: defaultdict(list))
    for name, source in sorted(files.items()):
        name = resource_path(name).lower()
        path = PurePosixPath(name)
        if not path.suffix:
            raise ValueError(f"Cannot pack a file without an extension: {name}")
        groups[path.suffix[1:]][str(path.parent) if str(path.parent) != "." else " "].append(
            (path.stem, Path(source))
        )
    tree, payload = bytearray(), bytearray()
    for extension, directories in sorted(groups.items()):
        tree.extend(extension.encode() + b"\0")
        for directory, entries in sorted(directories.items()):
            tree.extend(directory.encode() + b"\0")
            for name, source in entries:
                data = source.read_bytes()
                tree.extend(name.encode() + b"\0")
                tree.extend(struct.pack("<IHHIIH", zlib.crc32(data) & 0xFFFFFFFF, 0,
                                        0x7FFF, len(payload), len(data), 0xFFFF))
                payload.extend(data)
            tree.extend(b"\0")
        tree.extend(b"\0")
    tree.extend(b"\0")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_bytes(struct.pack("<III", 0x55AA1234, 1, len(tree)) + tree + payload)
    temporary.replace(output)


def class_spans(text):
    """Locate KV3 objects without mistaking braces inside strings/comments for syntax."""
    tokens = re.finditer(r'<!--.*?-->|//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|[{}]|\b_class\b',
                         text, re.DOTALL)
    stack, spans, marker = [], [], None
    for token in tokens:
        value = token.group()
        if value.startswith(("<!--", "//", "/*")):
            continue
        if value == "{":
            stack.append([token.start(), None])
        elif value == "}":
            if not stack:
                raise ValueError("Unbalanced ModelDoc braces")
            start, cls = stack.pop()
            if cls:
                spans.append((start, token.end(), cls))
        elif value == "_class":
            marker = len(stack)
        elif value.startswith('"') and marker is not None:
            if stack and marker == len(stack):
                stack[-1][1] = value[1:-1]
            marker = None
    if stack:
        raise ValueError("Unbalanced ModelDoc braces")
    return spans


def patch_modeldoc(template, mesh_path, materials=None):
    """Replace render geometry while retaining attachments, skeleton and animation nodes.

    Only a single default material group is supported in this initial workflow. Models
    with bodygroups/morph rules need a deliberately edited template, rather than silently
    retaining rules that refer to the old geometry.
    """
    mesh_path = resource_path(mesh_path)
    spans = class_spans(template)
    classes = [cls for _, _, cls in spans]
    if classes.count("RenderMeshList") != 1:
        raise ValueError("Template must contain exactly one RenderMeshList")
    unsupported = {"BodyGroupList", "MorphControlList", "MorphRuleList", "MeshGroupList", "MaterialGroup"} & set(classes)
    if unsupported:
        raise ValueError("Edit the hero template to remove obsolete geometry rules: " + ", ".join(sorted(unsupported)))
    replacement = ('{\n\t_class = "RenderMeshList"\n\tchildren = [\n'
                   '\t\t{\n\t\t\t_class = "RenderMeshFile"\n'
                   '\t\t\tname = "deadlock_toolkit"\n'
                   f'\t\t\tfilename = "{mesh_path}"\n'
                   '\t\t\timport_scale = 1.0\n\t\t}\n\t]\n}')
    edits = []
    if materials is not None:
        import json
        rows = []
        for source, target in sorted(materials.items()):
            target = resource_path(target)
            if not target.endswith(".vmat"):
                raise ValueError("Material bindings must reference .vmat files")
            rows.append('{ from = ' + json.dumps(source) + ' to = ' + json.dumps(target) + ' }')
        if classes.count("DefaultMaterialGroup") != 1:
            raise ValueError("Template must have exactly one DefaultMaterialGroup")
        group = ('{ _class = "DefaultMaterialGroup" remaps = [ ' + ", ".join(rows) +
                 ' ] use_global_default = false global_default_material = "" }')
    for start, end, cls in spans:
        if cls == "RenderMeshList":
            edits.append((start, end, replacement))
        elif cls == "LODGroupList":
            edits.append((start, end, '{ _class = "LODGroupList" children = [ ] }'))
        elif cls == "DefaultMaterialGroup" and materials is not None:
            edits.append((start, end, group))
    for start, end, value in sorted(edits, reverse=True):
        template = template[:start] + value + template[end:]
    return template


def native_weapon_meshes(template):
    """Find separate full-detail weapon meshes; lower LODs must not overlap them."""
    weapons=[]
    for start,end,cls in class_spans(template):
        if cls!='RenderMeshFile':continue
        fragment=template[start:end]
        name=re.search(r'\bname\s*=\s*"([^"]+)"',fragment)
        filename=re.search(r'\bfilename\s*=\s*"([^"]+)"',fragment)
        if not name or not filename:continue
        label=name.group(1).lower()
        tokens=set(re.split(r'[^a-z0-9]+',label))
        if not tokens & {'weapon','gun','book','pillow','balloon','hammer'}:continue
        if re.search(r'(?:^|[^a-z0-9])lod\d*(?:$|[^a-z0-9])',label):continue
        weapons.append({'name':name.group(1),'filename':filename.group(1),'fragment':fragment})
    return weapons


def automatic_modeldoc(template, mesh_path, materials, keep_native_gun=True):
    """Adapt a freshly extracted hero document without artist-maintained templates."""
    import json
    spans = class_spans(template)
    remove = {"LODGroupList", "BodyGroupList", "MeshGroupList", "MorphControlList", "MorphRuleList",
              "MaterialGroupList", "NmSkeletonList", "AnimGraph2List", "GameDataList"}
    edits = []
    for start, end, cls in spans:
        if cls in remove:
            comma = re.match(r'\s*,',template[end:])
            edits.append((start,end+comma.end() if comma else end,""))
    # Preserve native weapon geometry with the hero's own weights, attachments and materials.
    weapons = [mesh['fragment'] for mesh in native_weapon_meshes(template)] if keep_native_gun else []
    render = ('{ _class = "RenderMeshList" children = [ '
              '{ _class = "RenderMeshFile" name = "deadlock_toolkit" filename = ' + json.dumps(resource_path(mesh_path)) +
              ' import_scale = 1.0 }, ' + ", ".join(weapons) + ' ] }')
    for start, end, cls in spans:
        if cls == "RenderMeshList":
            edits.append((start, end, render))
    # Operate only on outermost selected nodes, avoiding overlapping nested text edits.
    outer = [edit for edit in edits if not any(other[0] < edit[0] and edit[1] < other[1] for other in edits)]
    for start, end, value in sorted(outer, reverse=True):
        template = template[:start] + value + template[end:]
    rows = ", ".join('{ from = ' + json.dumps(source) + ' to = ' + json.dumps(resource_path(target)) + ' }'
                     for source, target in sorted(materials.items()))
    material_node = ('{ _class = "MaterialGroupList" children = [ { _class = "DefaultMaterialGroup" '
                     'remaps = [ ' + rows + ' ] use_global_default = false global_default_material = "" } ] },')
    root = re.search(r'rootNode\s*=\s*\{\s*_class\s*=\s*"RootNode"\s*children\s*=\s*\[', template)
    if root is None:
        raise ValueError("Extracted ModelDoc has an unsupported root structure")
    template = template[:root.end()] + "\n" + material_node + template[root.end():]
    return template


def replace_resource_blocks(data, replacements):
    """Preserve untouched Source 2 blocks byte-for-byte while replacing named block indices."""
    if len(data) < 16:
        raise ValueError("Truncated compiled resource header")
    file_size, header_version, resource_version, table_offset, count = struct.unpack_from("<IHHII", data)
    if header_version != 12 or file_size > len(data):
        raise ValueError("Unsupported compiled resource header")
    table = 8 + table_offset
    entries = []
    for index in range(count):
        info = table + index * 12
        name, relative, size = struct.unpack_from("<4sII", data, info)
        offset = info + 4 + relative
        if offset + size > len(data):
            raise ValueError("Compiled resource block exceeds file bounds")
        entries.append((name, replacements.get(index, data[offset:offset + size])))
    if any(index < 0 or index >= count for index in replacements):
        raise ValueError("Unknown resource block index")
    result = bytearray(struct.pack("<IHHII", 0, header_version, resource_version, 8, count))
    result.extend(bytes(12 * count))
    for index, (name, payload) in enumerate(entries):
        result.extend(bytes((-len(result)) % 8))
        info = 16 + index * 12
        struct.pack_into("<4sII", result, info, name, len(result) - info - 4, len(payload))
        result.extend(payload)
    struct.pack_into("<I", result, 0, len(result))
    result.extend(data[file_size:])
    return bytes(result)


def keep_selection_physics_live(modelkeys):
    """Skip UI cloth pre-settling/freezing without changing gameplay physics."""
    import json
    text = str(modelkeys)
    path = ('CitadelModelSceneData_t', 'm_scenes', 'hero_selection', 'm_settings', 'hero_root')

    def set_flag(text, names):
        # Match scopes rather than strings: comments, other scenes and quoted
        # braces must not redirect an edit to the wrong hero_root.
        tokens = [token for token in re.finditer(
            r'<!--.*?-->|//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|[{}\[\]=,]|[^\s{}\[\]=,]+',
            text, re.DOTALL) if not token.group().startswith(('<!--', '//', '/*'))]
        closing, stack = {}, []
        for index, token in enumerate(tokens):
            value = token.group()
            if value in ('{', '['):
                stack.append(index)
            elif value in ('}', ']'):
                if not stack or tokens[stack[-1]].group() != ('{' if value == '}' else '['):
                    raise ValueError('Malformed retail model keys')
                closing[stack.pop()] = index
        if stack or not tokens or tokens[0].group() != '{' or closing.get(0) != len(tokens) - 1:
            raise ValueError('Malformed retail model keys')

        def edit_scope(start, remaining):
            end = closing[start]
            matches, index = [], start + 1
            while index < end:
                value = tokens[index].group()
                if value in ('{', '['):
                    index = closing[index] + 1
                    continue
                name = json.loads(value) if value.startswith('"') else value
                if name == remaining[0] and tokens[index + 1].group() == '=':
                    matches.append(index + 2)
                index += 1
            if len(matches) > 1:
                raise ValueError('Duplicate model-key setting: ' + remaining[0])
            if not matches:
                addition = remaining[-1] + ' = true'
                for name in reversed(remaining[:-1]):
                    addition = name + ' = { ' + addition + ' }'
                position = tokens[end].start()
                return text[:position] + '\n' + addition + '\n' + text[position:]
            value_index = matches[0]
            value = tokens[value_index]
            if len(remaining) > 1:
                if value.group() != '{':
                    raise ValueError('Expected model-key object: ' + remaining[0])
                return edit_scope(value_index, remaining[1:])
            if value.group() not in ('true', 'false'):
                raise ValueError('Expected model-key boolean: ' + remaining[0])
            return text[:value.start()] + 'true' + text[value.end():]

        return edit_scope(0, names)

    for flag in ('m_bDontPreSettleCloth', 'm_bDisableFreezeCloth'):
        text = set_flag(text, path + (flag,))
    return text


def retarget_model_keys(modelkeys, constraints):
    """Preserve retail keys, adding retarget rules and the UI physics timing fix."""
    if not constraints:
        return modelkeys
    if not isinstance(modelkeys, str):
        raise ValueError('Retarget export requires text model keys from the selected hero')
    return keep_selection_physics_live(append_runtime_constraints(modelkeys, constraints))


def append_runtime_constraints(modelkeys, constraints):
    """Append authored constraints while preserving all retail model-key metadata."""
    import json
    if not constraints:
        return str(modelkeys)
    def encode(value):
        if isinstance(value, dict):
            return '{ ' + ' '.join(key + ' = ' + encode(child) for key, child in value.items()) + ' }'
        if isinstance(value, list):
            return '[ ' + ', '.join(encode(child) for child in value) + ' ]'
        return json.dumps(value, allow_nan=False)
    text = str(modelkeys)
    tokens = list(re.finditer(r'<!--.*?-->|//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\bBoneConstraintList\b|[\[\]]|[^\s\[\]]+', text, re.DOTALL))
    marker = False
    depth = 0
    last = '['
    for token in tokens:
        value = token.group()
        if value.startswith(('<!--', '//', '/*', '"')):
            continue
        if value == 'BoneConstraintList':
            if marker: raise ValueError('Multiple BoneConstraintList entries')
            marker = True
        elif marker and value == '[':
            depth += 1
            last = '['
        elif marker and value == ']':
            depth -= 1
            if depth == 0:
                separator = ',' if last != '[' and not last.endswith(',') else ''
                addition = separator + '\n' + ',\n'.join(encode(rule) for rule in constraints) + ',\n'
                return text[:token.start()] + addition + text[token.start():]
            last = ']'
        elif marker:
            last = value
    if marker:
        raise ValueError('Malformed BoneConstraintList')
    end = text.rfind('}')
    if end < 0: raise ValueError('Malformed retail model keys')
    return text[:end] + '\nBoneConstraintList = ' + encode(constraints) + '\n' + text[end:]
