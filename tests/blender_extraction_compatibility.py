"""Compatibility copies must retain large arrays, blobs, flags and binary mesh blocks."""
import importlib
import importlib.util
import os
from pathlib import Path
import struct
import sys

import bpy
import numpy as np

sys.path.insert(0, bpy.utils.user_resource('SCRIPTS', path='addons'))
try: importlib.import_module('SourceIO')
except ImportError: importlib.import_module('SourceIO-master')
folder = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('compatibility_test_addon', folder/'__init__.py')
addon = importlib.util.module_from_spec(spec); sys.modules[spec.name]=addon; spec.loader.exec_module(addon)
from compatibility_test_addon.pipeline import compatible_extraction_model
from SourceIO.library.source2.blocks.kv3_block import KVBlock
from SourceIO.library.source2.keyvalues3.binary_keyvalues import write_valve_keyvalue3
from SourceIO.library.source2.keyvalues3.enums import KV3Signature, KV3Format, KV3CompressionMethod
from SourceIO.library.source2.keyvalues3.types import Object, Array, BinaryBlob, Float

utils, _, _ = addon.sourceio()
values = Object({'arrays':Array([Array([]) for _ in range(65536)]),
                 'blob':BinaryBlob(bytes(range(256))*4), 'scalar':Float(0.25),
                 'small_ints':np.array([0,1,127,255],dtype=np.uint8)})
encoded = utils.WritableMemoryBuffer()
write_valve_keyvalue3(encoded,values,KV3Format.generic,KV3Signature.KV3_V1,KV3CompressionMethod.UNCOMPRESSED)
payloads = [(b'MVTX', b'raw mesh buffer\0\xff'), (b'RED2', bytes(encoded.data)), (b'DATA', bytes(encoded.data))]
source = bytearray(struct.pack('<IHHII',0,12,1,8,len(payloads))+bytes(12*len(payloads)))
for i,(name,payload) in enumerate(payloads):
    source.extend(bytes((-len(source))%8))
    entry = 16+i*12
    struct.pack_into('<4sII',source,entry,name,len(source)-entry-4,len(payload))
    source.extend(payload)
struct.pack_into('<I',source,0,len(source))
source = bytes(source)+b'untouched trailing data'
compatible = compatible_extraction_model(source,'test.vmdl_c')
resource = addon.read_resource(compatible,'test.vmdl_c',model=True)
for index in (1,2):
    data = resource.get_block(KVBlock,block_id=index)
    assert len(data['arrays'])==65536 and all(not row for row in data['arrays'])
    assert data['blob']==values['blob'] and data['scalar']==0.25
    assert np.array_equal(data['small_ints'],values['small_ints'])
block = resource._header.blocks[0]
assert compatible[block.absolute_offset:block.absolute_offset+block.size]==payloads[0][1]
assert compatible.endswith(b'untouched trailing data')
assert source.endswith(b'untouched trailing data')
print('PASS: extraction copy preserves 65536 arrays, blobs, numeric values, binary mesh bytes and trailing data')
sys.stdout.flush()
os._exit(0)
