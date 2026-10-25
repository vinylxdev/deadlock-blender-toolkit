"""Metadata compatibility retries must compose with the existing graph fallback."""
import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

package = types.ModuleType('metadata_extraction_tests')
package.__path__ = [str(Path(__file__).parents[1])]
sys.modules[package.__name__] = package
pipeline = importlib.import_module(package.__name__ + '.pipeline')
METADATA_ERROR = 'Unknown KVType: 26\n at ResourceEditInfo2.Read\n'
GRAPH_ERROR = 'Unknown KVType: 26\n at ResourceEditInfo2.Read\n at AnimationGraphLoader.CollectClips\n'
MESH_ERROR = 'Unknown KVType: 26\n at KeyValuesOrNTRO.Read\n at Mesh.Read\n'


class MetadataExtraction(unittest.TestCase):
    def extract(self, folder):
        return pipeline.extract_hero(Path('/viewer'), b'retail', 'models/fencer.vmdl_c',
                                     folder/'game', folder/'content', folder/'output', lambda _:None)

    def test_metadata_retry_handles_zero_and_nonzero_exit(self):
        for diagnostic, nonzero in ((d,n) for d in (METADATA_ERROR,MESH_ERROR) for n in (False,True)):
            with self.subTest(diagnostic=diagnostic,nonzero=nonzero), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory); calls = []
                def run(command, cwd, log):
                    calls.append(command)
                    with log.open('a') as stream: stream.write(diagnostic if len(calls)==1 else 'Extracted\n')
                    if len(calls)==1 and nonzero: raise ValueError('Tool exited with code 1')
                    if len(calls)==2:
                        self.assertEqual(Path(command[2]).read_bytes(), b'compatible')
                        Path(command[4]).write_text('ModelDoc')
                with patch.object(pipeline, 'run_tool', side_effect=run), \
                        patch.object(pipeline, 'compatible_extraction_model', return_value=b'compatible') as adapt, \
                        patch.object(pipeline, 'embedded_animation_model') as graph:
                    self.assertEqual(self.extract(folder).read_text(), 'ModelDoc')
                adapt.assert_called_once_with(b'retail', 'models/fencer.vmdl_c')
                graph.assert_not_called()
                self.assertEqual((folder/'output/retail/fencer.vmdl_c').read_bytes(), b'retail')

    def test_graph_retry_uses_already_adapted_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); calls = []
            def run(command, cwd, log):
                calls.append(command)
                with log.open('a') as stream: stream.write((METADATA_ERROR, GRAPH_ERROR, 'Extracted\n')[len(calls)-1])
                if len(calls)==3:
                    self.assertEqual(Path(command[2]).read_bytes(), b'compatible embedded')
                    Path(command[4]).write_text('ModelDoc')
            with patch.object(pipeline, 'run_tool', side_effect=run), \
                    patch.object(pipeline, 'compatible_extraction_model', return_value=b'compatible'), \
                    patch.object(pipeline, 'embedded_animation_model', return_value=b'compatible embedded') as graph:
                self.extract(folder)
            graph.assert_called_once_with(b'compatible', 'models/fencer.vmdl_c')
            self.assertEqual(len(calls), 3)

    def test_repeated_metadata_failure_is_bounded_and_cannot_use_stale_modeldoc(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            stale = folder/'content/models/fencer.vmdl'; stale.parent.mkdir(parents=True); stale.write_text('Old')
            def run(command, cwd, log):
                with log.open('a') as stream: stream.write(METADATA_ERROR)
            with patch.object(pipeline, 'run_tool', side_effect=run) as tool, \
                    patch.object(pipeline, 'compatible_extraction_model', return_value=b'compatible'):
                with self.assertRaisesRegex(ValueError, 'cannot decode'):
                    self.extract(folder)
            self.assertEqual(tool.call_count, 2)

    def test_zero_exit_without_fresh_output_does_not_accept_cached_file(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            stale = folder/'content/models/fencer.vmdl'; stale.parent.mkdir(parents=True); stale.write_text('Old')
            def run(command, cwd, log): log.write_text('No output\n')
            with patch.object(pipeline, 'run_tool', side_effect=run):
                with self.assertRaisesRegex(ValueError, 'did not write'):
                    self.extract(folder)


if __name__ == '__main__':
    unittest.main()
