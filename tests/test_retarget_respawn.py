"""Countdown source replacement must preserve gameplay clips and authoring flags."""
import importlib
from pathlib import Path
import sys
import types
import unittest

package = types.ModuleType('respawn_tests')
package.__path__ = [str(Path(__file__).parents[1])]
sys.modules[package.__name__] = package
respawn = importlib.import_module(package.__name__ + '.retarget_respawn')


class RespawnModelDoc(unittest.TestCase):
    def test_only_countdown_sources_change_including_nested_sequences(self):
        source = '''{ _class = "RootNode" children = [
 { _class = "AnimationList" children = [
  { _class = "AnimBindPose" name = "bind" },
  { _class = "AnimFile" name = "primary_idle" source_filename = "idle.dmx" },
  { _class = "AnimFile" name = "respawn_countdown_idle" source_filename = "respawn.dmx"
    looping = true fade_in_time = 0.2 children = [ { _class = "AnimEvent" event_frame = 0 } ] },
  { _class = "AnimGroup" name = "countdown" children = [
   { _class = "AnimFile" name = "rebirth_countdown_idle" source_filename = "rebirth.dmx" },
  ] },
 ] },
] }'''
        changed, names = respawn.respawn_modeldoc(source, 'models/idle_frame_0.dmx')
        self.assertEqual(names, ['respawn_countdown_idle', 'rebirth_countdown_idle'])
        self.assertEqual(changed, source.replace('"respawn.dmx"', '"models/idle_frame_0.dmx"')
                         .replace('"rebirth.dmx"', '"models/idle_frame_0.dmx"'))
        self.assertEqual(respawn.respawn_modeldoc(changed, 'models/idle_frame_0.dmx')[0], changed)

    def test_heroes_without_countdown_are_unchanged(self):
        source = '{ _class = "AnimFile" name = "idle" source_filename = "idle.dmx" }'
        self.assertEqual(respawn.respawn_modeldoc(source, 'fallback.dmx'), (source, []))

    def test_missing_source_is_reported(self):
        with self.assertRaisesRegex(ValueError, 'Countdown animation has no source'):
            respawn.respawn_modeldoc('{ _class = "AnimFile" name = "respawn_countdown_idle" }', 'idle.dmx')


if __name__ == '__main__':
    unittest.main()
