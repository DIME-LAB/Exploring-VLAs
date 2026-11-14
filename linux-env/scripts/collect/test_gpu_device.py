"""GPU ordinal remapping, persistent per-card caches and cross-card memory refusal."""
import unittest
from pathlib import Path
from unittest.mock import patch
import gpu_device


class GpuDeviceTests(unittest.TestCase):
    def test_caches_are_persistent_per_box_and_card(self):
        first=gpu_device.cache_paths('GPU-aaaa','box-a');second=gpu_device.cache_paths('GPU-bbbb','box-a')
        self.assertNotEqual(first['root'],second['root'])
        self.assertTrue(first['shaders'].startswith('/mnt/a4500-backup/exploring-vlas/cache/isaac/'))
        with self.assertRaises(ValueError):gpu_device.cache_paths('GPU-aaaa','../other')

    def test_indexed_launcher_pins_cuda_by_uuid_instead_of_unsetting_it(self):
        import subprocess
        source=Path(__file__).resolve().parents[1].joinpath('isaac/isaacsim_launch.sh').read_text()
        self.assertNotIn('unset CUDA_VISIBLE_DEVICES',source)
        assignment=next(line.strip() for line in source.splitlines() if line.strip()=='export CUDA_VISIBLE_DEVICES="$gpu_uuid"')
        result=subprocess.run(['bash','-c','set -eu; gpu_uuid=GPU-aaaa; INSTANCE_ROOT=/assigned; '+assignment+'; printf "%s" "$CUDA_VISIBLE_DEVICES"'],capture_output=True,text=True,check=True)
        self.assertEqual(result.stdout,'GPU-aaaa')
