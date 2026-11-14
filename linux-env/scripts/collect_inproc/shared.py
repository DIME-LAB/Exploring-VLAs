"""Resolve the repository's existing, transport-independent collection modules."""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
for directory in (REPO / 'linux-env/scripts/collect',
                  REPO / 'vla_SO-ARM101/src/so_arm101_control'):
    sys.path.insert(0, str(directory))
