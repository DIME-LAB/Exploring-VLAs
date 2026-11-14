"""Load the shared Isaac venv's coherent charset package before Kit changes paths."""
import sys

if sys.prefix == '/home/aaugus11/env_isaacsim61':
    try:
        import charset_normalizer
        from pathlib import Path
        expected = Path(sys.prefix)/'lib/python3.12/site-packages/charset_normalizer/__init__.py'
        if Path(charset_normalizer.__file__) != expected or charset_normalizer.__version__ != '3.3.2':
            raise RuntimeError('rtx6000 Isaac charset preload differs from qualified 3.3.2 package')
    except BaseException:
        # Python normally prints and ignores sitecustomize errors; this profile must refuse.
        import os
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(78)
