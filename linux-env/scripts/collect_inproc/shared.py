"""Resolve the repository's existing, transport-independent collection modules."""
from pathlib import Path
import sys
import os

REPO = Path(__file__).resolve().parents[3]
for directory in (REPO / 'linux-env/scripts/collect',
                  REPO / 'vla_SO-ARM101/src/so_arm101_control'):
    sys.path.insert(0, str(directory))


def app_config():
    runtime=Path(os.environ['SOARM_INSTANCE_ROOT'])
    # SimulationApp checks process argv before adding its default --portable.
    # Our CLI is already parsed; keep its default cache out of the shared install.
    sys.argv=[sys.argv[0],'--portable-root',str(runtime/'portable')]
    return dict(headless=True,hide_ui=True,active_gpu=int(os.environ['SOARM_RENDERER_GPU_INDEX']),
        physics_gpu=0,multi_gpu=False,disable_viewport_updates=True,fast_shutdown=True,
        limit_cpu_threads=6,
        extra_args=['--/log/file='+os.environ['SOARM_CURRENT_KIT_LOG'],
            '--/app/settings/persistent=false',
            '--/rtx/shaderDb/shaderCachePath='+os.environ['SOARM_SHADER_CACHE_DIR'],
            '--/plugins/carb.tasking.plugin/threadCount=6','--/persistent/physics/numThreads=4',
            '--/persistent/simulation/minFrameRate=30',
            '--/exts/isaacsim.core.throttling/enable_async=false',
            '--/exts/omni.services.transport.server.http/port='+os.environ.get('SOARM_HTTP_PORT','8782'),
            '--/exts/omni.services.transport.server.http/allow_port_range=false'])


def publish_exit_ready(root,certificate,passed):
    """The child is drained; its owner must still prove process cleanup."""
    import hashlib
    from storage import atomic_json
    summary=root/'summary.json'
    if passed and not summary.exists():raise RuntimeError('Successful worker has no summary')
    atomic_json(root/'worker-exit-ready.json',dict(version=1,passed=passed,
        invocation=certificate['invocation'],producer_pid=os.getpid(),
        summary_sha256=hashlib.sha256(summary.read_bytes()).hexdigest() if summary.exists() else None))
