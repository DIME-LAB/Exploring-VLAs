"""Retire Humble callback owners before native node/context destruction."""
import time


def retire_executor(executor,spin_thread,timeout=5.):
    if executor.shutdown(timeout_sec=timeout) is not True:
        raise RuntimeError('ROS executor shutdown did not drain callbacks')
    spin_thread.join(timeout=timeout)
    if spin_thread.is_alive():raise RuntimeError('ROS executor spin thread did not retire')
    # Humble's public shutdown drains active handlers, not its ThreadPoolExecutor.
    # A queued handler captured is_shutdown=False before shutdown started.
    pool=executor._executor
    pool.shutdown(wait=False,cancel_futures=True)
    deadline=time.monotonic()+timeout
    workers=tuple(pool._threads)
    for worker in workers:worker.join(timeout=max(0.,deadline-time.monotonic()))
    active=[worker.name for worker in workers if worker.is_alive()]
    if active:raise RuntimeError('ROS executor callback pool did not retire: '+repr(active))
    return dict(spin_retired=True,callback_threads_retired=len(workers))


def shutdown_context(rclpy,context):
    if rclpy.get_default_context() is context:rclpy.shutdown()
    else:rclpy.shutdown(context=context)
    context.destroy()
