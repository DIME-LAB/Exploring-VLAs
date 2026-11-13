"""Contain callback failures at the ROS executor boundary."""
import functools
import inspect


def safe_callback(callback, on_error, service=False):
    def failed(exc, args):
        # Diagnostics/hooks must not turn a contained failure into a second
        # uncaught exception on the executor thread.
        try:
            on_error(getattr(callback, '__qualname__', repr(callback)), exc)
        except Exception:
            pass
        if service and len(args) >= 2:
            response = args[1]
            if hasattr(response, 'success'):
                response.success = False
            if hasattr(response, 'message'):
                response.message = 'ROS callback failed: '+str(exc)
            return response
        return None

    if inspect.iscoroutinefunction(callback):
        @functools.wraps(callback)
        async def guarded(*args, **kwargs):
            try:
                return await callback(*args, **kwargs)
            except Exception as exc:
                return failed(exc, args)
    else:
        @functools.wraps(callback)
        def guarded(*args, **kwargs):
            try:
                return callback(*args, **kwargs)
            except Exception as exc:
                return failed(exc, args)
    return guarded
