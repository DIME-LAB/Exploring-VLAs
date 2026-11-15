"""Main-thread Timeline pause acknowledgement without a nested physics update."""
import time


def pause_committed(timeline,wall_limit_s=1.,max_commits=3):
    """Flush deferred state using Kit's Timeline.commit(), then verify held state.

    Timeline.commit() applies pending state and invokes timeline callbacks on the
    main thread. It does not step the app/physics or advance simulation time.
    A failed acknowledgement grants no permission for native setters.
    """
    start=time.monotonic();timeline.pause();commits=0
    while timeline.is_playing():
        if commits>=max_commits or time.monotonic()-start>=wall_limit_s:
            raise TimeoutError('Timeline pause did not commit; native writes refused')
        timeline.commit();commits+=1
    elapsed=time.monotonic()-start
    if elapsed>wall_limit_s:raise TimeoutError('Timeline pause acknowledgement exceeded its wall deadline')
    return dict(held=True,commits=commits,wall_s=elapsed,physics_steps=0)
