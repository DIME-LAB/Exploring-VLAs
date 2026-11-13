import pytest
from fake_ros import Context, Executor
from lerobot_ros2_context import acquire


def test_shared_context_last_user_and_reconnect(ros):
    _, context = ros
    one, two = acquire(), acquire()
    node = one.node
    assert two.node is node
    assert context.starts == 1
    one.close()
    one.close()
    assert context.ok() and not node.destroyed
    two.close()
    assert node.destroyed and context.stops == 1
    with acquire() as three:
        assert three.node is not node
        assert context.starts == 2
    assert context.stops == 2


def test_external_context_and_executor_are_preserved(ros):
    context = Context()
    context.running = True
    executor = Executor(context)
    one = acquire(context=context, executor=executor)
    two = acquire(context=context)
    assert one.node is two.node and len(executor.nodes) == 1
    one.close()
    two.close()
    assert executor.nodes == []
    assert context.ok() and executor.stops == context.stops == 0


def test_initialized_default_context_is_not_stopped(ros):
    _, context = ros
    context.running = True
    with acquire() as lease:
        assert lease.node.context is context
    assert context.ok() and context.starts == context.stops == 0


def test_executor_context_mismatch_and_conflicting_executor(ros):
    _, context = ros
    with pytest.raises(ValueError):
        acquire(context=context, executor=Executor(Context()))
    with acquire():
        with pytest.raises(ValueError):
            acquire(context=context, executor=Executor(context))


def test_closed_lease_and_callback_failure(ros):
    lease = acquire()
    lease._session.error = ValueError("bad message")
    with pytest.raises(RuntimeError, match="callback failed"):
        lease.check()
    lease.close()
    with pytest.raises(RuntimeError, match="closed"):
        lease.check()
