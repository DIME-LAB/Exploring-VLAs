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
        assert three.node.context is not context
        assert three.node.context.starts == 1
    assert context.stops == 1


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


def test_cleanup_handle_access_after_context_stops(ros):
    lease = acquire()
    node = lease.node
    ros[1].shutdown()
    assert lease.node is node
    lease.close()
    assert node.destroyed
    with acquire() as next_lease:
        assert next_lease.node.context.ok()


def test_cyclone_requires_explicit_transport_config(ros, monkeypatch):
    monkeypatch.setenv("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
    monkeypatch.delenv("CYCLONEDDS_URI", raising=False)
    with pytest.raises(RuntimeError, match="linux-env/cyclonedds.xml"):
        acquire()
    assert ros[1].starts == 0


def test_closing_old_context_does_not_stop_replacement(ros):
    first = acquire()
    ros[0].shutdown()
    second = acquire()
    node = second.node
    first.close()
    assert node.context.ok()
    second.close()


def test_existing_context_does_not_need_environment_reinitialization(ros, monkeypatch):
    context = Context()
    context.running = True
    monkeypatch.setenv("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
    monkeypatch.delenv("CYCLONEDDS_URI", raising=False)
    with acquire(context=context, executor=Executor(context)) as lease:
        assert lease.node.context is context
    assert context.ok()
