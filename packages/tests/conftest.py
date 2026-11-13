import pytest
from fake_ros import install


@pytest.fixture
def ros(monkeypatch):
    import lerobot_ros2_context as shared
    assert not shared._sessions
    value = install(monkeypatch)
    yield value
    for session in list(shared._sessions.values()):
        session.close()
    shared._sessions.clear()
