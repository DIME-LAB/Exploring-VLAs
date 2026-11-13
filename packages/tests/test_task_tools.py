import asyncio
import json
import sys
import types
import pytest
import soarm101_task_tools as tools


def test_bridge_tools_use_task_client_argv(tmp_path, monkeypatch):
    script = tmp_path / "client.py"
    script.write_text('import json,sys\nprint(json.dumps({"status":"success","args":sys.argv[1:]}))\n')
    monkeypatch.setenv("LEROBOT_RUN_TASK_COMMAND", json.dumps([sys.executable, str(script)]))
    result = asyncio.run(tools.run_task("pick 'blue'", 42.0, "base-ft"))
    assert result["args"] == ["--instruction", "pick 'blue'", "--max-duration", "42.0", "--model", "base-ft"]
    result = asyncio.run(tools.set_task_instruction("place red"))
    assert result["args"] == ["--update-instruction", "--instruction", "place red"]
    assert asyncio.run(tools.cancel_task())["args"] == ["--cancel"]
    with pytest.raises(ValueError):
        asyncio.run(tools.run_task("", 1))
    monkeypatch.delenv("LEROBOT_RUN_TASK_COMMAND")
    with pytest.raises(ValueError, match="LEROBOT_RUN_TASK_COMMAND"):
        asyncio.run(tools.cancel_task())


def test_optional_panel_routes_prompt_model_and_cancel_to_same_tools(monkeypatch):
    from soarm101_task_tools import panel
    fields, buttons, calls = [], {}, []
    class Widget:
        def __init__(self, label="", **kwargs):
            self.text = label
            self.frame = self
            self.model = self
            self.value = ""
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def set_value(self, value):
            self.value = value
        @property
        def as_string(self):
            return str(self.value)
        @property
        def as_float(self):
            return float(self.value)
    def field():
        value = Widget()
        fields.append(value)
        return value
    def button(label, clicked_fn):
        buttons[label] = clicked_fn
    ui = types.ModuleType("omni.ui")
    ui.Window = ui.VStack = ui.HStack = ui.Label = Widget
    ui.StringField = ui.FloatField = field
    ui.Button = button
    omni = types.ModuleType("omni")
    omni.ui = ui
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.ui", ui)
    async def run(*args):
        calls.append(("run", args))
        return {"outcome": "accepted"}
    async def update(*args):
        calls.append(("update", args))
        return {"outcome": "instruction_updated"}
    async def cancel():
        calls.append(("cancel", ()))
        return {"status": "success"}
    monkeypatch.setattr(panel, "run_task", run)
    monkeypatch.setattr(panel, "set_task_instruction", update)
    monkeypatch.setattr(panel, "cancel_task", cancel)
    async def check():
        panel.show()
        fields[0].set_value("pick blue")
        fields[1].set_value("base-ft")
        fields[2].set_value(35)
        buttons["Run task"]()
        buttons["Update prompt"]()
        buttons["Cancel and home"]()
        await asyncio.gather(*list(panel._panel[1]))
    asyncio.run(check())
    assert calls == [("run", ("pick blue", 35.0, "base-ft")), ("update", ("pick blue",)), ("cancel", ())]
