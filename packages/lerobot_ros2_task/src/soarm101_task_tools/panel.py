# Copyright 2026 Exploring-VLAs contributors. Licensed under Apache-2.0.
"""Load explicitly inside Kit after soarm101_task_tools is available."""
import asyncio
from . import run_task, set_task_instruction, cancel_task

_panel = None


def show():
    global _panel
    import omni.ui as ui
    window = ui.Window("LeRobot RunTask", width=420, height=230)
    tasks = set()
    with window.frame:
        with ui.VStack(spacing=6):
            ui.Label("Instruction")
            instruction = ui.StringField()
            ui.Label("Model key or pretrained path (empty uses node default)")
            model = ui.StringField()
            with ui.HStack():
                ui.Label("Maximum run seconds")
                duration = ui.FloatField()
                duration.model.set_value(60)
            status = ui.Label("Ready")
            def submit(awaitable):
                async def execute():
                    try:
                        result = await awaitable
                        status.text = str(result.get("outcome", result.get("status", "sent")))
                    except Exception as exc:
                        status.text = str(exc)
                task = asyncio.ensure_future(execute())
                tasks.add(task)
                task.add_done_callback(tasks.discard)
            ui.Button("Run task", clicked_fn=lambda: submit(run_task(
                instruction.model.as_string, duration.model.as_float, model.model.as_string)))
            ui.Button("Update prompt", clicked_fn=lambda: submit(set_task_instruction(instruction.model.as_string)))
            ui.Button("Cancel and home", clicked_fn=lambda: submit(cancel_task()))
    _panel = (window, tasks)
    return window
