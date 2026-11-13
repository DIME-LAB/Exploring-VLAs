# Copyright 2026 Exploring-VLAs contributors. Licensed under Apache-2.0.
"""Opt-in bridge tools. No changes to the deployed Isaac extension are needed."""
import asyncio
import json
import math
import os

MCP_NAMESPACE = "lerobot_task"
DOORWAY_ADMISSION = 1


async def _call(*args):
    command = json.loads(os.environ.get("LEROBOT_RUN_TASK_COMMAND", "[]"))
    if not command or not isinstance(command, list) or not all(isinstance(s, str) for s in command):
        raise ValueError("Set LEROBOT_RUN_TASK_COMMAND to a JSON argv for the sourced ROS task client")
    process = await asyncio.create_subprocess_exec(*command, *args, stdout=asyncio.subprocess.PIPE,
                                                 stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(process.communicate(), timeout=20)
    except BaseException:
        if process.returncode is None:
            process.terminate()
            await process.wait()
        raise
    if process.returncode:
        raise RuntimeError(err.decode()[-2000:])
    return json.loads(out.decode().splitlines()[-1])


async def run_task(instruction, max_duration=60.0, model=""):
    if not instruction.strip() or not math.isfinite(max_duration) or max_duration <= 0:
        raise ValueError("A nonempty instruction and positive finite duration are required")
    return await _call("--instruction", instruction, "--max-duration", str(max_duration), "--model", model)


async def set_task_instruction(instruction):
    if not instruction.strip():
        raise ValueError("Instruction must be nonempty")
    return await _call("--update-instruction", "--instruction", instruction)


async def cancel_task():
    return await _call("--cancel")


def register(registrar):
    registrar.tool("Home, run the selected LeRobot model, then return home", {
        "instruction": {"type": "string"}, "max_duration": {"type": "number", "default": 60},
        "model": {"type": "string", "default": ""}}, name="run_task")(run_task)
    registrar.tool("Change the active task instruction", {
        "instruction": {"type": "string"}}, name="set_task_instruction")(set_task_instruction)
    registrar.tool("Cancel the active task and return home", {}, name="cancel_task")(cancel_task)


def attach_to_soarm101(bridge):
    """Add the requested aliases to the existing SO-ARM tool owner explicitly."""
    from omni_kit_mcp import OwnerRegistrar
    if "soarm101_tools" not in bridge.get_owners():
        raise RuntimeError("Load soarm101_tools before attaching RunTask aliases")
    register(OwnerRegistrar(bridge, "soarm101_tools", "soarm101"))
    return ["soarm101.run_task", "soarm101.set_task_instruction", "soarm101.cancel_task"]
