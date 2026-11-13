"""soarm101_tools: the SO-ARM101 digital twin's commands as omni-kit-mcp tools.

Every entry of the soarm101-dt extension's tool table (isaac/exts/soarm101-dt/so_arm101_dt/
tool_table.py) is registered as ``soarm101.<name>`` and forwarded to the running extension
instance's ``_cmd_*`` method. Arbitrary code goes through the bridge's built-in ``run_python``.

Payload contract:
  * success: the handler's result dict, returned as-is (it carries ``"status": "success"``).
  * failure: the handler returned ``status != "success"`` or raised -> this tool raises
    ``ToolError(message)``, the bridge answers an error frame, and ``kit_mcp.client`` raises
    ``BridgeError`` with that message. Extra keys the handler returned (e.g. ``traceback``) ride
    along on the error frame.

Loaded per launch by linux-env/scripts/isaac/isaacsim_launch.sh:
    KIT_MCP_TOOL_PATHS=<repo>/isaac KIT_MCP_TOOL_MODULES=soarm101_tools
"""

import importlib.util
import os
import sys
import traceback

MCP_NAMESPACE = "soarm101"
#: Written against the bridge's request-admission contract (omni-kit-mcp docs/protocol.md).
#: No handler here pumps frames synchronously: the extension's async handlers await
#: ``next_update_async`` instead, so none carries the exclusive mark.
DOORWAY_ADMISSION = 1

_EXT_MODULE = "so_arm101_dt.extension"
_TABLE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "exts", "soarm101-dt", "so_arm101_dt", "tool_table.py")


def _load_table():
    """Read the extension's tool table by path: it is pure data, so this needs no Kit and does
    not depend on the extension having started before the bridge autoloads this package."""
    spec = importlib.util.spec_from_file_location("_soarm101_tool_table", _TABLE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MCP_TOOL_REGISTRY, mod.MCP_HANDLERS


def _instance():
    from omni_kit_mcp import ToolError

    ext = sys.modules.get(_EXT_MODULE)
    inst = ext.get_instance() if ext is not None and hasattr(ext, "get_instance") else None
    if inst is None:
        raise ToolError("soarm101-dt extension is not running (enable it with --enable soarm101-dt)")
    return inst


def _make_handler(tool, method):
    async def handler(**params):
        from omni_kit_mcp import ToolError

        fn = getattr(_instance(), method, None)
        if fn is None:
            raise ToolError(f"soarm101.{tool}: extension has no method {method}")
        try:
            result = fn(**params)
            if hasattr(result, "__await__"):
                result = await result
        except ToolError:
            raise
        except Exception as e:  # noqa: BLE001 - surfaced to the caller as a tool error
            raise ToolError(f"soarm101.{tool} raised {type(e).__name__}: {e}",
                            {"traceback": traceback.format_exc()})
        if not isinstance(result, dict):
            raise ToolError(f"soarm101.{tool} returned {type(result).__name__}, not a result dict")
        if result.get("status") != "success":
            details = {k: v for k, v in result.items() if k not in ("status", "message")}
            raise ToolError(str(result.get("message") or f"soarm101.{tool} failed"), details)
        return result

    handler.__name__ = tool
    handler.__qualname__ = tool
    return handler


def register(registrar):
    registry, handlers = _load_table()
    for tool, method in handlers.items():
        entry = registry.get(tool, {})
        registrar.tool(entry.get("description", tool),
                       dict(entry.get("parameters") or {}),
                       name=tool)(_make_handler(tool, method))
