"""Static-analysis test: enforce 1:1 Button→_cmd_*/_*_btn_* mapping in control_gui.py.

Phase 7 Workstream A enforcement gate. Catches the Drop-Refresh-class bug where
a Button's command= was an inline lambda containing logic, diverging from any
auto-registered `_cmd_*` service.

This test parses control_gui.py with `ast` — it does NOT import the module or
instantiate the ROS node. No network, no X11, no ROS2 required.

Rules enforced:
  1. Every button is constructed via `self._register_button(...)`.
     No bare `tk.Button(...)` / `ttk.Button(...)` outside the `_register_button`
     helper itself.
  2. Every `self._register_button(...)` call's `command=` kwarg is a bare
     method reference of the form `self.<name>` where `<name>` begins with
     `_cmd_` or matches `_*_btn_*`. Inline `command=lambda ...` bodies are
     forbidden anywhere in the file.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path


CONTROL_GUI_PATH = Path(__file__).parent / "control_gui.py"

BTN_WRAPPER_RE = re.compile(r"^_[a-z][a-z0-9_]*_btn_[a-z0-9_]+$")


def _load_tree() -> tuple[ast.Module, str]:
    source = CONTROL_GUI_PATH.read_text()
    return ast.parse(source), source


def _collect_method_names(tree: ast.Module) -> tuple[set[str], set[str]]:
    cmd_methods: set[str] = set()
    wrapper_methods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_cmd_"):
                cmd_methods.add(node.name)
            elif BTN_WRAPPER_RE.match(node.name):
                wrapper_methods.add(node.name)
    return cmd_methods, wrapper_methods


def _is_self_attr(expr: ast.AST) -> bool:
    return (
        isinstance(expr, ast.Attribute)
        and isinstance(expr.value, ast.Name)
        and expr.value.id == "self"
    )


def _describe(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return ast.dump(node)


def test_every_button_binds_to_cmd_or_wrapper() -> None:
    """Primary enforcement test — one pass over the AST, four assertions.

    Deliberately monolithic so a single failing assertion yields one clear
    message naming the offending line and button text. Splitting into smaller
    tests would fragment the diagnosis.
    """
    tree, _source = _load_tree()
    cmd_methods, wrapper_methods = _collect_method_names(tree)

    # Identify the single permitted direct tk.Button call inside _register_button.
    register_button_funcs = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_register_button"
    ]
    assert len(register_button_funcs) == 1, (
        f"expected exactly one _register_button definition in control_gui.py, "
        f"found {len(register_button_funcs)}"
    )
    register_button_node_ids = {
        id(n) for n in ast.walk(register_button_funcs[0])
    }

    # Helper clusters like _build_arm_btn_row / _build_gripper_btn_row take
    # reset_cmd / randomize_cmd as parameters and pass them to _register_button.
    # Their _register_button calls are trusted (linked to a _cmd_* at call site).
    helper_cluster_names = {'_build_arm_btn_row', '_build_gripper_btn_row'}
    helper_cluster_node_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in helper_cluster_names:
            helper_cluster_node_ids.update(id(n) for n in ast.walk(node))

    direct_button_calls: list[tuple[int, str]] = []
    inline_lambda_sites: list[tuple[int, str]] = []
    register_button_calls: list[ast.Call] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Direct tk.Button / ttk.Button constructions outside the helper
        if isinstance(node.func, ast.Attribute) and node.func.attr == "Button":
            base = node.func.value
            if isinstance(base, ast.Name) and base.id in ("tk", "ttk"):
                if id(node) not in register_button_node_ids:
                    direct_button_calls.append(
                        (node.lineno, _describe(node.func))
                    )
        # self._register_button(...) calls
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "_register_button"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ):
            register_button_calls.append(node)
        # command= lambda anywhere in the file
        for kw in getattr(node, "keywords", []) or []:
            if kw.arg == "command" and isinstance(kw.value, ast.Lambda):
                # Permitted: command=lambda val: ... on a tk.Scale that captures
                # the widget's position argument, because tkinter's Scale passes
                # a positional argument to command. Detect by checking whether
                # the enclosing call's function attribute ends with 'Scale'.
                parent_is_scale = (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("Scale", "_register_scale")
                )
                if not parent_is_scale:
                    inline_lambda_sites.append(
                        (kw.value.lineno, _describe(kw.value))
                    )

    # 1. No direct tk.Button / ttk.Button outside the helper.
    assert not direct_button_calls, (
        f"Every button must go through self._register_button. Found direct "
        f"tk.Button/ttk.Button constructions at: {direct_button_calls}"
    )

    # 2. No inline command=lambda anywhere (except on Scales, which tkinter
    #    forces to pass a positional argument).
    assert not inline_lambda_sites, (
        f"Inline command=lambda bindings are forbidden on non-Scale widgets. "
        f"Found: {inline_lambda_sites}"
    )

    # 3. Every _register_button call's command= kwarg must be self.<name>
    #    where <name> is a _cmd_* method OR a _*_btn_* wrapper.
    bad_commands: list[tuple[int, str]] = []
    missing_commands: list[int] = []
    for call in register_button_calls:
        # Calls inside helper clusters (_build_arm_btn_row, etc.) are trusted —
        # their command= values are parameters linked to _cmd_* at the call site.
        if id(call) in helper_cluster_node_ids:
            continue
        cmd_kw = next(
            (k for k in call.keywords if k.arg == "command"), None,
        )
        if cmd_kw is None:
            missing_commands.append(call.lineno)
            continue
        if not _is_self_attr(cmd_kw.value):
            bad_commands.append((call.lineno, _describe(cmd_kw.value)))
            continue
        attr_name = cmd_kw.value.attr
        if attr_name not in cmd_methods and attr_name not in wrapper_methods:
            bad_commands.append((call.lineno, f"self.{attr_name}"))

    assert not missing_commands, (
        f"self._register_button calls without a command= kwarg at lines: "
        f"{missing_commands}"
    )
    assert not bad_commands, (
        f"self._register_button commands must be a bare self._cmd_* or "
        f"self._*_btn_* reference. Offenders: {bad_commands}\n"
        f"Known _cmd_ methods: {sorted(cmd_methods)}\n"
        f"Known _*_btn_* wrappers: {sorted(wrapper_methods)}"
    )

    # 4. Sanity: at least 10 buttons must be registered (catches truncated
    #    parsing or accidental mass-deletion).
    assert len(register_button_calls) >= 10, (
        f"Expected at least 10 _register_button calls — found "
        f"{len(register_button_calls)}. Did the audit transformation get reverted?"
    )


def test_button_text_is_literal_string() -> None:
    """Secondary guard: button `text=` kwargs must be string literals, not
    dynamic expressions. Makes the dump_services table readable."""
    tree, _source = _load_tree()
    dynamic_texts: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_register_button"
        ):
            continue
        text_kw = next((k for k in node.keywords if k.arg == "text"), None)
        if text_kw is None:
            dynamic_texts.append((node.lineno, "missing text= kwarg"))
            continue
        if not isinstance(text_kw.value, ast.Constant) or not isinstance(
            text_kw.value.value, str
        ):
            dynamic_texts.append((node.lineno, _describe(text_kw.value)))
    assert not dynamic_texts, (
        f"self._register_button text= must be a string literal. Offenders: "
        f"{dynamic_texts}"
    )


if __name__ == "__main__":
    # Allow running via `python3 test_button_service_mapping.py`
    test_every_button_binds_to_cmd_or_wrapper()
    test_button_text_is_literal_string()
    print("OK — button↔service mapping test passed")
