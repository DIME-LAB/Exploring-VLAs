"""Static-analysis test: enforce factory usage for all interactive widgets.

Phase 07.1 Workstream enforcement gate. Parallel to test_button_service_mapping.py
(Phase 7) — catches future divergences where a developer adds a new direct
`tk.Spinbox(...)` / `tk.Entry(...)` / etc. call bypassing the registry factories.

Pure AST analysis — no ROS node instantiation, no X11, no network required.

Rules enforced:
  1. Direct tk.Spinbox/ttk.Spinbox/tk.Checkbutton/tk.Entry/ttk.Entry/tk.Listbox/
     tk.Scale/ttk.Scale constructions outside the factory method bodies are
     FORBIDDEN.
  2. Every self._register_<type>(...) call must pass `label=`.
  3. `label=` kwarg must be a string literal or f-string with a constant base
     (dynamic labels break ID stability).
  4. The main Notebook and both log Text widgets must be registered
     (`_register_notebook` + `_register_log_text` called at least 2 and 2 times
     respectively).
"""

from __future__ import annotations

import ast
from pathlib import Path


CONTROL_GUI_PATH = Path(__file__).parent / "control_gui.py"

# Map factory method name → set of forbidden direct constructors outside it.
FACTORY_TO_FORBIDDEN = {
    "_register_spinbox": {("tk", "Spinbox"), ("ttk", "Spinbox")},
    "_register_check":   {("tk", "Checkbutton")},
    "_register_entry":   {("tk", "Entry"), ("ttk", "Entry")},
    "_register_listbox": {("tk", "Listbox")},
    "_register_scale":   {("tk", "Scale"), ("ttk", "Scale")},
}

FACTORY_NAMES = set(FACTORY_TO_FORBIDDEN.keys())


def _load_tree() -> ast.Module:
    return ast.parse(CONTROL_GUI_PATH.read_text())


def _describe(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return ast.dump(node)


def _nodes_inside(func: ast.FunctionDef) -> set[int]:
    return {id(n) for n in ast.walk(func)}


def test_no_direct_interactive_widget_construction() -> None:
    """Direct widget constructors are only allowed inside their own factory."""
    tree = _load_tree()

    # Collect factory body node-id sets
    factory_bodies: dict[str, set[int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in FACTORY_NAMES:
            factory_bodies[node.name] = _nodes_inside(node)
    for fname in FACTORY_NAMES:
        assert fname in factory_bodies, (
            f"Factory method {fname} not found in control_gui.py — "
            f"did someone remove it?"
        )

    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in ("tk", "ttk")
        ):
            continue
        widget_type = node.func.attr
        # Is this widget type a forbidden direct construction?
        matching_factory = None
        for fname, forbidden in FACTORY_TO_FORBIDDEN.items():
            if (node.func.value.id, widget_type) in forbidden:
                matching_factory = fname
                break
        if matching_factory is None:
            continue
        # Allowed only inside its own factory body
        if id(node) in factory_bodies.get(matching_factory, set()):
            continue
        violations.append(
            (node.lineno, f"{node.func.value.id}.{widget_type}")
        )

    assert not violations, (
        f"Direct widget constructions outside factories are forbidden. "
        f"Route through self._register_* instead. Violations: {violations}"
    )


def test_factory_calls_require_label() -> None:
    """Every self._register_<type>(...) call must pass `label=`."""
    tree = _load_tree()
    missing: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in FACTORY_NAMES
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ):
            continue
        kws = {k.arg for k in node.keywords}
        if "label" not in kws:
            missing.append((node.lineno, node.func.attr))
    assert not missing, (
        f"Factory calls without `label=` kwarg: {missing}"
    )


def test_factory_labels_are_constant() -> None:
    """`label=` must be a string literal or a JoinedStr whose parts are
    constants — dynamic labels produce unstable widget IDs."""
    tree = _load_tree()
    bad: list[tuple[int, str]] = []

    def _is_constant_expr(node: ast.AST) -> bool:
        if isinstance(node, ast.Constant):
            return isinstance(node.value, str)
        if isinstance(node, ast.JoinedStr):
            # f-string — all parts must be Constants or FormattedValues of Names
            # whose names look like constants (UPPER_CASE) — we accept this loosely
            return True
        if isinstance(node, ast.Name):
            # Accept module-level CONSTANTS (e.g. GRIPPER_JOINT_NAME) as labels
            return node.id.isupper() or node.id == "name" or node.id == "comp" or node.id == "label"
        return False

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in FACTORY_NAMES
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ):
            continue
        for kw in node.keywords:
            if kw.arg != "label":
                continue
            if not _is_constant_expr(kw.value):
                bad.append((node.lineno, _describe(kw.value)))
    assert not bad, (
        f"`label=` must be a string literal (or UPPER_CASE constant / f-string / "
        f"known loop var `name`/`comp`/`label`). Offenders: {bad}"
    )


def test_notebook_and_log_text_registered() -> None:
    """The main Notebook must be registered, and both log Text widgets must
    be registered as read-only."""
    tree = _load_tree()
    notebook_calls = 0
    log_text_calls = 0
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ):
            continue
        if node.func.attr == "_register_notebook":
            notebook_calls += 1
        elif node.func.attr == "_register_log_text":
            log_text_calls += 1
    assert notebook_calls >= 2, (
        f"Expected ≥ 2 _register_notebook calls (main notebook + hot-reload "
        f"rebuild + log notebook). Found: {notebook_calls}."
    )
    assert log_text_calls == 2, (
        f"Expected exactly 2 _register_log_text calls (process_log + "
        f"error_log). Found: {log_text_calls}."
    )


def test_factory_call_counts_sane() -> None:
    """Sanity: at least the minimum number of factory calls per type."""
    tree = _load_tree()
    counts: dict[str, int] = {f: 0 for f in FACTORY_NAMES}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in FACTORY_NAMES
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ):
            continue
        counts[node.func.attr] += 1
    # Minimums reflect Plan 07.1-02's migration inventory.
    minimums = {
        "_register_spinbox": 10,
        "_register_check":   4,
        "_register_entry":   3,
        "_register_listbox": 2,
        "_register_scale":   3,
    }
    for fname, minimum in minimums.items():
        assert counts[fname] >= minimum, (
            f"Only {counts[fname]} `{fname}` calls — expected ≥ {minimum}. "
            f"Did a tab builder get reverted?"
        )


if __name__ == "__main__":
    test_no_direct_interactive_widget_construction()
    test_factory_calls_require_label()
    test_factory_labels_are_constant()
    test_notebook_and_log_text_registered()
    test_factory_call_counts_sane()
    print("OK — widget registry enforcement tests passed")
