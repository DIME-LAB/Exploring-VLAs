#!/usr/bin/env python3
"""Tk adapter for the shared SO-ARM101 motion node."""
import tkinter as tk
from tkinter import ttk
from so_arm101_control.pick_place import *

from so_arm101_control.scene_transactions import scene_transaction

class SOArm101ControlGUI(PickPlaceNode):
    """ROS2 node with embedded Tkinter GUI for SO-ARM101 control."""

    def __init__(self):
        super().__init__('so_arm101_control_gui')
        self.real_js_sub = self.create_subscription(
            JointState, 'real_joint_states', self._real_js_callback, 10)
        self.ext_cmd_sub = self.create_subscription(
            JointState, 'joint_commands', self._ext_cmd_callback, 10)
        # GUI
        self.running = True
        # Button registry — populated by _register_button, audited by _cmd_dump_services.
        # Every entry: {'text', 'command_name', 'tab', 'section', 'widget_class'}.
        self._button_registry = []
        # Widget registry — populated by _register_spinbox/_check/_entry/_listbox/_scale/
        # _notebook/_log_text factories. Audited by _srv_list_widgets.
        # Entries: {id: {'type', 'widget' (WeakRef), 'var', 'tab', 'section', 'label', 'writable'}}
        self._widget_registry: dict = {}
        self._widget_registry_lock = threading.Lock()
        self._widget_registry_rebuilding = True
        self._setup_gui_thread()

        # --- Debug services ---
        # Auto-register: any method named _cmd_* → ~/cmd_name (Trigger)
        # Manual services: _srv_* stay registered individually below
        self._debug_services = []
        for name in sorted(dir(self)):
            if name.startswith('_cmd_'):
                srv_name = name[5:]
                cb = self._make_trigger_callback(name)
                srv = self.create_service(Trigger, f'~/{srv_name}', cb)
                self._debug_services.append(srv)
                self.get_logger().info(f'  service: ~/{srv_name}')

        # Manual services (read/write UI fields directly)
        for name, cb in [
            ('get_joint_positions', self._srv_get_joint_positions),
            ('get_ik_target', self._srv_get_ik_target),
            ('set_ik_target', self._srv_set_ik_target),
            ('get_ee_pose', self._srv_get_ee_pose),
            ('get_tcp_pose', self._srv_get_tcp_pose),
            ('get_log', self._srv_get_log),
            ('list_commands', self._srv_list_commands),
            ('dump_services', self._srv_dump_services),
            ('get_widget_state', self._srv_get_widget_state),
            ('screenshot', self._srv_screenshot),
            ('list_widgets', self._srv_list_widgets),
            ('get_widget_value', self._srv_get_widget_value),
            ('set_widget_value', self._srv_set_widget_value),
        ]:
            self._debug_services.append(
                self.create_service(Trigger, f'~/{name}', cb))
            self.get_logger().info(f'  service: ~/{name}')

        self.get_logger().info('SO-ARM101 Control GUI initialized')

    def _dispatch_motion(self, callback):
        if self._gui_ready:
            self.root.after(0, callback)

    def _prepare_cycle(self):
        ready = threading.Event()
        def refresh():
            self._cmd_qs_refresh_all()
            self.root.after(1250, ready.set)
        self.root.after(0, refresh)
        if not ready.wait(timeout=5.0):
            raise TimeoutError('Quickstart refresh did not complete on the GUI thread')

    def _select_grasp_object(self, name):
        self._qs_sync_grasp_listbox(name)

    # ------------------------------------------------------------------
    # Async service helper (thread-safe future wait)
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Button factory — central registration for audit + enforcement
    # ------------------------------------------------------------------
    # Every tk.Button / ttk.Button in this file MUST go through this helper.
    # The `command=` kwarg MUST be a bare method reference:
    #   - self._cmd_*         (auto-registers as a ~/cmd Trigger service)
    #   - self._*_btn_*       (thin UI-state wrapper that forwards to a _cmd_*)
    # Inline `command=lambda: ...` bodies are forbidden — enforced by
    # test_button_service_mapping.py (Phase 7 workstream A).

    def _register_button(self, parent, *, text, command, tab=None, section=None,
                         bg='#b0b0b0', fg='#1a1a1a', **kwargs):
        """Central button factory. Records the button in self._button_registry
        for _cmd_dump_services audit, then creates a styled tk.Button."""
        cmd_name = getattr(command, '__name__', repr(command))
        self._button_registry.append({
            'text': text,
            'command_name': cmd_name,
            'tab': tab,
            'section': section,
            'widget_class': 'tk.Button',
        })
        return tk.Button(parent, text=text, command=command,
                         bg=bg, fg=fg, **kwargs)

    # ------------------------------------------------------------------
    # Widget factory helpers (Phase 07.1) — every interactive widget
    # MUST go through these so _srv_list_widgets / _srv_get_widget_value /
    # _srv_set_widget_value can see them. Audited by test_widget_registry.py.
    # ------------------------------------------------------------------

    def _widget_registry_add(self, label, wtype, widget, var,
                             tab, section, writable, choices=()):
        """Insert a widget into the registry. Duplicate labels are tolerated
        at insertion time; _finalize_widget_registry resolves collisions with
        @tab suffixes after all builders have run."""
        entry = {
            'type': wtype,
            'widget': weakref.ref(widget),
            'var': var,
            'tab': tab,
            'section': section,
            'label': label,
            'writable': writable,
            'choices': tuple(choices),
        }
        with self._widget_registry_lock:
            if label in self._widget_registry and tab:
                resolved_id = f'{label}@{tab}'
            else:
                resolved_id = label
            self._widget_registry[resolved_id] = entry

    def _finalize_widget_registry(self):
        """Called after all tab builders complete. Scans for label collisions
        and retroactively adds @tab suffixes so every id is unique."""
        with self._widget_registry_lock:
            by_label: dict = {}
            for rid, entry in list(self._widget_registry.items()):
                by_label.setdefault(entry['label'], []).append(rid)
            for label, rids in by_label.items():
                if len(rids) == 1:
                    continue
                for rid in rids:
                    if '@' in rid:
                        continue
                    entry = self._widget_registry[rid]
                    tab = entry.get('tab') or '?'
                    new_id = f'{label}@{tab}'
                    if new_id == rid:
                        continue
                    self._widget_registry[new_id] = entry
                    del self._widget_registry[rid]
            self._widget_registry_rebuilding = False

    def _register_spinbox(self, parent, *, label, textvariable, from_, to,
                          tab=None, section=None, **kwargs):
        """tk.Spinbox (or ttk.Spinbox via _use_ttk=True). Writable; set coerces to float."""
        use_ttk = kwargs.pop('_use_ttk', False)
        cls = ttk.Spinbox if use_ttk else tk.Spinbox
        widget = cls(parent, textvariable=textvariable,
                     from_=from_, to=to, **kwargs)
        self._widget_registry_add(label, 'Spinbox', widget, textvariable,
                                  tab, section, writable=True)
        return widget

    def _register_check(self, parent, *, label, variable,
                        tab=None, section=None, **kwargs):
        """tk.Checkbutton. Label doubles as the button's visible text."""
        widget = tk.Checkbutton(parent, text=label, variable=variable, **kwargs)
        self._widget_registry_add(label, 'Checkbutton', widget, variable,
                                  tab, section, writable=True)
        return widget

    def _register_entry(self, parent, *, label, textvariable,
                        tab=None, section=None, **kwargs):
        """tk.Entry. `label` is explicit (no sibling-Label heuristic)."""
        widget = tk.Entry(parent, textvariable=textvariable, **kwargs)
        self._widget_registry_add(label, 'Entry', widget, textvariable,
                                  tab, section, writable=True)
        return widget

    def _register_listbox(self, parent, *, label, tab=None, section=None,
                          **kwargs):
        """tk.Listbox. Writable; set expects a non-negative row index."""
        widget = tk.Listbox(parent, **kwargs)
        self._widget_registry_add(label, 'Listbox', widget, None,
                                  tab, section, writable=True)
        return widget

    def _register_combobox(self, parent, *, label, textvariable, values,
                           tab=None, section=None, **kwargs):
        """ttk.Combobox (readonly). User picks one of `values`; current
        selection is bound to `textvariable` (StringVar)."""
        widget = ttk.Combobox(
            parent, textvariable=textvariable, values=values,
            state='readonly', **kwargs)
        self._widget_registry_add(label, 'Combobox', widget, textvariable,
                                  tab, section, writable=True)
        return widget

    def _register_scale(self, parent, *, label, variable, from_, to,
                        tab=None, section=None, **kwargs):
        """tk.Scale (or ttk.Scale via _use_ttk=True). Writable; coerces to float."""
        use_ttk = kwargs.pop('_use_ttk', False)
        cls = ttk.Scale if use_ttk else tk.Scale
        widget = cls(parent, variable=variable, from_=from_, to=to, **kwargs)
        self._widget_registry_add(label, 'Scale', widget, variable,
                                  tab, section, writable=True)
        return widget

    def _register_notebook(self, widget, *, label='tab'):
        """Register an existing ttk.Notebook under a reserved id.
        Caller constructs the notebook — we don't apply styling."""
        self._widget_registry_add(label, 'Notebook', widget, None,
                                  tab=None, section=None, writable=True)
        return widget

    def _register_log_text(self, widget, *, label):
        """Register a tk.Text widget as read-only log output."""
        self._widget_registry_add(label, 'Text', widget, None,
                                  tab=None, section=None, writable=False)
        return widget

    # ------------------------------------------------------------------
    # Shared UI cluster: Reset Arm / Randomize / Reset Gripper / Open / Close
    # duplicated between FK and IK tabs (Phase 7 Plan 07-03 / D-16..D-18).
    # ------------------------------------------------------------------

    def _build_arm_btn_row(self, parent, *, tab, reset_cmd):
        """Build the `[Reset Arm] [Reset Grasp Home] [Randomize]` button row.

        tab: 'FK' or 'IK' — used for dump_services tagging.
        reset_cmd: which command 'Reset Arm' invokes (different on FK vs IK).
        """
        self._register_button(parent, text='Reset Arm', tab=tab, section='Arm',
                              command=reset_cmd).pack(side=tk.LEFT, padx=5)
        if tab == 'FK':
            self._register_button(parent, text='Reset Grasp Home', tab=tab,
                                  section='Arm',
                                  command=self._cmd_reset_grasp_home).pack(
                side=tk.LEFT, padx=5)
        randomize_cmd = (self._cmd_randomize_arm if tab == 'FK'
                         else self._cmd_ik_randomize)
        self._register_button(parent, text='Randomize', tab=tab, section='Arm',
                              command=randomize_cmd).pack(side=tk.LEFT, padx=5)

    def _build_gripper_btn_row(self, parent, *, tab):
        """Build `[Reset Gripper] [Open] [Close]` row into `parent`."""
        self._register_button(parent, text='Reset Gripper', tab=tab,
                              section='Gripper',
                              command=self._cmd_gripper_zero).pack(
            side=tk.LEFT, padx=5)
        self._register_button(parent, text='Open', tab=tab,
                              section='Gripper',
                              command=self._cmd_gripper_open).pack(
            side=tk.LEFT, padx=5)
        self._register_button(parent, text='Close', tab=tab,
                              section='Gripper',
                              command=self._cmd_gripper_close).pack(
            side=tk.LEFT, padx=5)

    # ------------------------------------------------------------------
    # Debug service helpers
    # ------------------------------------------------------------------

    def _make_trigger_callback(self, method_name):
        """Factory: returns a Trigger callback that dispatches to tkinter thread.

        If the _cmd_* method sets self._motion_event before returning,
        the callback waits for that event too (up to 30s), so the service
        response is deferred until the full motion sequence completes.
        """
        def _callback(request, response):
            done_event = threading.Event()
            result = {'ok': True, 'msg': ''}

            def _run():
                try:
                    # Clear any stale motion event / error / status
                    self._motion_event = None
                    self._cmd_error = None
                    self._last_motion_status = None
                    # Tag the next trajectory dump with this cmd's name so
                    # post-hoc log analysis can match dumps to commands.
                    # _cmd_foo → tag 'foo'. Anything _execute_full_trajectory
                    # writes from here until the next _cmd_* gets this tag.
                    if method_name.startswith('_cmd_'):
                        self._last_motion_tag = method_name[5:]
                    getattr(self, method_name)()
                    # Check if the command signaled failure synchronously
                    # (geometric IK reject, missing service, etc). Async
                    # motion verdict is read later after _motion_event fires.
                    if self._cmd_error:
                        result['ok'] = False
                        result['msg'] = self._cmd_error
                    else:
                        # Placeholder — overwritten below by motion verdict
                        # if _motion_event was registered. For commands that
                        # don't set _motion_event (pure sync cmds), this
                        # stands as the final message.
                        result['msg'] = f'{method_name} dispatched'
                except Exception as e:
                    result['ok'] = False
                    result['msg'] = str(e)
                finally:
                    done_event.set()

            if getattr(self, '_gui_ready', False):
                self.root.after(0, _run)
            else:
                result['ok'] = False
                result['msg'] = 'GUI not available'
                done_event.set()

            # Wait for the _cmd_* method to return (interruptible by shutdown)
            while not done_event.is_set() and self.running:
                done_event.wait(timeout=0.5)

            # If the command registered a motion event, wait for motion too.
            # Timeout prevents permanent hang if async callback chain breaks.
            motion_evt = getattr(self, '_motion_event', None)
            motion_registered = motion_evt is not None
            motion_timed_out = False
            if motion_registered:
                elapsed = 0.0
                timeout_s = 60.0
                while not motion_evt.is_set() and self.running and elapsed < timeout_s:
                    motion_evt.wait(timeout=0.5)
                    elapsed += 0.5
                if not motion_evt.is_set():
                    motion_timed_out = True
                    result['ok'] = False
                    result['msg'] = f'Motion timed out after {timeout_s:.0f}s'
                self._motion_event = None

            # Read authoritative motion verdict. Motion primitives set
            # _last_motion_status right before on_complete_event.set(),
            # so by the time we get here it reflects the actual outcome.
            if motion_registered and not motion_timed_out:
                status = getattr(self, '_last_motion_status', None)
                if status is None:
                    # Primitive didn't populate status — treat as failure.
                    # Every on_complete_event.set() site MUST write status.
                    result['ok'] = False
                    result['msg'] = (f'{method_name}: motion completed but '
                                     'no _last_motion_status was reported')
                else:
                    result['ok'] = bool(status.get('ok', False))
                    msg = status.get('msg') or status.get('outcome') \
                        or 'unknown'
                    result['msg'] = f'{method_name}: {msg}'

            # Check for deferred errors (set by background threads after _run returned)
            cmd_err = getattr(self, '_cmd_error', None)
            if cmd_err:
                result['ok'] = False
                result['msg'] = cmd_err
                self._cmd_error = None

            response.success = result['ok']
            response.message = result['msg']
            return response
        return _callback

    def _srv_list_commands(self, request, response):
        """Return all available _cmd_* command names for agent discovery."""
        commands = sorted(n[5:] for n in dir(self) if n.startswith('_cmd_'))
        response.success = True
        response.message = ', '.join(commands)
        return response

    def _srv_get_widget_state(self, request, response):
        """Return the runtime state of a widget from _button_registry.

        Pass the widget id via the `ik_target` parameter. Matches on:
          - Button text (e.g. 'Reset Arm') — first match wins across tabs.
          - Bound method name (e.g. '_cmd_zero_arm', '_ik_btn_set_joints').
          - 'text@tab' disambiguator (e.g. 'Reset Arm@FK').

        Returns JSON-like `key=value, ...` pairs in response.message:
          id=_cmd_zero_arm, text=Reset Arm, tab=FK, section=Arm,
          enabled=true, visible=true, state=normal

        Usage:
          ros2 param set /so_arm101_control_gui ik_target 'Reset Arm@FK'
          ros2 service call /so_arm101_control_gui/get_widget_state std_srvs/srv/Trigger {}
        """
        hint = self.get_parameter('ik_target').get_parameter_value().string_value.strip()
        if not hint:
            response.success = False
            response.message = ('no widget id. Set param first: '
                                "ros2 param set /so_arm101_control_gui "
                                "ik_target '<button text or _cmd_ name>'")
            return response

        # Allow 'text@tab' disambiguation
        qualifier = None
        if '@' in hint:
            hint, qualifier = hint.rsplit('@', 1)

        match = None
        for entry in self._button_registry:
            if qualifier and entry.get('tab') != qualifier:
                continue
            if entry.get('text') == hint or entry.get('command_name') == hint:
                match = entry
                break

        if match is None:
            response.success = False
            response.message = (f'widget not found: {hint!r}'
                                + (f' (tab={qualifier!r})' if qualifier else ''))
            return response

        # Walk the tkinter tree to find the live widget by recreating the
        # lookup: registry stores text+tab+section, not the widget ref (by
        # design — avoids WeakRef bookkeeping for a GUI that doesn't churn).
        # Find the button by scanning winfo_children recursively for a
        # tk.Button whose text matches.
        def _find_button(parent, text, tab_hint=None):
            for child in parent.winfo_children():
                if isinstance(child, tk.Button) and child.cget('text') == text:
                    return child
                found = _find_button(child, text, tab_hint)
                if found is not None:
                    return found
            return None

        widget = _find_button(self.root, match['text'])
        if widget is None:
            response.success = False
            response.message = (f'widget {match["text"]!r} registered but not '
                                'currently in the tree — did the tab rebuild?')
            return response

        try:
            state = str(widget.cget('state'))
        except tk.TclError:
            state = 'unknown'
        try:
            enabled = state != 'disabled'
        except Exception:
            enabled = True
        try:
            visible = bool(widget.winfo_viewable())
        except Exception:
            visible = True

        parts = [
            f'id={match.get("command_name")}',
            f'text={match.get("text")}',
            f'tab={match.get("tab") or "-"}',
            f'section={match.get("section") or "-"}',
            f'enabled={"true" if enabled else "false"}',
            f'visible={"true" if visible else "false"}',
            f'state={state}',
        ]
        response.success = True
        response.message = ', '.join(parts)
        return response

    def _srv_screenshot(self, request, response):
        """Save a PNG of the main GUI window; return its path + dimensions.

        Pass the desired output path via the `ik_target` parameter. When
        empty, defaults to /tmp/so_arm101_control_gui.png.

        Uses xdotool + ImageMagick `import` for reliability across tk
        versions. Falls back to `import -window root` if xdotool fails.

        Usage:
          ros2 param set /so_arm101_control_gui ik_target '/tmp/gui.png'
          ros2 service call /so_arm101_control_gui/screenshot std_srvs/srv/Trigger {}

        Response message format: path=/tmp/gui.png, width=1280, height=720
        """
        import subprocess
        target = self.get_parameter('ik_target').get_parameter_value().string_value.strip()
        if not target:
            target = '/tmp/so_arm101_control_gui.png'

        # Resolve window id via xdotool (match the tkinter window title)
        title = self.root.title() if getattr(self, 'root', None) else 'SO-ARM101'
        window_id = None
        try:
            out = subprocess.check_output(
                ['xdotool', 'search', '--name', title],
                encoding='utf-8', timeout=3)
            ids = [line.strip() for line in out.splitlines() if line.strip()]
            window_id = ids[0] if ids else None
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                FileNotFoundError):
            window_id = None

        try:
            if window_id:
                subprocess.check_call(
                    ['import', '-window', window_id, target], timeout=5)
            else:
                # Fallback: whole root display
                subprocess.check_call(['import', '-window', 'root', target],
                                       timeout=5)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                FileNotFoundError) as e:
            response.success = False
            response.message = f'screenshot failed: {e}'
            return response

        # Read dimensions via `identify` if available; otherwise skip.
        width, height = 0, 0
        try:
            out = subprocess.check_output(
                ['identify', '-format', '%w %h', target],
                encoding='utf-8', timeout=3).strip()
            w, h = out.split()
            width, height = int(w), int(h)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                FileNotFoundError, ValueError):
            pass

        response.success = True
        response.message = f'path={target}, width={width}, height={height}'
        return response

    # ------------------------------------------------------------------
    # Widget registry services (Phase 07.1) — list/get/set any registered
    # widget by name. Agent-facing full CLI-level control surface.
    # ------------------------------------------------------------------

    def _widget_rebuilding_response(self, response):
        if self._widget_registry_rebuilding or not getattr(self, '_gui_ready', False):
            response.success = False
            response.message = 'GUI rebuilding — retry in 1s'
            return True
        return False

    @staticmethod
    def _widget_live_value(entry):
        """Read the current value of a widget entry. Returns (value_str, extra_str)."""
        widget = entry['widget']() if entry['widget'] else None
        var = entry.get('var')
        wtype = entry['type']
        if widget is None:
            return '(widget freed)', ''
        try:
            if wtype == 'Spinbox' or wtype == 'Scale':
                val = var.get() if var is not None else widget.get()
                return f'{val}', ''
            if wtype == 'Entry':
                val = var.get() if var is not None else widget.get()
                return f'{val}', ''
            if wtype == 'Checkbutton':
                val = bool(var.get()) if var is not None else False
                return ('true' if val else 'false'), ''
            if wtype == 'Listbox':
                sel = widget.curselection()
                idx = int(sel[0]) if sel else -1
                count = widget.size()
                text = widget.get(idx) if idx >= 0 else ''
                items = [widget.get(i) for i in range(count)]
                return text, f'index={idx}, count={count}, items=[{"|".join(items)}]'
            if wtype == 'Radiobutton':
                return str(var.get()), 'values=[' + '|'.join(entry['choices']) + ']'
            if wtype == 'Combobox':
                val = var.get() if var is not None else widget.get()
                values = list(widget.cget('values') or ())
                return f'{val}', f'values=[{"|".join(values)}]'
            if wtype == 'Notebook':
                active = widget.select()
                return widget.tab(active, 'text') if active else '', ''
            if wtype == 'Text':
                raw = widget.get('1.0', tk.END)
                if len(raw) > 10240:
                    raw = '…(truncated)…' + raw[-10240:]
                # Compact summary in extra field
                lines = raw.rstrip('\n').split('\n')
                summary = f'chars={len(raw)}, lines={len(lines)}'
                return raw, summary
        except tk.TclError as e:
            return f'(tcl error: {e})', ''
        return '(unknown type)', ''

    def _build_list_widgets_output(self):
        """Build the markdown table for _srv_list_widgets."""
        lines = []
        lines.append('# Widgets (auto-generated from list_widgets)')
        lines.append('')
        lines.append('> Regenerated from control_gui.py via '
                     '`ros2 service call /so_arm101_control_gui/list_widgets '
                     'std_srvs/srv/Trigger {}`. Do NOT hand-edit.')
        lines.append('')
        lines.append('## Widgets')
        lines.append('')
        lines.append('| Tab | Section | Label | Type | Current Value | Writable |')
        lines.append('|-----|---------|-------|------|---------------|----------|')
        with self._widget_registry_lock:
            items = sorted(self._widget_registry.items(),
                           key=lambda kv: (kv[1].get('tab') or '',
                                           kv[1].get('section') or '',
                                           kv[1].get('label') or ''))
            rows = []
            agent_only = []
            for rid, entry in items:
                value, _extra = self._widget_live_value(entry)
                # Truncate value for table readability
                vshort = value if len(value) <= 30 else value[:27] + '…'
                # Escape | to keep table rendering
                vshort = vshort.replace('|', '\\|').replace('\n', ' ')
                row = (
                    f'| {entry.get("tab") or "-"} '
                    f'| {entry.get("section") or "-"} '
                    f'| `{rid}` '
                    f'| {entry["type"]} '
                    f'| {vshort} '
                    f'| {"yes" if entry["writable"] else "no"} |'
                )
                if entry.get('tab') or entry.get('section'):
                    rows.append(row)
                else:
                    agent_only.append(row)
            lines.extend(rows)
            lines.append('')
            lines.append('## Agent-only widgets (no tab/section)')
            lines.append('')
            lines.append('| Label | Type | Current Value | Writable |')
            lines.append('|-------|------|---------------|----------|')
            for row in agent_only:
                # Strip the tab/section columns for this sub-table
                parts = [p.strip() for p in row.split('|')]
                # parts: ['', tab, section, label, type, value, writable, '']
                lines.append(f'| {parts[3]} | {parts[4]} | {parts[5]} | {parts[6]} |')
        return '\n'.join(lines)

    def _srv_list_widgets(self, request, response):
        """Return the full widget inventory as markdown."""
        if self._widget_rebuilding_response(response):
            return response
        try:
            response.message = self._build_list_widgets_output()
            response.success = True
        except Exception as e:
            response.success = False
            response.message = f'list_widgets error: {e}'
        return response

    def _srv_get_widget_value(self, request, response):
        """Read a widget's current value by id (passed via widget_id param).

        Also accepts button IDs — if widget_id matches a button in
        _button_registry, returns the same state _srv_get_widget_state
        would return (D-11 uniform API).
        """
        if self._widget_rebuilding_response(response):
            return response
        wid = self.get_parameter('widget_id').get_parameter_value().string_value.strip()
        if not wid:
            response.success = False
            response.message = ('no widget_id. Set param first: '
                                "ros2 param set /so_arm101_control_gui widget_id '<id>'")
            return response

        # If it's a button, delegate to the existing widget-state service
        for btn in self._button_registry:
            if btn.get('command_name') == wid or btn.get('text') == wid:
                # Temporarily reuse get_widget_state path via ik_target
                old_param = self.get_parameter('ik_target').get_parameter_value().string_value
                try:
                    from rclpy.parameter import Parameter as RclParam
                    self.set_parameters([RclParam('ik_target', RclParam.Type.STRING, wid)])
                    return self._srv_get_widget_state(request, response)
                finally:
                    self.set_parameters([RclParam('ik_target', RclParam.Type.STRING, old_param)])

        with self._widget_registry_lock:
            entry = self._widget_registry.get(wid)
        if entry is None:
            response.success = False
            response.message = f'widget {wid!r} not found — call ~/list_widgets for the full inventory'
            return response

        value, extra = self._widget_live_value(entry)
        parts = [
            f'id={wid}',
            f'type={entry["type"]}',
            f'tab={entry.get("tab") or "-"}',
            f'section={entry.get("section") or "-"}',
            f'writable={"true" if entry["writable"] else "false"}',
            f'value={value}',
        ]
        if extra:
            parts.append(f'extra={extra}')
        response.success = True
        response.message = ', '.join(parts)
        return response

    def _srv_set_widget_value(self, request, response):
        """Write a value to a registered widget by id.

        Reads widget_id and widget_value params. Coerces the value to the
        widget's expected type (D-15..D-18). Dispatches the write on the
        GUI thread and waits up to 1.0s for completion (D-19).
        """
        if self._widget_rebuilding_response(response):
            return response
        wid = self.get_parameter('widget_id').get_parameter_value().string_value.strip()
        raw = self.get_parameter('widget_value').get_parameter_value().string_value
        if not wid:
            response.success = False
            response.message = ('no widget_id. Set both params first: '
                                "ros2 param set ... widget_id '<id>' && "
                                "ros2 param set ... widget_value '<value>'")
            return response

        # Buttons reject writes
        for btn in self._button_registry:
            if btn.get('command_name') == wid or btn.get('text') == wid:
                response.success = False
                response.message = (f'{wid!r} is a button — use '
                                    f'`ros2 service call ~/{btn.get("command_name","<cmd>")[5:]}`')
                return response

        with self._widget_registry_lock:
            entry = self._widget_registry.get(wid)
        if entry is None:
            response.success = False
            response.message = f'widget {wid!r} not found'
            return response
        if not entry['writable']:
            response.success = False
            response.message = f'widget {wid!r} is read-only'
            return response

        widget_ref = entry['widget']
        widget = widget_ref() if widget_ref else None
        var = entry.get('var')
        wtype = entry['type']
        if widget is None:
            response.success = False
            response.message = f'widget {wid!r} freed — retry after next refresh'
            return response

        # ---- Auto-switch tab so the write is visible in the GUI ----
        # When the target widget is tagged with a tab (FK/IK/Grasp/RViz),
        # switch the main notebook to that tab BEFORE writing, so a human
        # watching the GUI sees the change in context. Skip for the notebook
        # itself (setting 'tab' IS the switch) and for untagged widgets.
        target_tab = entry.get('tab')
        if target_tab and wtype != 'Notebook':
            with self._widget_registry_lock:
                nb_entry = self._widget_registry.get('tab')
            if nb_entry is not None:
                nb_widget = nb_entry['widget']() if nb_entry['widget'] else None
                if nb_widget is not None:
                    try:
                        tabs = [nb_widget.tab(t, 'text') for t in nb_widget.tabs()]
                        if target_tab in tabs:
                            def _switch_tab():
                                try:
                                    nb_widget.select(tabs.index(target_tab))
                                except tk.TclError:
                                    pass
                            if getattr(self, '_gui_ready', False):
                                self.root.after(0, _switch_tab)
                    except tk.TclError:
                        pass

        # ---- Coerce per type ----
        done = threading.Event()
        outcome = {'ok': False, 'msg': ''}

        def _do_write():
            try:
                if wtype in ('Spinbox', 'Scale'):
                    try:
                        val = float(raw)
                    except ValueError:
                        outcome['msg'] = f'{wid!r} expects float, got {raw!r}'
                        return
                    # Clamp to widget's configured range if available
                    try:
                        lo = float(widget.cget('from'))
                        hi = float(widget.cget('to'))
                        if val < lo or val > hi:
                            self._append_log(
                                f'set_widget_value: {wid} clamped {val} to '
                                f'[{lo}, {hi}]', 'warn')
                            val = max(lo, min(val, hi))
                    except (tk.TclError, ValueError):
                        pass
                    if var is not None:
                        var.set(val)
                    else:
                        widget.set(val)
                    # tkinter Scale's `command` only fires on user drag, not
                    # on var.set(val). For arm/IK joint sliders that means
                    # _on_slider doesn't run, so self.joint_positions stays
                    # stale and plan_execute reads the wrong target. Fire
                    # _on_slider manually for joint-name-matched widgets.
                    if wtype == 'Scale':
                        joint_name = wid.split('@')[0]  # strip tab suffix if any
                        if joint_name in ALL_JOINT_NAMES and hasattr(
                                self, '_on_slider'):
                            try:
                                self._on_slider(joint_name, float(val))
                            except Exception:
                                pass
                    outcome['ok'] = True
                    outcome['msg'] = f'wrote {val} to {wid}'
                    return
                if wtype == 'Checkbutton':
                    low = raw.strip().lower()
                    if low in ('true', '1', 'yes', 'on'):
                        b = True
                    elif low in ('false', '0', 'no', 'off', ''):
                        b = False
                    else:
                        outcome['msg'] = f'{wid!r} expects bool (true/false/1/0/yes/no), got {raw!r}'
                        return
                    if var is not None:
                        var.set(b)
                    outcome['ok'] = True
                    outcome['msg'] = f'wrote {"true" if b else "false"} to {wid}'
                    return
                if wtype == 'Entry':
                    if var is not None:
                        var.set(raw)
                    else:
                        widget.delete(0, tk.END)
                        widget.insert(0, raw)
                    outcome['ok'] = True
                    outcome['msg'] = f'wrote {raw!r} to {wid}'
                    return
                if wtype == 'Listbox':
                    try:
                        idx = int(raw)
                    except ValueError:
                        outcome['msg'] = f'{wid!r} expects int row index, got {raw!r}'
                        return
                    count = widget.size()
                    if idx < 0 or idx >= count:
                        outcome['msg'] = f'row {idx} out of range (count={count})'
                        return
                    widget.selection_clear(0, tk.END)
                    widget.selection_set(idx)
                    widget.see(idx)
                    outcome['ok'] = True
                    outcome['msg'] = f'selected row {idx} in {wid}'
                    return
                if wtype == 'Notebook':
                    # Find the tab with matching text
                    tabs = [widget.tab(t, 'text') for t in widget.tabs()]
                    if raw not in tabs:
                        outcome['msg'] = (f'tab {raw!r} not found; '
                                          f'available: {", ".join(tabs)}')
                        return
                    widget.select(tabs.index(raw))
                    outcome['ok'] = True
                    outcome['msg'] = f'switched to tab {raw}'
                    return
                if wtype == 'Radiobutton':
                    if raw not in entry['choices']:
                        outcome['msg'] = f"{raw!r} not in radio values {entry['choices']}"
                        return
                    var.set(raw)
                    outcome['ok'] = True
                    outcome['msg'] = f'wrote {raw!r} to {wid}'
                    return
                if wtype == 'Combobox':
                    values = list(widget.cget('values') or ())
                    if raw not in values:
                        outcome['msg'] = (
                            f'{raw!r} not in combobox values '
                            f'[{", ".join(values)}]')
                        return
                    if var is not None:
                        var.set(raw)
                    else:
                        widget.set(raw)
                    outcome['ok'] = True
                    outcome['msg'] = f'selected {raw!r} in {wid}'
                    return
                outcome['msg'] = f'unsupported widget type {wtype}'
            except Exception as e:
                outcome['msg'] = f'write failed: {e}'
            finally:
                done.set()

        if getattr(self, '_gui_ready', False):
            self.root.after(0, _do_write)
        else:
            response.success = False
            response.message = 'GUI not ready'
            return response

        if not done.wait(timeout=1.0):
            response.success = False
            response.message = f'write to {wid} timed out after 1.0s'
            return response

        response.success = outcome['ok']
        response.message = outcome['msg'] or ('ok' if outcome['ok'] else 'unknown error')
        return response

    def _srv_get_joint_positions(self, request, response):
        """Return current joint positions as name=value pairs."""
        with self.joint_lock:
            pairs = [f'{n}={self.joint_positions[n]:.6f}'
                     for n in ALL_JOINT_NAMES]
        response.success = True
        response.message = ', '.join(pairs)
        return response

    def _srv_get_ik_target(self, request, response):
        """Read current IK target fields (XYZ + quaternion)."""
        done_event = threading.Event()
        result = {'msg': ''}

        def _read():
            try:
                parts = []
                for axis in ['X', 'Y', 'Z']:
                    parts.append(f'{axis}={self.xyz_vars[axis].get():.6f}')
                for comp in ['Roll', 'shoulder_lift', 'Yaw']:
                    parts.append(f'{comp}={self.rpy_vars[comp].get():.1f}')
                qx, qy, qz, qw = self._rpy_deg_to_quat(
                    self.rpy_vars['Roll'].get(),
                    self.rpy_vars['shoulder_lift'].get(),
                    self.rpy_vars['Yaw'].get())
                parts.extend([f'qx={qx:.6f}', f'qy={qy:.6f}',
                              f'qz={qz:.6f}', f'qw={qw:.6f}'])
                result['msg'] = ', '.join(parts)
            except Exception as e:
                result['msg'] = f'error: {e}'
            finally:
                done_event.set()

        if getattr(self, '_gui_ready', False):
            self.root.after(0, _read)
        else:
            result['msg'] = 'GUI not available'
            done_event.set()

        done_event.wait(timeout=2.0)
        response.success = 'error' not in result['msg']
        response.message = result['msg']
        return response

    def _srv_set_ik_target(self, request, response):
        """Write IK target fields. Pass key=value pairs in request.message.
        Example: ros2 service call ... '{message: "x=0.12, z=0.15, qw=1.0"}'
        Supported keys: x, y, z, qx, qy, qz, qw (case-insensitive)."""
        done_event = threading.Event()
        result = {'ok': True, 'msg': ''}

        # Parse key=value pairs from the trigger message field
        raw = getattr(request, 'message', '') if hasattr(request, 'message') else ''
        # Trigger doesn't have a message field on request — use a workaround:
        # We'll accept the values from the service call's yaml string that gets
        # stuffed into the Trigger request. But Trigger.Request has no fields.
        # So we need to use a different approach — pass via ROS param or topic.
        # Actually, let's use a simple convention: the caller sets a parameter first.

        # Better approach: read from a latched parameter
        # For now, parse from the node's parameter
        raw = self.get_parameter('ik_target').get_parameter_value().string_value

        if not raw:
            response.success = False
            response.message = (
                'Set param first: ros2 param set /so_arm101_control_gui '
                'ik_target "x=0.12,y=0.0,z=0.15,qw=1.0" '
                'then call this service')
            return response

        # Parse
        updates = {}
        for part in raw.replace(' ', '').split(','):
            if '=' in part:
                k, v = part.split('=', 1)
                updates[k.lower()] = float(v)

        def _write():
            try:
                axis_map = {'x': 'X', 'y': 'Y', 'z': 'Z'}
                rpy_map = {'roll': 'Roll', 'pitch': 'shoulder_lift', 'yaw': 'Yaw'}
                for k, v in updates.items():
                    if k in axis_map and axis_map[k] in self.xyz_vars:
                        self.xyz_vars[axis_map[k]].set(v)
                    elif k in rpy_map and rpy_map[k] in self.rpy_vars:
                        self.rpy_vars[rpy_map[k]].set(v)
                set_keys = ', '.join(f'{k}={v:.4f}' for k, v in updates.items())
                result['msg'] = f'Set: {set_keys}'
            except Exception as e:
                result['ok'] = False
                result['msg'] = str(e)
            finally:
                done_event.set()

        if getattr(self, '_gui_ready', False):
            self.root.after(0, _write)
        else:
            result['ok'] = False
            result['msg'] = 'GUI not available'
            done_event.set()

        done_event.wait(timeout=2.0)
        response.success = result['ok']
        response.message = result['msg']
        return response

    def _srv_get_ee_pose(self, request, response):
        """Read current End-Effector pose values (gripper link)."""
        parts = []
        for key in ['X', 'Y', 'Z', 'qx', 'qy', 'qz', 'qw']:
            parts.append(f'{key}={self.ee_labels[key].get()}')
        response.success = True
        response.message = ', '.join(parts)
        return response

    def _srv_get_tcp_pose(self, request, response):
        """Look up tcp_link pose in base frame via TF2."""
        try:
            t = self._tf_buffer.lookup_transform(
                'base', 'tcp_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5))
            p = t.transform.translation
            q = t.transform.rotation
            response.success = True
            response.message = (
                f'X={p.x:.6f}, Y={p.y:.6f}, Z={p.z:.6f}, '
                f'qx={q.x:.6f}, qy={q.y:.6f}, qz={q.z:.6f}, qw={q.w:.6f}')
        except Exception as e:
            response.success = False
            response.message = f'TF lookup failed: {e}'
        return response

    def _srv_get_log(self, request, response):
        """Read last 20 lines from Process Log."""
        done_event = threading.Event()
        result = {'msg': ''}

        def _read():
            try:
                content = self._process_log.get('1.0', 'end').strip()
                lines = content.split('\n')
                result['msg'] = '\n'.join(lines) if lines else '(empty)'
            except Exception as e:
                result['msg'] = f'error: {e}'
            finally:
                done_event.set()

        if getattr(self, '_gui_ready', False):
            self.root.after(0, _read)
        else:
            result['msg'] = 'GUI not available'
            done_event.set()

        done_event.wait(timeout=2.0)
        response.success = 'error' not in result['msg']
        response.message = result['msg']
        return response

    # ------------------------------------------------------------------
    # Controller command publishing (via action interface)
    # ------------------------------------------------------------------





    # ------------------------------------------------------------------
    # Joint state feedback
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # GUI setup
    # ------------------------------------------------------------------

    def _setup_gui_thread(self):
        # macOS requires tkinter on the main thread (Cocoa constraint).
        # Defer _run_gui to be called from main() instead of a daemon thread.
        pass

    def _run_gui(self):
        self.root = tk.Tk()
        self.root.title('SO-ARM101 Control')
        self.root.geometry('580x780')
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

        # Status bar
        self.status_var = tk.StringVar(value='Mode: Simulation')
        status_frame = tk.Frame(self.root)
        status_frame.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(status_frame, textvariable=self.status_var, anchor='w',
                 font=('Arial', 10, 'bold')).pack(side=tk.LEFT)

        self.hw_var = tk.StringVar(value='sim')
        tk.Radiobutton(status_frame, text='Real Hardware', variable=self.hw_var,
                       value='real', command=self._toggle_hardware).pack(side=tk.RIGHT, padx=2)
        tk.Radiobutton(status_frame, text='Simulation', variable=self.hw_var,
                       value='sim', command=self._toggle_hardware).pack(side=tk.RIGHT, padx=2)

        # Ground plane collision toggle (common to all tabs)
        scene_frame = tk.Frame(self.root)
        scene_frame.pack(fill=tk.X, padx=5, pady=1)
        self._ground_plane_var = tk.BooleanVar(value=True)
        self._register_check(scene_frame, label='Ground Plane',
                             section='Scene',
                             variable=self._ground_plane_var,
                             command=self._cmd_toggle_ground_plane).pack(side=tk.LEFT)
        tk.Label(scene_frame, text='  Z:').pack(side=tk.LEFT)
        self._ground_z_var = tk.DoubleVar(value=0.0)
        self._register_spinbox(scene_frame, label='Ground Z',
                               section='Scene',
                               textvariable=self._ground_z_var,
                               from_=-0.5, to=0.5, increment=0.01,
                               width=6).pack(side=tk.LEFT)
        # Publish ground plane on startup after MoveIt is ready
        self.root.after(3000, self._cmd_toggle_ground_plane)

        # Notebook (tabs)
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 0))
        self._notebook = notebook
        self._register_notebook(notebook, label='tab')

        self._build_individual_tab(notebook)
        self._build_arm_control_tab(notebook)
        self._build_grasp_tab(notebook)
        self._build_quickstart_tab(notebook)
        self._build_real_test_tab(notebook)
        self._build_record_sim_tab(notebook)
        self._build_inference_tab(notebook)
        self._build_display_tab(notebook)

        # Auto-populate IK fields when switching to IK tab
        notebook.bind('<<NotebookTabChanged>>', self._on_tab_changed)

        # --- Log Panel (bottom) ---
        self._build_log_panel()

        # Hot reload keybindings
        self.root.bind('<Control-r>', lambda e: self._hot_reload_logic())
        self.root.bind('<Control-Shift-R>', lambda e: self._hot_reload_gui())

        self._finalize_widget_registry()
        self._gui_ready = True

        # Signal heartbeat. Python runs signal handlers only on the main thread
        # between bytecodes, and _tkinter's mainloop sits blocked in Tcl waiting
        # for an event: on an idle window main()'s SIGINT/SIGTERM handler (which
        # schedules _on_close) never ran, so `ros2 launch` / deploy_sim.sh had to
        # escalate to SIGKILL. A cheap periodic after() wakes the loop so pending
        # signals are handled within ~0.2 s.
        def _signal_heartbeat():
            if self.running:
                self.root.after(200, _signal_heartbeat)
        self.root.after(200, _signal_heartbeat)

        self.root.mainloop()
        self._gui_ready = False

    # ------------------------------------------------------------------
    # Tab 1: Individual Joint Control
    # ------------------------------------------------------------------

    def _build_individual_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='FK')

        self.sliders = {}
        self.slider_labels = {}

        # --- Arm Section ---
        arm_frame = ttk.LabelFrame(frame, text='Arm')
        arm_frame.pack(fill=tk.X, padx=10, pady=(5, 2))

        for name in ARM_JOINT_NAMES:
            lo, hi = JOINT_LIMITS[name]
            row = tk.Frame(arm_frame)
            row.pack(fill=tk.X, padx=5, pady=2)

            tk.Label(row, text=name, width=14, anchor='w').pack(side=tk.LEFT, padx=(5, 0))

            var = tk.DoubleVar(value=0.0)
            slider = self._register_scale(
                row, label=name, tab='FK', section='Joint Sliders',
                variable=var, from_=lo, to=hi,
                orient=tk.HORIZONTAL, resolution=0.001, length=300,
                command=lambda val, n=name: self._on_slider(n, float(val)))
            slider.pack(side=tk.LEFT, padx=5)

            lbl = tk.Label(row, text='0.000', width=8)
            lbl.pack(side=tk.LEFT)

            self.sliders[name] = var
            self.slider_labels[name] = lbl

        arm_btn_frame = tk.Frame(arm_frame)
        arm_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        self._build_arm_btn_row(arm_btn_frame, tab='FK',
                                reset_cmd=self._cmd_zero_arm)

        # --- Gripper Section ---
        gripper_frame = ttk.LabelFrame(frame, text='Gripper')
        gripper_frame.pack(fill=tk.X, padx=10, pady=(2, 2))

        jaw_row = tk.Frame(gripper_frame)
        jaw_row.pack(fill=tk.X, padx=5, pady=2)

        lo, hi = JOINT_LIMITS[GRIPPER_JOINT_NAME]
        tk.Label(jaw_row, text=GRIPPER_JOINT_NAME, width=14, anchor='w').pack(side=tk.LEFT, padx=(5, 0))

        jaw_var = tk.DoubleVar(value=0.0)
        jaw_slider = self._register_scale(
            jaw_row, label=GRIPPER_JOINT_NAME, tab='FK', section='Joint Sliders',
            variable=jaw_var, from_=lo, to=hi,
            orient=tk.HORIZONTAL, resolution=0.001, length=300,
            command=lambda val: self._on_slider(GRIPPER_JOINT_NAME, float(val)))
        jaw_slider.pack(side=tk.LEFT, padx=5)

        jaw_lbl = tk.Label(jaw_row, text='0.000', width=8)
        jaw_lbl.pack(side=tk.LEFT)

        self.sliders[GRIPPER_JOINT_NAME] = jaw_var
        self.slider_labels[GRIPPER_JOINT_NAME] = jaw_lbl

        gripper_btn_frame = tk.Frame(gripper_frame)
        gripper_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        self._build_gripper_btn_row(gripper_btn_frame, tab='FK')

        # --- Action Buttons (always visible) ---
        action_frame = tk.Frame(frame)
        action_frame.pack(fill=tk.X, padx=10, pady=(8, 5))
        self.set_joints_btn = self._register_button(
            action_frame, text='Set Joints', tab='FK', section='Actions',
            command=self._cmd_set_joints)
        self.set_joints_btn.pack(side=tk.LEFT, padx=5)
        self.execute_btn = self._register_button(
            action_frame, text='Plan & Execute', tab='FK', section='Actions',
            command=self._cmd_plan_execute)
        self.execute_btn.pack(side=tk.LEFT, padx=5)
        tk.Label(action_frame, text='Speed:', font=('Arial', 9)).pack(side=tk.LEFT, padx=(10, 2))
        self.velocity_scale_var = tk.DoubleVar(value=0.5)
        self._last_speed_val = 0.5
        vcmd = (self.root.register(self._validate_speed), '%P')
        self.velocity_scale_spin = self._register_spinbox(
            action_frame, label='Speed', tab='FK', section='Actions',
            textvariable=self.velocity_scale_var,
            from_=0.1, to=1.0, increment=0.1,
            width=4, font=('Arial', 9), validate='all', validatecommand=vcmd)
        self.velocity_scale_spin.pack(side=tk.LEFT)

    def _validate_speed(self, value_str):
        """Validate speed spinbox: allow empty (during editing) and 0.1-1.0."""
        if value_str == '':
            return True
        try:
            v = float(value_str)
            if v > 1.0:
                self._append_log('Speed must be between 0.1 and 1.0 (100% of joint limits)', 'warn')
                return False
            if v < 0.1:
                self._append_log('Speed must be between 0.1 and 1.0', 'warn')
                return False
            if v != self._last_speed_val:
                self._last_speed_val = v
                pct = int(v * 100)
                self._append_log(f'Speed set to {v:.1f} ({pct}% of joint limits)')
            return True
        except ValueError:
            return False

    def _on_slider(self, joint_name, value):
        self._slider_driven = True
        with self.joint_lock:
            self.joint_positions[joint_name] = value
        if joint_name in self.slider_labels:
            self.slider_labels[joint_name].config(text=f'{value:.3f}')
        # Sync IK tab jaw label (sliders share the same DoubleVar)
        if joint_name == GRIPPER_JOINT_NAME and hasattr(self, '_ik_jaw_label'):
            self._ik_jaw_label.config(text=f'{value:.3f}')
        # Switch planning group based on which joint is being moved
        if joint_name == GRIPPER_JOINT_NAME:
            self._select_planning_group('gripper')
        else:
            self._select_planning_group('arm')
        # Update RViz goal state only — robot doesn't move until button click
        self._publish_goal_state()

    def _cmd_zero_arm(self):
        """Reset arm joints to zero."""
        self._slider_driven = True
        self._select_planning_group('arm')
        with self.joint_lock:
            for name in ARM_JOINT_NAMES:
                self.joint_positions[name] = 0.0
        for name in ARM_JOINT_NAMES:
            if name in self.sliders:
                self.sliders[name].set(0.0)
                self.slider_labels[name].config(text='0.000')
        self._publish_goal_state()
        self.status_var.set('Arm zeroed')

    def _cmd_reset_grasp_home(self):
        """Set slider targets to the grasp-home pose (gripper pointing down).

        Mirrors _cmd_zero_arm: sets joint_positions + sliders + goal state,
        but with the grasp_home config instead of zeros. Does NOT execute
        motion — only stages the target (same discipline as Reset Arm).
        """
        from so_arm101_control.compute_workspace import WRIST_ROLL_URDF_PITCH
        target = {name: 0.0 for name in ARM_JOINT_NAMES}
        target['wrist_flex'] = math.pi / 2
        target['wrist_roll'] = -math.pi / 2 + WRIST_ROLL_URDF_PITCH
        self._slider_driven = True
        self._select_planning_group('arm')
        with self.joint_lock:
            for name in ARM_JOINT_NAMES:
                self.joint_positions[name] = target[name]
        for name in ARM_JOINT_NAMES:
            if name in self.sliders:
                self.sliders[name].set(target[name])
                self.slider_labels[name].config(text=f'{target[name]:.3f}')
        self._publish_goal_state()
        self.status_var.set('Grasp home target staged')

    def _cmd_gripper_zero(self):
        """Reset gripper to zero (closed)."""
        self._gripper_command(0.0)
        self.status_var.set('Gripper zeroed')


    def _cmd_randomize_arm(self):
        """Set arm joints to a random collision-free configuration."""
        self._select_planning_group('arm')
        self.status_var.set('Finding random valid state...')

        def _find_valid():
            max_attempts = 100
            for attempt in range(max_attempts):
                # Generate random joint values within limits
                positions = {}
                for name in ARM_JOINT_NAMES:
                    lo, hi = JOINT_LIMITS[name]
                    positions[name] = random.uniform(lo, hi)

                # Check validity via MoveIt
                if MOVEIT_AVAILABLE and hasattr(self, 'validity_client') \
                        and self.validity_client.service_is_ready():
                    req = GetStateValidity.Request()
                    req.robot_state.joint_state.name = list(ARM_JOINT_NAMES)
                    req.robot_state.joint_state.position = [
                        positions[n] for n in ARM_JOINT_NAMES]
                    req.group_name = 'arm'
                    future = self.validity_client.call_async(req)
                    self._wait_future(future, timeout_sec=1.0)
                    if future.result() is not None and not future.result().valid:
                        continue  # collision — retry
                # Valid (or no validity service available) — apply it
                if getattr(self, '_gui_ready', False):
                    self.root.after(0, self._apply_random_arm, positions)
                return

            # Exhausted attempts — apply last one anyway
            self._append_log(f'No collision-free state after {max_attempts} attempts', 'warn')
            if getattr(self, '_gui_ready', False):
                self.root.after(0, self._apply_random_arm, positions)

        threading.Thread(target=_find_valid, daemon=True).start()

    def _apply_random_arm(self, positions):
        """Apply validated random arm positions to sliders and goal state."""
        self._slider_driven = True
        with self.joint_lock:
            self.joint_positions.update(positions)
        for name in ARM_JOINT_NAMES:
            val = positions[name]
            if name in self.sliders:
                self.sliders[name].set(val)
                self.slider_labels[name].config(text=f'{val:.3f}')
        self._publish_goal_state()
        self.status_var.set('Arm randomized (valid)')

    # ------------------------------------------------------------------
    # Planning Mode
    # ------------------------------------------------------------------



    def _cmd_set_joints(self):
        """Send current slider positions directly to arm + gripper controllers."""
        with self.joint_lock:
            positions = dict(self.joint_positions)
        self._send_arm_goal(positions, duration_s=0.5)
        jaw = positions.get(GRIPPER_JOINT_NAME, 0.0)
        self._send_gripper_goal(jaw, duration_s=0.5)
        if self.use_real_hardware:
            self._send_hw_command(positions)
        joints_str = ', '.join(f'{n}: {positions[n]:.3f}' for n in ARM_JOINT_NAMES)
        jaw_str = f'{GRIPPER_JOINT_NAME}: {jaw:.3f}'
        self.status_var.set('Joints set')
        self._append_log(f'Set Joints → {joints_str}, {jaw_str}')
        # Allow joint_states to sync sliders again after robot reaches goal
        self.root.after(1000, self._clear_slider_driven)

    def _clear_slider_driven(self):
        """Allow joint_states callback to sync sliders again."""
        self._slider_driven = False

    # ------------------------------------------------------------------
    # Collision-free IK planning: geometric IK → collision check → MoveIt path
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # FK tab Plan & Execute (MoveIt collision-aware path planning)
    # ------------------------------------------------------------------

    def _cmd_plan_execute(self, target=None, on_complete=None, planner_id='',
                          planning_time=10.0):
        """Plan and execute via MoveIt with collision avoidance.

        Args:
            target: dict of joint name → angle. If None, reads from slider
                positions (GUI button mode).
            on_complete: optional threading.Event to set when done
                (programmatic/service mode).
            planner_id: OMPL planner to use (e.g. 'RRTstar' for optimized
                path). Empty string = default (RRTConnect).
            planning_time: max seconds for planner (default 10s, use 30s
                for optimizing planners like RRTstar).
        """
        if not MOVEIT_AVAILABLE or self.plan_client is None:
            self.status_var.set('MoveIt not available')
            self._last_motion_status = {
                'ok': False, 'outcome': 'moveit_unavailable',
                'msg': 'MoveIt not available'}
            if on_complete:
                on_complete.set()
            return
        if not self.plan_client.service_is_ready():
            self.status_var.set('Planning service not ready...')
            self._append_log('/plan_kinematic_path service not ready', 'warn')
            self._last_motion_status = {
                'ok': False, 'outcome': 'plan_service_not_ready',
                'msg': '/plan_kinematic_path service not ready'}
            if on_complete:
                on_complete.set()
            return

        # If a lego is attached, re-snap its tcp_link-local pose from Isaac's
        # current /objects_poses_sim reading. Arm should be at rest here (any
        # previous motion has completed), so the re-seat happens on synchronized
        # TF+pose streams — no Layer-B-lag baked in. Fixes the "attached lego
        # offset from gripper" RViz visualization bug.
        if getattr(self, '_attached_lego_name', None):
            try:
                self._refresh_attached_pose()
            except Exception as exc:
                self._append_log(f'refresh_attached_pose failed: {exc}', 'warn')

        self.root.after(0, lambda: self.execute_btn.config(state=tk.DISABLED))
        self._set_status('Planning...')
        # Tell the RViz MotionPlanning panel we're planning for the arm.
        # Gripper commands published "gripper" to /rviz/moveit/select_planning_group
        # (control_gui.py:6102). Without this switch-back, RViz's Goal State ghost
        # stays locked to the gripper group and renders no orange arm ghost.
        self._select_planning_group('arm')
        # Phase 9: if no on_complete was passed (button click or Trigger
        # service), create one and register as _motion_event so the service
        # wrapper waits for the real motion verdict instead of returning
        # "dispatched" immediately.
        if on_complete is None:
            on_complete = threading.Event()
            self._motion_event = on_complete
        self._plan_execute_on_complete = on_complete

        if target is None:
            with self.joint_lock:
                target = {n: self.joint_positions[n] for n in ARM_JOINT_NAMES}
                self._execute_jaw_target = self.joint_positions.get(GRIPPER_JOINT_NAME, 0.0)
        else:
            # Sync sliders so RViz ghost updates to match the target
            with self.joint_lock:
                for n in ARM_JOINT_NAMES:
                    self.joint_positions[n] = target[n]
            self._execute_jaw_target = None

        goal_str = ', '.join(f'{n}: {target[n]:.3f}' for n in ARM_JOINT_NAMES)
        self._append_log(f'Plan & Execute \u2192 {goal_str}')

        constraints = Constraints()
        for name in ARM_JOINT_NAMES:
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = target[name]
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        request = GetMotionPlan.Request()
        mpr = MotionPlanRequest()
        mpr.group_name = 'arm'
        mpr.pipeline_id = 'ompl'
        mpr.planner_id = planner_id
        attempts_var = getattr(self, '_planning_attempts_var', None)
        mpr.num_planning_attempts = attempts_var.get() if attempts_var else 50
        mpr.allowed_planning_time = planning_time
        vel_scale = self.velocity_scale_var.get()
        mpr.max_velocity_scaling_factor = vel_scale
        mpr.max_acceleration_scaling_factor = vel_scale
        mpr.goal_constraints.append(constraints)

        # Phase 9: populate start_state from live joint positions. Without
        # this MoveIt emits "Found empty JointState message" errors and
        # silently aborts planning (verified: 4 such errors per call when
        # start_state was default-constructed; no trajectory returned).
        # _ompl_plan_sync already does this; FK-tab plan_execute was the
        # one remaining path missing it.
        start_state = RobotState()
        start_state.joint_state.name = list(ALL_JOINT_NAMES)
        with self.joint_lock:
            start_state.joint_state.position = [
                float(self._actual_positions.get(
                    n, self.joint_positions.get(n, 0.0)))
                for n in ALL_JOINT_NAMES]
        mpr.start_state = start_state

        request.motion_plan_request = mpr

        future = self.plan_client.call_async(request)
        future.add_done_callback(self._ros_callback(self._plan_and_execute_callback))

    def _plan_and_execute_callback(self, future):
        """Handle planning result — display trajectory in RViz, then execute."""
        on_complete = getattr(self, '_plan_execute_on_complete', None)
        try:
            resp = future.result()
        except Exception as e:
            self._set_status(f'Planning failed: {e}')
            self.root.after(0, lambda: self.execute_btn.config(state=tk.NORMAL))
            self._last_motion_status = {
                'ok': False, 'outcome': 'plan_exception',
                'msg': f'planning exception: {e!r}'}
            if on_complete:
                on_complete.set()
            return

        error_code = resp.motion_plan_response.error_code.val
        if error_code != 1:
            self._set_status(f'Planning failed (error {error_code})')
            self._append_log(f'Planning failed (error {error_code})', 'warn')
            self.root.after(0, lambda: self.execute_btn.config(state=tk.NORMAL))
            self._last_motion_status = {
                'ok': False, 'outcome': 'plan_failed',
                'msg': f'plan_kinematic_path error_code={error_code}',
                'error_code': error_code}
            if on_complete:
                on_complete.set()
            return

        robot_trajectory = resp.motion_plan_response.trajectory
        pt = resp.motion_plan_response.planning_time
        n_pts = len(robot_trajectory.joint_trajectory.points)
        self._append_log(f'Plan found ({pt:.3f}s), {n_pts} points')

        # Mode B post-check — subsamples between waypoints for collisions
        # OMPL's ~3.6°-spaced discrete check can miss. Same discipline the
        # grasp path runs at _ompl_plan_validate_execute (line 5883).
        # Without this, OMPL-validated trajectories can still clip cups at
        # sub-segment states and get executed. Parity closed 2026-04-24.
        chk = self._trajectory_first_invalid_with_contacts(
            robot_trajectory.joint_trajectory)
        if not chk.ok:
            contact_summary = '; '.join(
                f'{c.contact_body_1}↔{c.contact_body_2}(d={c.depth*1000:.1f}mm)'
                for c in chk.contacts[:5]) or 'no contact info'
            where = (f'wp[{chk.bad_wp}]' if chk.bad_subidx is None
                     else f'wp[{chk.bad_wp}→{chk.bad_wp + 1}] sub-t={chk.sub_t:.2f}')
            self._set_status(f'Plan rejected by Mode B: {where}')
            self._append_log(
                f'  Mode B rejected {where} '
                f'({chk.n_wps} wps + {chk.n_sub} subs): {contact_summary}',
                'warn')
            self.root.after(0, lambda: self.execute_btn.config(state=tk.NORMAL))
            self._last_motion_status = {
                'ok': False, 'outcome': 'plan_mode_b_rejected',
                'msg': f'Mode B rejected {where}: {contact_summary}',
                'bad_wp': chk.bad_wp, 'bad_subidx': chk.bad_subidx,
                'sub_t': chk.sub_t}
            if on_complete:
                on_complete.set()
            return
        self._append_log(
            f'  Mode B validated ({chk.n_wps} wps + {chk.n_sub} subs clear)')

        # Display trajectory in RViz
        from moveit_msgs.msg import DisplayTrajectory
        display_msg = DisplayTrajectory()
        display_msg.trajectory.append(robot_trajectory)
        self._display_traj_pub.publish(display_msg)

        # Execute via arm_controller
        if not self.arm_action_client.server_is_ready():
            self._set_status('Arm controller not ready')
            self.root.after(0, lambda: self.execute_btn.config(state=tk.NORMAL))
            self._last_motion_status = {
                'ok': False, 'outcome': 'action_server_not_ready',
                'msg': 'arm_controller action server not ready'}
            if on_complete:
                on_complete.set()
            return

        self._set_status(f'Executing ({n_pts} points)...')
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = robot_trajectory.joint_trajectory
        send_future = self.arm_action_client.send_goal_async(goal)
        send_future.add_done_callback(self._ros_callback(self._execute_response))

        # Send gripper to its goal position too (only from GUI button path)
        if getattr(self, '_execute_jaw_target', None) is not None:
            self._send_gripper_goal(self._execute_jaw_target, duration_s=1.0)

    def _execute_response(self, future):
        """Handle trajectory execution acceptance."""
        on_complete = getattr(self, '_plan_execute_on_complete', None)
        try:
            goal_handle = future.result()
        except Exception as e:
            self._set_status(f'Execution failed: {e}')
            self.root.after(0, lambda: self.execute_btn.config(state=tk.NORMAL))
            self._last_motion_status = {
                'ok': False, 'outcome': 'send_exception',
                'msg': f'action send_goal exception: {e!r}'}
            if on_complete:
                on_complete.set()
            return
        if not goal_handle.accepted:
            self._set_status('Execution rejected')
            self.root.after(0, lambda: self.execute_btn.config(state=tk.NORMAL))
            self._last_motion_status = {
                'ok': False, 'outcome': 'goal_rejected',
                'msg': 'arm action server rejected the goal'}
            if on_complete:
                on_complete.set()
            return
        self._set_status('Executing trajectory...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._ros_callback(self._execute_result))

    def _execute_result(self, future):
        """Handle trajectory execution result."""
        on_complete = getattr(self, '_plan_execute_on_complete', None)
        try:
            res = future.result()
            ec = getattr(getattr(res, 'result', None), 'error_code', None)
            ok = (ec is None) or (ec == 0)
            if ok:
                self._set_status('Execution complete')
                self._append_log('Trajectory execution complete')
                self._last_motion_status = {
                    'ok': True, 'outcome': 'completed',
                    'msg': 'plan_execute trajectory complete',
                    'error_code': ec}
            else:
                self._set_status(f'Execution error: ec={ec}')
                self._append_log(f'Trajectory error_code={ec}', 'error')
                self._last_motion_status = {
                    'ok': False, 'outcome': 'trajectory_error',
                    'msg': f'trajectory error_code={ec}',
                    'error_code': ec}
        except Exception as e:
            self._set_status(f'Execution error: {e}')
            self._append_log(f'Execution error: {e}', 'error')
            self._last_motion_status = {
                'ok': False, 'outcome': 'result_exception',
                'msg': f'result callback exception: {e!r}'}
        self.root.after(0, lambda: self.execute_btn.config(state=tk.NORMAL))
        # Let joint_states sync sliders back to actual robot position
        self.root.after(1000, self._clear_slider_driven)
        if on_complete:
            on_complete.set()

    def _set_status(self, text):
        """Thread-safe status bar update."""
        if getattr(self, '_gui_ready', False):
            self.root.after(0, self.status_var.set, text)

    # ------------------------------------------------------------------
    # Log Panel
    # ------------------------------------------------------------------

    def _build_log_panel(self):
        """Build the bottom log panel with Process Log and System Errors tabs."""
        self._log_outer = outer = tk.Frame(self.root)
        outer.pack(fill=tk.BOTH, padx=5, pady=(2, 5), expand=False)

        log_notebook = ttk.Notebook(outer)
        log_notebook.pack(fill=tk.BOTH, expand=True)
        self._log_notebook = log_notebook
        self._register_notebook(log_notebook, label='log_tab')

        # Process Log tab — text + buttons inside
        proc_frame = tk.Frame(log_notebook, bg='#1e1e1e')
        log_notebook.add(proc_frame, text='Process Log')
        proc_btn = tk.Frame(proc_frame)
        proc_btn.pack(side=tk.RIGHT, fill=tk.Y)
        self._register_button(proc_btn, text='Clear', tab='Logs', section='Process', width=6,
                              command=self._log_btn_clear).pack(fill=tk.BOTH, expand=True, padx=3, pady=(3, 2))
        self._register_button(proc_btn, text='Copy', tab='Logs', section='Process', width=6,
                              command=self._log_btn_copy).pack(fill=tk.BOTH, expand=True, padx=3, pady=(2, 3))
        proc_scroll = tk.Scrollbar(proc_frame)
        proc_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._process_log = tk.Text(proc_frame, height=6, wrap=tk.WORD,
                                     font=('Consolas', 9), state=tk.DISABLED,
                                     bg='#1e1e1e', fg='#d4d4d4', borderwidth=0)
        self._register_log_text(self._process_log, label='process_log')
        self._process_log.config(yscrollcommand=proc_scroll.set)
        proc_scroll.config(command=self._process_log.yview)
        self._process_log.tag_configure('info', foreground='#d4d4d4')
        self._process_log.tag_configure('warn', foreground='#ffa500')
        self._process_log.tag_configure('error', foreground='#ff4444')
        self._process_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # System Errors tab — text + buttons inside
        err_frame = tk.Frame(log_notebook, bg='#1e1e1e')
        log_notebook.add(err_frame, text='System Errors')
        err_btn = tk.Frame(err_frame)
        err_btn.pack(side=tk.RIGHT, fill=tk.Y)
        self._register_button(err_btn, text='Clear', tab='Logs', section='Errors', width=6,
                              command=self._log_btn_clear).pack(fill=tk.BOTH, expand=True, padx=3, pady=(3, 2))
        self._register_button(err_btn, text='Copy', tab='Logs', section='Errors', width=6,
                              command=self._log_btn_copy).pack(fill=tk.BOTH, expand=True, padx=3, pady=(2, 3))
        err_scroll = tk.Scrollbar(err_frame)
        err_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._error_log = tk.Text(err_frame, height=6, wrap=tk.WORD,
                                   font=('Consolas', 9), state=tk.DISABLED,
                                   bg='#1e1e1e', fg='#ff6b6b', borderwidth=0)
        self._register_log_text(self._error_log, label='error_log')
        self._error_log.config(yscrollcommand=err_scroll.set)
        err_scroll.config(command=self._error_log.yview)
        self._error_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Capture stderr and uncaught exceptions into System Errors
        self._setup_stderr_capture()

    def _get_active_log_widget(self):
        """Return the Text widget for the currently selected log tab."""
        idx = self._log_notebook.index(self._log_notebook.select())
        return self._process_log if idx == 0 else self._error_log

    def _log_btn_clear(self):
        widget = self._get_active_log_widget()
        widget.config(state=tk.NORMAL)
        widget.delete('1.0', tk.END)
        widget.config(state=tk.DISABLED)

    def _log_btn_copy(self):
        widget = self._get_active_log_widget()
        content = widget.get('1.0', tk.END).strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self._append_log('Log copied to clipboard')

    def _setup_stderr_capture(self):
        """Redirect stderr and sys.excepthook to System Errors log."""
        if hasattr(self, '_stderr_captured'):
            return  # already redirected (e.g. after GUI rebuild)
        self._stderr_captured = True
        import sys

        gui = self  # prevent closure issues
        original_stderr = sys.stderr

        class StderrRedirector:
            def __init__(self):
                self._buffer = ''

            def write(self, text):
                original_stderr.write(text)  # still print to terminal
                if not text.strip():
                    return
                self._buffer += text
                if '\n' in self._buffer:
                    lines = self._buffer.split('\n')
                    for line in lines[:-1]:
                        if line.strip():
                            gui._append_system_error(line)
                    self._buffer = lines[-1]

            def flush(self):
                if self._buffer.strip():
                    gui._append_system_error(self._buffer)
                    self._buffer = ''
                original_stderr.flush()

        sys.stderr = StderrRedirector()

        original_excepthook = sys.excepthook

        def _excepthook(exc_type, exc_value, exc_tb):
            import traceback
            tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
            gui._append_system_error(tb_str)
            original_excepthook(exc_type, exc_value, exc_tb)

        sys.excepthook = _excepthook

    def _append_system_error(self, text):
        """Thread-safe append to System Errors log only."""
        import datetime
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        line = f'[{timestamp}] {text}\n'

        def _do():
            if not hasattr(self, '_error_log'):
                return
            self._error_log.config(state=tk.NORMAL)
            self._error_log.insert(tk.END, line)
            self._error_log.see(tk.END)
            self._error_log.config(state=tk.DISABLED)
            # Auto-switch to System Errors tab
            self._log_notebook.select(1)

        if getattr(self, '_gui_ready', False):
            self.root.after(0, _do)

    def _append_log(self, text, level='info'):
        """Thread-safe log append to Process Log. level: 'info', 'warn', 'error'.

        When level is 'warn' or 'error', also sets _cmd_error so the
        trigger service callback can report success=False.

        ALSO routes to the node's rclpy logger so every GUI-level motion
        event (Attached, Detached, Grasp Home, OMPL post-check, etc.)
        lands in the launch rosout log with a wall-clock timestamp.
        Without this, those events only existed in the in-memory Tk log
        buffer (ring-truncated to a few dozen lines), making post-hoc
        reconstruction of a run impossible. If you're grep'ing
        /tmp/so_arm101_mtc.log for a sequence, these lines are now there.
        """
        import datetime
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        line = f'[{timestamp}] {text}\n'

        if level in ('warn', 'error'):
            self._cmd_error = text

        # Persist to rosout. Use the level mapping so grep-by-severity works.
        try:
            logger = self.get_logger()
            if level == 'error':
                logger.error(text)
            elif level == 'warn':
                logger.warn(text)
            else:
                logger.info(text)
        except Exception:
            pass  # Node may not be fully initialized during teardown

        def _do_append():
            if not hasattr(self, '_process_log'):
                return
            self._process_log.config(state=tk.NORMAL)
            self._process_log.insert(tk.END, line, level)
            self._process_log.see(tk.END)
            self._process_log.config(state=tk.DISABLED)

        if getattr(self, '_gui_ready', False):
            self.root.after(0, _do_append)

    # ------------------------------------------------------------------
    # Hot Reload (Ctrl+R = logic, Ctrl+Shift+R = logic + GUI rebuild)
    # ------------------------------------------------------------------

    def _hot_reload_logic(self):
        """Reload control_gui + compute_workspace modules and patch methods.

        Preserves all ROS2 infra, widgets, locks, and runtime state.
        Only method implementations and module-level constants are updated.

        Reloads from the SOURCE directory (not build/) so edits to src/
        take effect without running colcon build first.
        """
        import importlib
        import importlib.util
        import sys

        try:
            # Resolve source directory from the build path
            # build/.../control_gui.py → src/.../control_gui.py
            src_dir = None
            mod_file = sys.modules.get('so_arm101_control.control_gui')
            if mod_file and hasattr(mod_file, '__file__'):
                build_path = mod_file.__file__
                # Replace /build/ with /src/ in the path
                if '/build/' in build_path:
                    src_path = build_path.replace('/build/', '/src/')
                    if os.path.isfile(src_path):
                        src_dir = os.path.dirname(src_path)

            def _reload_from_source(mod_name):
                """Reload a module from src/ instead of build/."""
                mod = sys.modules.get(mod_name)
                if mod is None:
                    return None
                if src_dir:
                    filename = os.path.basename(mod.__file__)
                    src_file = os.path.join(src_dir, filename)
                    if os.path.isfile(src_file):
                        mod.__file__ = src_file
                        # Also update the loader's path so reload reads the right file
                        spec = importlib.util.spec_from_file_location(mod_name, src_file)
                        mod.__spec__ = spec
                        mod.__loader__ = spec.loader
                return importlib.reload(mod)

            # Reload compute_workspace first (control_gui imports from it)
            _reload_from_source('so_arm101_control.compute_workspace')

            _reload_from_source('so_arm101_control.grasp_geometry')
            new_motion = _reload_from_source('so_arm101_control.pick_place')
            _patch_methods(self, new_motion.PickPlaceNode)

            # Reload control_gui
            new_mod = _reload_from_source('so_arm101_control.control_gui')

            # Patch all methods on this running instance
            _patch_methods(self, new_mod.SOArm101ControlGUI)

            # Register any newly-added _cmd_* methods as Trigger services.
            # Without this, new debug services introduced via hot-reload stay
            # callable as Python methods but are invisible to ROS2 service calls.
            existing_names = set()
            for s in self._debug_services:
                sn = getattr(s, 'srv_name', '') or ''
                existing_names.add(sn.rsplit('/', 1)[-1])
            new_count = 0
            for name in sorted(dir(self)):
                if not name.startswith('_cmd_'):
                    continue
                srv_name = name[5:]
                if srv_name in existing_names:
                    continue
                cb = self._make_trigger_callback(name)
                srv = self.create_service(Trigger, f'~/{srv_name}', cb)
                self._debug_services.append(srv)
                new_count += 1
            if new_count:
                self._append_log(
                    f'HOT RELOAD: registered {new_count} new _cmd_* service(s)', 'info')

            src_label = ' (from src/)' if src_dir else ' (from build/)'
            self._append_log(f'HOT RELOAD: logic reloaded{src_label}', 'info')
        except Exception as e:
            import traceback
            self._append_log(f'HOT RELOAD FAILED: {e}', 'error')
            self._append_system_error(traceback.format_exc())

    def _save_gui_state(self):
        """Snapshot all GUI variable values before a GUI rebuild."""
        state = {}

        # FK tab
        state['sliders'] = {n: v.get() for n, v in self.sliders.items()}
        state['velocity_scale'] = self.velocity_scale_var.get()
        state['slider_driven'] = self._slider_driven
        state['last_speed_val'] = self._last_speed_val

        # IK tab
        state['xyz'] = {k: v.get() for k, v in self.xyz_vars.items()}
        state['rpy'] = {k: v.get() for k, v in self.rpy_vars.items()}
        state['ik_last_valid'] = dict(self._ik_last_valid)
        state['ik_valid'] = self._ik_valid
        state['ik_planned_target'] = self._ik_planned_target
        state['ik_status'] = self.ik_status_var.get()
        state['ee_labels'] = {k: v.get() for k, v in self.ee_labels.items()}

        # Grasp tab
        state['grasp_topic'] = self._grasp_topic_var.get()
        state['bbox_topic'] = self._bbox_topic_var.get()
        state['bbox_enabled'] = self._bbox_enabled_var.get()
        state['grasp_arm_duration'] = self._grasp_arm_duration_var.get()
        state['grasp_approach_height'] = self._grasp_approach_height_var.get()
        state['grasp_obj_z'] = self._grasp_obj_z_var.get()
        state['grasp_cross'] = self._grasp_cross_var.get()
        state['grasp_grip_close'] = self._grasp_grip_close_var.get()
        state['grasp_grip_open'] = self._grasp_grip_open_var.get()
        state['grasp_grip_duration'] = self._grasp_grip_duration_var.get()
        state['jaw_open_clearance'] = self._jaw_open_clearance_var.get()
        state['jaw_close_clearance'] = self._jaw_close_clearance_var.get()
        state['tcp_clearance'] = self._tcp_clearance_var.get()
        state['drop_hover_above_rim'] = self._drop_hover_above_rim_var.get()

        # Object listbox
        sel = self.obj_listbox.curselection()
        state['obj_selection'] = sel[0] if sel else None

        # Log contents
        state['process_log'] = self._process_log.get('1.0', 'end-1c')
        state['error_log'] = self._error_log.get('1.0', 'end-1c')
        state['log_tab'] = self._log_notebook.index(self._log_notebook.select())

        # Active notebook tab
        state['active_tab'] = self._notebook.index(self._notebook.select())

        return state

    def _restore_gui_state(self, state):
        """Restore GUI variable values after a GUI rebuild."""
        # FK tab
        for name, val in state.get('sliders', {}).items():
            if name in self.sliders:
                self.sliders[name].set(val)
                if name in self.slider_labels:
                    self.slider_labels[name].config(text=f'{val:.3f}')
        if 'velocity_scale' in state:
            self.velocity_scale_var.set(state['velocity_scale'])
        self._slider_driven = state.get('slider_driven', False)
        self._last_speed_val = state.get('last_speed_val', 0.5)

        # IK tab — suppress IK solves during batch restore
        self._ik_trace_active = False
        for k, v in state.get('xyz', {}).items():
            if k in self.xyz_vars:
                self.xyz_vars[k].set(v)
        for k, v in state.get('rpy', {}).items():
            if k in self.rpy_vars:
                self.rpy_vars[k].set(v)
        self._ik_trace_active = True
        self._ik_last_valid = state.get('ik_last_valid', {})
        self._ik_valid = state.get('ik_valid', True)
        self._ik_planned_target = state.get('ik_planned_target')
        self.ik_status_var.set(state.get('ik_status', 'Ready'))
        for k, v in state.get('ee_labels', {}).items():
            if k in self.ee_labels:
                self.ee_labels[k].set(v)

        # Grasp tab
        self._grasp_topic_var.set(state.get('grasp_topic', ''))
        self._bbox_topic_var.set(state.get('bbox_topic', ''))
        self._bbox_enabled_var.set(state.get('bbox_enabled', True))
        self._grasp_arm_duration_var.set(state.get('grasp_arm_duration', 2.5))
        self._grasp_approach_height_var.set(state.get('grasp_approach_height', 0.020))
        self._grasp_obj_z_var.set(state.get('grasp_obj_z', 0.0))
        self._grasp_cross_var.set(state.get('grasp_cross', False))
        self._grasp_grip_close_var.set(state.get('grasp_grip_close', -10.0))
        self._grasp_grip_open_var.set(state.get('grasp_grip_open', 100.0))
        self._grasp_grip_duration_var.set(state.get('grasp_grip_duration', 3.0))
        self._jaw_open_clearance_var.set(state.get('jaw_open_clearance', 5.0))
        self._jaw_close_clearance_var.set(state.get('jaw_close_clearance', 0.0))
        self._tcp_clearance_var.set(state.get('tcp_clearance', 1.0))
        self._drop_hover_above_rim_var.set(state.get('drop_hover_above_rim', 0.050))

        # Repopulate listbox from current objects_data
        self._populate_object_list()
        if state.get('obj_selection') is not None:
            idx = state['obj_selection']
            if idx < self.obj_listbox.size():
                self.obj_listbox.selection_set(idx)

        # Restore log contents
        self._process_log.config(state=tk.NORMAL)
        self._process_log.insert('1.0', state.get('process_log', ''))
        self._process_log.see(tk.END)
        self._process_log.config(state=tk.DISABLED)

        self._error_log.config(state=tk.NORMAL)
        self._error_log.insert('1.0', state.get('error_log', ''))
        self._error_log.see(tk.END)
        self._error_log.config(state=tk.DISABLED)

        # Restore active tabs
        if state.get('log_tab') is not None:
            try:
                self._log_notebook.select(state['log_tab'])
            except Exception:
                pass
        if state.get('active_tab') is not None:
            try:
                self._notebook.select(state['active_tab'])
            except Exception:
                pass

    def _hot_reload_gui(self):
        """Rebuild all GUI tabs with new code. Preserves state and ROS2 infra."""
        try:
            state = self._save_gui_state()
            self._hot_reload_logic()

            self._gui_ready = False
            self._ik_debounce_id = None
            # Widget registry is keyed by label + tab. Tabs are about to be
            # rebuilt from scratch, so the current entries are stale.
            # Flag services to short-circuit with "retry in 1s" until finalize.
            with self._widget_registry_lock:
                self._widget_registry.clear()
            self._widget_registry_rebuilding = True

            # Destroy notebook and log panel
            self._notebook.destroy()
            self._log_outer.destroy()

            # Recreate notebook
            notebook = ttk.Notebook(self.root)
            notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 0))
            self._notebook = notebook
            self._register_notebook(notebook, label='tab')

            # Rebuild tabs with error-safe wrappers
            for builder, tab_name in [
                (self._build_individual_tab, 'FK'),
                (self._build_arm_control_tab, 'IK'),
                (self._build_grasp_tab, 'Grasp'),
                (self._build_quickstart_tab, 'Quickstart'),
                (self._build_real_test_tab, 'Real Test'),
                (self._build_record_sim_tab, 'Record Sim'),
                (self._build_inference_tab, 'Inference'),
                (self._build_display_tab, 'RViz'),
            ]:
                try:
                    builder(notebook)
                except Exception as e:
                    err_frame = ttk.Frame(notebook)
                    notebook.add(err_frame, text=f'{tab_name} (ERROR)')
                    tk.Label(err_frame, text=f'Build error:\n{e}',
                             fg='red', wraplength=500, justify=tk.LEFT
                             ).pack(padx=10, pady=10)

            notebook.bind('<<NotebookTabChanged>>', self._on_tab_changed)

            # Rebuild log panel
            self._build_log_panel()

            # Restore all state
            self._restore_gui_state(state)

            self._finalize_widget_registry()
            self._gui_ready = True
            self._append_log('HOT RELOAD: GUI rebuilt (Ctrl+Shift+R)', 'info')

        except Exception as e:
            import traceback
            self._gui_ready = True
            self._append_log(f'HOT RELOAD GUI FAILED: {e}', 'error')
            self._append_system_error(traceback.format_exc())

    # ------------------------------------------------------------------
    # Tab 2: Arm Control (IK via MoveIt)
    # ------------------------------------------------------------------

    def _build_arm_control_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='IK')
        self._ik_tab_frame = frame

        # --- Arm (XYZ + RPY) side by side ---
        coord_frame = ttk.LabelFrame(frame, text='Arm')
        coord_frame.pack(fill=tk.X, padx=10, pady=5)

        self.xyz_vars = {}
        self._ik_spinboxes = {}    # field -> Spinbox widget (for color changes)
        self._ik_last_valid = {}   # field -> last value that produced valid IK
        self._ik_solve_gen = 0     # generation counter for async IK results
        self._ik_solve_lock = threading.Lock()  # prevent concurrent rclpy.spin
        self._ik_debounce_id = None
        self._ik_trace_active = True  # guard to suppress traces during programmatic updates

        columns_row = tk.Frame(coord_frame)
        columns_row.pack(fill=tk.X, padx=5, pady=2)

        # Left column: XYZ
        xyz_col = tk.Frame(columns_row)
        xyz_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(xyz_col, text='Position', font=('Arial', 9, 'bold')).pack(anchor='w', padx=2)
        for label, default in [('X', 0.12), ('Y', 0.0), ('Z', 0.15)]:
            lo, hi = WORKSPACE_BOUNDS[label]
            row = tk.Frame(xyz_col)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=f'{label}:', width=3).pack(side=tk.LEFT)
            var = tk.DoubleVar(value=default)
            spin = self._register_spinbox(
                row, label=label, tab='IK', section='Position',
                textvariable=var, from_=lo, to=hi,
                increment=0.01, width=8, format='%.3f')
            spin.pack(side=tk.LEFT, padx=3)
            var.trace_add('write', lambda *a, f=label: self._on_ik_var_changed(f))
            self.xyz_vars[label] = var
            self._ik_spinboxes[label] = spin
            self._ik_last_valid[label] = default

        # Right column: RPY
        rpy_col = tk.Frame(columns_row)
        rpy_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(rpy_col, text='Orientation', font=('Arial', 9, 'bold')).pack(anchor='w', padx=2)
        self.rpy_vars = {}
        for comp, default in [('Roll', 0.0), ('shoulder_lift', 0.0), ('Yaw', 0.0)]:
            row = tk.Frame(rpy_col)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=f'{comp[0]}:', width=3).pack(side=tk.LEFT)
            var = tk.DoubleVar(value=default)
            spin = self._register_spinbox(
                row, label=comp, tab='IK', section='Orientation',
                textvariable=var, from_=-180.0, to=180.0,
                increment=1.0, width=8, format='%.1f')
            spin.pack(side=tk.LEFT, padx=3)
            var.trace_add('write', lambda *a, f=comp: self._on_ik_var_changed(f))
            self.rpy_vars[comp] = var
            self._ik_spinboxes[comp] = spin
            self._ik_last_valid[comp] = default

        # Buttons inside Arm frame
        arm_btn_frame = tk.Frame(coord_frame)
        arm_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        self._build_arm_btn_row(arm_btn_frame, tab='IK',
                                reset_cmd=self._cmd_ik_reset)

        # --- Gripper Section (shares DoubleVar with FK tab) ---
        gripper_frame2 = ttk.LabelFrame(frame, text='Gripper')
        gripper_frame2.pack(fill=tk.X, padx=10, pady=(2, 2))

        jaw_row2 = tk.Frame(gripper_frame2)
        jaw_row2.pack(fill=tk.X, padx=5, pady=2)
        lo, hi = JOINT_LIMITS[GRIPPER_JOINT_NAME]
        tk.Label(jaw_row2, text=GRIPPER_JOINT_NAME, width=14, anchor='w').pack(side=tk.LEFT, padx=(5, 0))
        # Reuse the FK tab's DoubleVar so both sliders stay in sync automatically
        jaw_var_shared = self.sliders[GRIPPER_JOINT_NAME]
        self._ik_jaw_slider = self._register_scale(
            jaw_row2, label=f'{GRIPPER_JOINT_NAME}@IK',
            tab='IK', section='Joint Sliders',
            variable=jaw_var_shared, from_=lo, to=hi,
            orient=tk.HORIZONTAL, resolution=0.001, length=300,
            command=lambda val: self._on_slider(GRIPPER_JOINT_NAME, float(val)))
        self._ik_jaw_slider.pack(side=tk.LEFT, padx=5)
        self._ik_jaw_label = tk.Label(jaw_row2, text='0.000', width=8)
        self._ik_jaw_label.pack(side=tk.LEFT)

        gripper_btn_frame2 = tk.Frame(gripper_frame2)
        gripper_btn_frame2.pack(fill=tk.X, padx=5, pady=5)
        self._build_gripper_btn_row(gripper_btn_frame2, tab='IK')

        # --- Action buttons: Set Joints / Plan & Execute ---
        ik_btn_frame = tk.Frame(frame)
        ik_btn_frame.pack(fill=tk.X, padx=10, pady=5)
        self._register_button(ik_btn_frame, text='Set Joints', tab='IK', section='Actions',
                              command=self._ik_btn_set_joints).pack(side=tk.LEFT, padx=5)
        self._register_button(ik_btn_frame, text='Plan & Execute', tab='IK', section='Actions',
                              command=self._ik_btn_plan_execute).pack(side=tk.LEFT, padx=5)

        # IK state tracking
        self.ik_status_var = tk.StringVar(value='Ready')
        self._ik_valid = True
        self._ik_planned_target = None

        # Hidden EE pose storage — used by services
        self.ee_labels = {}
        for key in ['X', 'Y', 'Z', 'qx', 'qy', 'qz', 'qw']:
            self.ee_labels[key] = tk.StringVar(value='---')

    def _build_grasp_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='Grasp')

        # --- Object Source ---
        topic_frame = ttk.LabelFrame(frame, text='Object Source')
        topic_frame.pack(fill=tk.X, padx=10, pady=5)

        topic_row = tk.Frame(topic_frame)
        topic_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(topic_row, text='Topic:', anchor='w').pack(side=tk.LEFT)
        default_topic = '/objects_poses_real' if self.use_real_hardware else '/objects_poses_sim'
        self._grasp_topic_var = tk.StringVar(value=default_topic)
        self._register_entry(topic_row, label='Grasp Topic', tab='Grasp', section='Topic',
                             textvariable=self._grasp_topic_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        self._register_button(topic_row, text='Update Topic', tab='Grasp', section='Topic',
                              command=self._cmd_grasp_update_topic).pack(side=tk.RIGHT, padx=(2, 0))
        self._register_button(topic_row, text='Refresh Objects', tab='Grasp', section='Topic',
                              command=self._cmd_grasp_refresh).pack(side=tk.RIGHT, padx=(2, 0))

        opts_row = tk.Frame(topic_frame)
        opts_row.pack(fill=tk.X, padx=5, pady=2)
        default_bbox_topic = '/objects_bbox_real' if self.use_real_hardware else '/objects_bbox_sim'
        tk.Label(opts_row, text='BBox:', anchor='w').pack(side=tk.LEFT)
        self._bbox_topic_var = tk.StringVar(value=default_bbox_topic)
        self._register_entry(opts_row, label='BBox Topic', tab='Grasp', section='Topic',
                             textvariable=self._bbox_topic_var, width=22).pack(
            side=tk.LEFT, padx=(5, 5))
        self._bbox_enabled_var = tk.BooleanVar(value=True)
        self._register_check(opts_row, label='TCP offset', tab='Grasp', section='Topic',
                             variable=self._bbox_enabled_var).pack(side=tk.LEFT, padx=5)

        # --- Detected Objects ---
        obj_frame = ttk.LabelFrame(frame, text='Detected Objects')
        obj_frame.pack(fill=tk.X, padx=10, pady=5)

        self.obj_listbox = self._register_listbox(
            obj_frame, label='Detected Objects', tab='Grasp',
            section='Detected Objects',
            height=6, font=('Consolas', 9),
            selectbackground='#d0d0d0', selectforeground='#1a1a1a')
        self.obj_listbox.pack(fill=tk.X, padx=5, pady=2)

        # --- Arm | Gripper columns ---
        ctrl_cols = ttk.Frame(frame)
        ctrl_cols.pack(fill=tk.X, padx=10, pady=5)

        # Left column: Arm
        arm_col = ttk.LabelFrame(ctrl_cols, text='Arm')
        arm_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 3))

        arm_dur_row = tk.Frame(arm_col)
        arm_dur_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(arm_dur_row, text='Duration (s):', anchor='w').pack(side=tk.LEFT)
        self._grasp_arm_duration_var = tk.DoubleVar(value=2.5)
        self._register_spinbox(arm_dur_row, label='Arm Duration (s)',
                               tab='Grasp', section='Arm',
                               textvariable=self._grasp_arm_duration_var,
                               from_=0.5, to=10.0, increment=0.5,
                               width=8, format='%.1f').pack(side=tk.LEFT, padx=(5, 0))

        approach_row = tk.Frame(arm_col)
        approach_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(approach_row, text='Approach height (m):', anchor='w').pack(side=tk.LEFT)
        # Phase 9: 20mm chosen over the original 50mm to avoid a grasp-
        # race: if descent's OMPL plan fails, grasp_move still returns
        # success while the arm is stuck at approach height — gripper_close
        # then attaches the block at the wrong tcp offset (trace showed
        # block_in_tcp.z=0.050 instead of ~0). Keeping approach_h small
        # bounds the worst-case attach offset to ~20mm rather than 50mm.
        # Reachability at 20mm is 100% across BLOCK_RANDOM bounds
        # (compute_grasp_reachability.py). 50mm fails 7% of cells.
        self._grasp_approach_height_var = tk.DoubleVar(value=0.020)
        self._register_spinbox(approach_row, label='Approach height (m)',
                               tab='Grasp', section='Arm',
                               textvariable=self._grasp_approach_height_var,
                               from_=0.00, to=0.20, increment=0.01,
                               width=6, format='%.2f').pack(side=tk.LEFT, padx=(5, 0))

        obj_z_row = tk.Frame(arm_col)
        obj_z_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(obj_z_row, text='Object Z (m):', anchor='w').pack(side=tk.LEFT)
        self._grasp_obj_z_var = tk.DoubleVar(value=0.0)
        self._register_spinbox(obj_z_row, label='Object Z (m)',
                               tab='Grasp', section='Arm',
                               textvariable=self._grasp_obj_z_var,
                               from_=-0.05, to=0.20, increment=0.005,
                               width=6, format='%.3f').pack(side=tk.LEFT, padx=(5, 0))

        self._grasp_cross_var = tk.BooleanVar(value=False)
        self._register_check(arm_col, label='Cross-axis grasp',
                             tab='Grasp', section='Arm',
                             variable=self._grasp_cross_var).pack(
            fill=tk.X, padx=5, pady=1)

        self._register_button(arm_col, text='Home', tab='Grasp', section='Arm',
                              command=self._cmd_grasp_home).pack(fill=tk.X, padx=5, pady=2)

        # grasp_home velocity_scale override. 0.0 = no override (use the
        # planning default); anything > 0 swaps velocity_scale_var for
        # just the grasp_home plan, then restores. Lets the lag-vs-speed
        # knob live in the GUI so it's tunable without ros2 param set.
        # Mirrors home_velocity_scale ROS2 parameter via _cmd_apply_home_speed.
        home_speed_row = tk.Frame(arm_col)
        home_speed_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(home_speed_row, text='Home speed scale:',
                 anchor='w').pack(side=tk.LEFT)
        self._home_speed_var = tk.DoubleVar(value=0.0)
        self._register_spinbox(home_speed_row, label='Home speed scale',
                               tab='Grasp', section='Arm',
                               textvariable=self._home_speed_var,
                               from_=0.0, to=1.0, increment=0.05,
                               format='%.2f', width=6).pack(side=tk.LEFT, padx=(5, 0))
        self._register_button(home_speed_row, text='Apply', tab='Grasp',
                              section='Arm',
                              command=self._cmd_apply_home_speed).pack(
                              side=tk.LEFT, padx=(5, 0))

        self._grasp_move_btn = self._register_button(
            arm_col, text='Move to Grab', tab='Grasp', section='Arm',
            command=self._cmd_grasp_move)
        self._grasp_move_btn.pack(fill=tk.X, padx=5, pady=2)

        # Right column: Gripper
        grip_col = ttk.LabelFrame(ctrl_cols, text='Gripper')
        grip_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(3, 0))

        _jaw_min_deg = math.degrees(JOINT_LIMITS['gripper_joint'][0])
        _jaw_max_deg = math.degrees(JOINT_LIMITS['gripper_joint'][1])
        grip_range_row = tk.Frame(grip_col)
        grip_range_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(grip_range_row, text='Range:', anchor='w').pack(side=tk.LEFT)
        self._grasp_grip_close_var = tk.DoubleVar(value=_jaw_min_deg)
        self._register_spinbox(grip_range_row, label='Grip Close',
                               tab='Grasp', section='Gripper',
                               textvariable=self._grasp_grip_close_var,
                               from_=_jaw_min_deg, to=_jaw_max_deg,
                               increment=5, width=5, format='%.0f').pack(side=tk.LEFT, padx=(5, 0))
        tk.Label(grip_range_row, text='-').pack(side=tk.LEFT)
        self._grasp_grip_open_var = tk.DoubleVar(value=_jaw_max_deg)
        self._register_spinbox(grip_range_row, label='Grip Open',
                               tab='Grasp', section='Gripper',
                               textvariable=self._grasp_grip_open_var,
                               from_=_jaw_min_deg, to=_jaw_max_deg,
                               increment=5, width=5, format='%.0f').pack(side=tk.LEFT)
        tk.Label(grip_range_row, text='\u00b0').pack(side=tk.LEFT)

        grip_dur_row = tk.Frame(grip_col)
        grip_dur_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(grip_dur_row, text='Duration (s):', anchor='w').pack(side=tk.LEFT)
        self._grasp_grip_duration_var = tk.DoubleVar(value=3.0)
        self._register_spinbox(grip_dur_row, label='Grip Duration (s)',
                               tab='Grasp', section='Gripper',
                               textvariable=self._grasp_grip_duration_var,
                               from_=0.2, to=5.0, increment=0.1,
                               width=8, format='%.1f').pack(side=tk.LEFT, padx=(5, 0))

        clearance_row = tk.Frame(grip_col)
        clearance_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(clearance_row, text='Open clearance (mm):', anchor='w').pack(side=tk.LEFT)
        self._jaw_open_clearance_var = tk.DoubleVar(value=JAW_OPEN_CLEARANCE_M * 1000)
        self._register_spinbox(clearance_row, label='Open clearance (mm)',
                               tab='Grasp', section='Gripper',
                               textvariable=self._jaw_open_clearance_var,
                               from_=-5.0, to=20.0, increment=0.5,
                               width=5, format='%.1f').pack(side=tk.LEFT, padx=(5, 0))

        close_cl_row = tk.Frame(grip_col)
        close_cl_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(close_cl_row, text='Close clearance (mm):', anchor='w').pack(side=tk.LEFT)
        self._jaw_close_clearance_var = tk.DoubleVar(value=JAW_CLOSE_CLEARANCE_M * 1000)
        self._register_spinbox(close_cl_row, label='Close clearance (mm)',
                               tab='Grasp', section='Gripper',
                               textvariable=self._jaw_close_clearance_var,
                               from_=-10.0, to=10.0, increment=0.5,
                               width=5, format='%.1f').pack(side=tk.LEFT, padx=(5, 0))

        tcp_clear_row = tk.Frame(grip_col)
        tcp_clear_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(tcp_clear_row, text='TCP clearance (mm):', anchor='w').pack(side=tk.LEFT)
        self._tcp_clearance_var = tk.DoubleVar(value=TCP_CLEARANCE_M * 1000)
        self._register_spinbox(tcp_clear_row, label='TCP clearance (mm)',
                               tab='Grasp', section='Gripper',
                               textvariable=self._tcp_clearance_var,
                               from_=-5.0, to=10.0, increment=0.5,
                               width=5, format='%.1f').pack(side=tk.LEFT, padx=(5, 0))

        grip_btn_row1 = tk.Frame(grip_col)
        grip_btn_row1.pack(fill=tk.X, padx=5, pady=2)
        self._register_button(grip_btn_row1, text='Grasp Open', tab='Grasp', section='Gripper',
                              command=self._cmd_gripper_open_for_object).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self._register_button(grip_btn_row1, text='Grasp Close', tab='Grasp', section='Gripper',
                              command=self._cmd_gripper_close_for_object).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        grip_btn_row2 = tk.Frame(grip_col)
        grip_btn_row2.pack(fill=tk.X, padx=5, pady=2)
        self._register_button(grip_btn_row2, text='Open', tab='Grasp', section='Gripper Range',
                              command=self._cmd_gripper_open_range).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self._register_button(grip_btn_row2, text='Close', tab='Grasp', section='Gripper Range',
                              command=self._cmd_gripper_close_range).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        # --- Drop Targets ---
        drop_src_frame = ttk.LabelFrame(frame, text='Drop Source')
        drop_src_frame.pack(fill=tk.X, padx=10, pady=5)

        drop_topic_row = tk.Frame(drop_src_frame)
        drop_topic_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(drop_topic_row, text='Topic:', anchor='w').pack(side=tk.LEFT)
        self._drop_topic_var = tk.StringVar(value='/drop_poses')
        self._register_entry(drop_topic_row, label='Drop Topic',
                             tab='Grasp', section='Drop Source',
                             textvariable=self._drop_topic_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        self._register_button(drop_topic_row, text='Update Drop Topic', tab='Grasp', section='Drop Source',
                              command=self._drop_btn_update_topic).pack(side=tk.RIGHT, padx=(2, 0))
        self._register_button(drop_topic_row, text='Refresh Drops', tab='Grasp', section='Drop Source',
                              command=self._cmd_drop_refresh).pack(side=tk.RIGHT, padx=(2, 0))

        drop_list_frame = ttk.LabelFrame(frame, text='Drop Targets')
        drop_list_frame.pack(fill=tk.X, padx=10, pady=5)
        self._drop_listbox = self._register_listbox(
            drop_list_frame, label='Drop Targets',
            tab='Grasp', section='Drop Targets',
            height=3, font=('Consolas', 9),
            selectbackground='#d0d0d0', selectforeground='#1a1a1a')
        self._drop_listbox.pack(fill=tk.X, padx=5, pady=2)

        drop_frame = ttk.LabelFrame(frame, text='Drop')
        drop_frame.pack(fill=tk.X, padx=10, pady=5)

        drop_dur_row = tk.Frame(drop_frame)
        drop_dur_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(drop_dur_row, text='Sweep duration (s):', anchor='w').pack(side=tk.LEFT)
        # 3.0 s default matches grasp_home / grasp_move / drop_point —
        # gives the controller time to track the wrist_flex sweep without
        # accumulating the multi-degree lag that bites at higher rates.
        self._drop_duration_var = tk.DoubleVar(value=3.0)
        self._register_spinbox(drop_dur_row, label='Sweep Duration (s)',
                               tab='Grasp', section='Drop',
                               textvariable=self._drop_duration_var,
                               from_=0.5, to=10.0, increment=0.5,
                               width=8, format='%.1f').pack(side=tk.LEFT, padx=(5, 0))

        # Hover above cup rim for drop sweep — the block center lands at
        # cup_center_z + CUP_BODY_HEIGHT_M/2 + this value, i.e. the cup's
        # rim height + this clearance. Default 30 mm is tuned for the
        # current 97 mm cup: enough clearance that the block falls past
        # the rim cleanly, small enough that drop impact is minimal.
        drop_hover_row = tk.Frame(drop_frame)
        drop_hover_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(drop_hover_row, text='Hover above rim (m):',
                 anchor='w').pack(side=tk.LEFT)
        # Default 50 mm: gives the attached block's bounding box clearance
        # above the cup rim during the wrist_flex sweep (90°→45°). At 30 mm,
        # the block swept arc clipped cup walls at 1-3 mm; OMPL used to
        # paper over this with tortuous paths, deterministic planner
        # surfaces it honestly. User-tunable if a specific lego/cup combo
        # needs more/less clearance.
        self._drop_hover_above_rim_var = tk.DoubleVar(value=0.050)
        self._register_spinbox(drop_hover_row, label='Hover Above Rim (m)',
                               tab='Grasp', section='Drop',
                               textvariable=self._drop_hover_above_rim_var,
                               from_=0.00, to=0.15, increment=0.005,
                               width=8, format='%.3f').pack(side=tk.LEFT, padx=(5, 0))

        drop_btn_row = tk.Frame(drop_frame)
        drop_btn_row.pack(fill=tk.X, padx=5, pady=2)
        self._register_button(drop_btn_row, text='Point to Drop', tab='Grasp', section='Drop',
                              command=self._cmd_drop_point).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self._register_button(drop_btn_row, text='Sweep to Drop', tab='Grasp', section='Drop',
                              command=self._cmd_drop_sweep).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 2))
        self._register_button(drop_btn_row, text='Release', tab='Grasp', section='Drop',
                              command=self._cmd_drop_release).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        # --- MTC Spike (disabled) ---
        # The Pick & Drop button is commented out; backend (_cmd_mtc_pick_place,
        # _mtc_run_client, and the so_arm101_mtc C++ node) is kept intact so
        # we can re-enable this quickly for the IK iteration spike. Toggle the
        # three lines below ON to restore the button.
        # mtc_frame = ttk.LabelFrame(frame, text='MTC (spike)')
        # mtc_frame.pack(fill=tk.X, padx=10, pady=5)
        # mtc_btn_row = tk.Frame(mtc_frame)
        # mtc_btn_row.pack(fill=tk.X, padx=5, pady=2)
        # self._register_button(mtc_btn_row, text='MTC Pick & Drop',
        #                       tab='Grasp', section='MTC',
        #                       command=self._cmd_mtc_pick_place).pack(
        #                           side=tk.LEFT, fill=tk.X, expand=True)

        # Start drop subscription immediately
        self._update_drop_topic(self._drop_topic_var.get())

        # Initial subscription to default topic
        self._cmd_grasp_update_topic()

    # ------------------------------------------------------------------
    # Quickstart tab — high-level player for the full pick-drop lifecycle.
    # Each step just delegates to the matching _cmd_* handler on the Grasp
    # tab, so all settings (sweep duration, hover, padding, topic, grip
    # duration, …) continue to flow through from the Grasp tab untouched.
    # ------------------------------------------------------------------

    # Pick-and-drop step sequence. Each tuple: (label, callable, kwargs).
    # The callable is bound on the class, so we reference it by name and
    # resolve with getattr(self, name) in the runner — keeps this table
    # readable and independent of method order.
    # Phase 10.2: 'point to drop cup' (_cmd_drop_point) removed — drop_sweep
    # is now self-contained and does its own face-cup pan internally before
    # the IK-planned tilt. The Drop Point button in the GUI still works for
    # manual use; the auto-sequence just doesn't dispatch it as a separate
    # step anymore.


    def _build_quickstart_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='Quickstart')

        # Player state (thread/events).
        self._qs_state = 'idle'             # 'idle' | 'running' | 'paused'
        self._qs_thread = None
        # set() = running; clear() = pause-wall (next step blocks on wait)
        self._qs_resume_evt = threading.Event()
        self._qs_resume_evt.set()
        # set() = abort at next step boundary
        self._qs_abort_evt = threading.Event()
        self._qs_status_var = tk.StringVar(value='Idle')
        self._qs_step_var = tk.StringVar(value='—')

        # ===== TOP: Refresh-all bar =====
        top_bar = ttk.Frame(frame)
        top_bar.pack(fill=tk.X, padx=10, pady=(8, 4))
        self._register_button(
            top_bar, text='🔄  Refresh all (objects + drops)',
            tab='Quickstart', section='Top',
            command=self._cmd_qs_refresh_all,
        ).pack(fill=tk.X, ipady=4)

        # ===== BODY: two columns =====
        body = ttk.Frame(frame)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        # --- LEFT column: object listbox ---
        left = ttk.LabelFrame(body, text='Detected objects  (pick one)')
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self._qs_listbox = tk.Listbox(
            left, height=16, exportselection=False,
            font=('TkFixedFont', 10))
        self._qs_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- RIGHT column: player + steps ---
        right = ttk.Frame(body)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))

        ctrl_frame = ttk.LabelFrame(right, text='Pick & Drop Player')
        ctrl_frame.pack(fill=tk.X, padx=0, pady=(0, 4))
        ctrl_row = ttk.Frame(ctrl_frame)
        ctrl_row.pack(fill=tk.X, padx=5, pady=5)
        self._register_button(
            ctrl_row, text='▶ Play',
            tab='Quickstart', section='Player',
            command=self._cmd_qs_play,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2), ipady=2)
        self._register_button(
            ctrl_row, text='⏸ Pause',
            tab='Quickstart', section='Player',
            command=self._cmd_qs_pause,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 2), ipady=2)
        self._register_button(
            ctrl_row, text='⏮ Restart',
            tab='Quickstart', section='Player',
            command=self._cmd_qs_restart,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0), ipady=2)
        status_row = ttk.Frame(ctrl_frame)
        status_row.pack(fill=tk.X, padx=5, pady=(0, 5))
        tk.Label(status_row, text='Status:', anchor='w').pack(side=tk.LEFT)
        tk.Label(status_row, textvariable=self._qs_status_var,
                 anchor='w', font=('TkDefaultFont', 9, 'bold')).pack(
                     side=tk.LEFT, padx=(5, 15))
        tk.Label(status_row, text='Step:', anchor='w').pack(side=tk.LEFT)
        tk.Label(status_row, textvariable=self._qs_step_var,
                 anchor='w').pack(side=tk.LEFT, padx=(5, 0))

        # Individual step buttons — right column below Player
        steps_frame = ttk.LabelFrame(
            right, text='Individual steps  (Grasp-tab handlers)')
        steps_frame.pack(fill=tk.BOTH, expand=True)
        for (label, method_name, kwargs) in self._QS_SEQUENCE:
            row = ttk.Frame(steps_frame)
            row.pack(fill=tk.X, padx=5, pady=1)
            tk.Label(row, text=f'{label}', anchor='w', width=22).pack(side=tk.LEFT)
            self._register_button(
                row, text='Run',
                tab='Quickstart', section=label,
                command=lambda m=method_name, kw=kwargs: self._qs_run_step_oneshot(m, kw)
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # (MTC spike re-enable: uncomment a button row here pointing at
        # self._cmd_mtc_pick_place when iterating on the MTC task graph.)

        # Initial populate — use refresh_all so legos AND drops get a fresh
        # pass on first tab build. Deferred via root.after so the grasp tab
        # (source of the legacy listbox) is already constructed.
        self.root.after(800, self._cmd_qs_refresh_all)

    # ---- Quickstart helpers ----

    def _cmd_qs_refresh_all(self):
        """Hit both Grasp-tab refresh buttons (objects AND drops), then
        repopulate our local listbox once the topic subscriptions have
        received fresh data. Non-blocking — schedules the local repopulate
        via root.after(1000, ...) because _cmd_grasp_refresh uses 500 ms
        and _cmd_drop_refresh waits up to 2 s for fresh /drop_poses data.
        """
        self._append_log('Quickstart: refresh (objects + drops)')
        try:
            self._cmd_grasp_refresh()
        except Exception as exc:
            self._append_log(f'Quickstart refresh objects failed: {exc}', 'warn')
        try:
            self._cmd_drop_refresh()
        except Exception as exc:
            self._append_log(f'Quickstart refresh drops failed: {exc}', 'warn')
        # Our local listbox mirrors self.objects_data which _cmd_grasp_refresh
        # repopulates after ~500 ms — wait a bit longer for safety.
        self.root.after(1200, self._qs_refresh_objects)

    def _cmd_qs_select(self):
        """Select an entry in the Quickstart listbox by name (ik_target param)
        or first item. Mirrors _cmd_grasp_select but targets _qs_listbox so
        _cmd_qs_play passes its _qs_get_selected_object() guard when driven
        headlessly over ROS services.
        """
        if not hasattr(self, '_qs_listbox') or self._qs_listbox.size() == 0:
            self._append_log('Quickstart: no objects to select — run qs_refresh_all first', 'warn')
            self._cmd_error = 'qs listbox empty'
            return
        name_hint = self.get_parameter('ik_target').get_parameter_value().string_value.strip()
        target_idx = 0
        if name_hint and '=' not in name_hint:
            matched = False
            for i in range(self._qs_listbox.size()):
                if self._qs_listbox.get(i).split('  ')[0] == name_hint:
                    target_idx = i
                    matched = True
                    break
            if not matched:
                self._append_log(f'Quickstart: "{name_hint}" not in listbox — defaulting to first', 'warn')
        self._qs_listbox.selection_clear(0, tk.END)
        self._qs_listbox.selection_set(target_idx)
        self._qs_listbox.see(target_idx)
        picked = self._qs_listbox.get(target_idx).split('  ')[0]
        self._append_log(f'Quickstart selected: {picked}')


    def _qs_refresh_objects(self):
        """Populate the Quickstart listbox from the shared objects_data dict
        (same source as the Grasp tab's listbox). Filters out legos that
        have already been placed in a cup — they're no longer pick
        candidates, and trying to grasp them fails the workspace check
        (cups sit at the edge of the drop-IK envelope, often beyond the
        grasp-IK r_max). Preserves selection if the previously-selected
        entry still exists and is still graspable.
        """
        prev = None
        if self._qs_listbox.curselection():
            prev = self._qs_listbox.get(self._qs_listbox.curselection()[0]).split('  ')[0]
        self._qs_listbox.delete(0, tk.END)
        with self.objects_lock:
            objects_snapshot = dict(self.objects_data)
        names = sorted(objects_snapshot.keys())
        skipped_in_cup = []
        restore_idx = None
        listbox_idx = 0
        for n in names:
            pose = objects_snapshot.get(n, {})
            if self._lego_is_in_cup(pose):
                skipped_in_cup.append(n)
                continue
            x, y, z = pose.get('x', 0), pose.get('y', 0), pose.get('z', 0)
            self._qs_listbox.insert(
                tk.END, f'{n:<18s}  x={x:+.3f} y={y:+.3f} z={z:+.3f}')
            if n == prev:
                restore_idx = listbox_idx
            listbox_idx += 1
        if restore_idx is not None:
            self._qs_listbox.selection_set(restore_idx)
        if skipped_in_cup:
            self._append_log(
                f'Quickstart: skipped {len(skipped_in_cup)} in-cup lego(s): '
                f'{skipped_in_cup}')

    def _qs_get_selected_object(self):
        sel = self._qs_listbox.curselection()
        if not sel:
            return None
        return self._qs_listbox.get(sel[0]).split('  ')[0]

    def _qs_set_status(self, status, step=None):
        """Thread-safe status/step update via Tk's event loop."""
        def _update():
            self._qs_status_var.set(status)
            if step is not None:
                self._qs_step_var.set(step)
        self.root.after(0, _update)


    def _qs_sync_grasp_listbox(self, obj_name):
        """Mirror the Quickstart selection into the Grasp tab's obj_listbox
        so every _cmd_* downstream reads the same selection. Waits for
        population if the listbox is empty (_cmd_grasp_refresh populates
        asynchronously via root.after — 0.5 s was not enough).
        """
        if not hasattr(self, 'obj_listbox') or self.obj_listbox.size() == 0:
            self._cmd_grasp_refresh()
            deadline = time.time() + 3.0
            while time.time() < deadline and (
                    not hasattr(self, 'obj_listbox') or
                    self.obj_listbox.size() == 0):
                time.sleep(0.1)
        for i in range(self.obj_listbox.size()):
            if self.obj_listbox.get(i).split('  ')[0] == obj_name:
                self.obj_listbox.selection_clear(0, tk.END)
                self.obj_listbox.selection_set(i)
                return True
        return False

    def _qs_sync_drop_listbox(self, drop_name):
        """Same pattern for the drop listbox — _cmd_drop_refresh waits up
        to 2 s for fresh /drop_poses, so wait that long for population."""
        if not hasattr(self, '_drop_listbox') or self._drop_listbox.size() == 0:
            self._cmd_drop_refresh()
            deadline = time.time() + 3.0
            while time.time() < deadline and (
                    not hasattr(self, '_drop_listbox') or
                    self._drop_listbox.size() == 0):
                time.sleep(0.1)
        for i in range(self._drop_listbox.size()):
            entry = self._drop_listbox.get(i).split(' [')[0]
            if entry == drop_name:
                self._drop_listbox.selection_clear(0, tk.END)
                self._drop_listbox.selection_set(i)
                return True
        return False


    # ---- Step-runner entry points ----

    def _qs_run_step_oneshot(self, method_name, kwargs):
        """Run a single step from the individual-step buttons. Non-blocking
        on the Tk thread; the step itself may spawn its own motion thread.
        Skips if the Player is currently running (to avoid interleaving)."""
        if self._qs_state == 'running':
            self._append_log(
                'Quickstart step ignored: Player is running — use Pause first', 'warn')
            return
        self._qs_set_status('Single step', step=method_name)
        threading.Thread(
            target=self._qs_execute_step,
            args=(method_name, kwargs),
            daemon=True).start()


    # ---- Player controls ----

    def _cmd_qs_play(self):
        """Play: start the lifecycle from the first step, OR resume from pause."""
        if self._qs_state == 'paused':
            self._qs_set_status('Running')
            self._qs_resume_evt.set()
            self._qs_state = 'running'
            return
        if self._qs_state == 'running':
            self._append_log('Quickstart already running', 'warn')
            return
        if self._qs_get_selected_object() is None:
            self._append_log('Quickstart Play: select an object first', 'err')
            return
        self._qs_abort_evt.clear()
        self._qs_resume_evt.set()
        self._qs_state = 'running'
        self._qs_thread = threading.Thread(target=self._qs_run, daemon=True)
        self._qs_thread.start()

    def _cmd_qs_pause(self):
        if self._qs_state != 'running':
            self._append_log('Quickstart Pause: not running', 'warn')
            return
        self._qs_resume_evt.clear()
        self._qs_state = 'paused'
        self._qs_set_status('Paused')

    def _cmd_qs_restart(self):
        """Abort current cycle and return to idle. Does NOT restart a new
        cycle automatically — user presses Play again (intentional, so a
        mid-motion abort doesn't race into a fresh motion command).
        """
        self._qs_abort_evt.set()
        self._qs_resume_evt.set()  # unblock any paused wait so the runner exits
        self._qs_state = 'idle'
        self._qs_set_status('Aborted', step='—')


    # ------------------------------------------------------------------
    # Tab: Real Test  (real-mode pipeline: YOLOE legos + ArUco cups)
    # ------------------------------------------------------------------
    # Mirrors the Quickstart flow but sources poses from the real-camera
    # detection stack (/objects_poses_real, /drop_poses_real) instead of
    # sim ground truth. Cups are detected ONCE per cycle at a scan pose
    # and cached; legos refresh every cycle. Built in isolation here for
    # end-to-end validation, will refactor into existing tabs once stable.

    def _ensure_real_state(self):
        """Lazy-init Real Test tab state vars on the live instance.

        __init__ runs once per process, so when hot-reload introduces new
        state vars they don't materialize on the running instance until
        they're accessed. This helper is called at the top of the tab
        builder and every _cmd_real_* handler so the running session
        picks up the new vars without a full relaunch.
        """
        if not hasattr(self, '_cached_cup_poses'):
            self._cached_cup_poses = {}
        if not hasattr(self, '_cached_lego_poses'):
            # Same shape as _cached_cup_poses + objects_data:
            # {child_frame_id (e.g. 'red_lego_0') →
            #  {'x','y','z','qx','qy','qz','qw'}} in frame 'base'.
            # Populated by Refresh Legos Pose at grasp_home (where wrist
            # camera looks down at workspace). Consumed by Run Real Pick
            # & Drop instead of the live /objects_poses_real feed.
            self._cached_lego_poses = {}
        if not hasattr(self, '_real_selected_color_var'):
            # Bound to the color dropdown in Real Test → Run.
            # '' = nothing selected (also the empty-state when no color has
            # both a cached lego AND a cached cup).
            self._real_selected_color_var = tk.StringVar(value='')
        if not hasattr(self, '_real_sweep_range_deg_var'):
            # Drop Scan sweep range in degrees from start point. Default 90°
            # (instead of full 180°) so the user can constrain the sweep to
            # the side where cups are likely to be.
            self._real_sweep_range_deg_var = tk.IntVar(value=90)
        if not hasattr(self, '_real_sweep_reversed_var'):
            # False = canonical direction (start at +90°, sweep right→left
            # toward -90°). True = swap (start at -90°, sweep left→right).
            self._real_sweep_reversed_var = tk.BooleanVar(value=False)

    def _build_real_test_tab(self, notebook):
        self._ensure_real_state()
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='Real Test')

        # ===== Section: Setup =====
        # Three primitive motions used to position the arm before scanning,
        # plus the wrist-roll sign toggle that decides which pose makes the
        # cup ArUco markers visible to the wrist camera.
        setup = ttk.LabelFrame(frame, text='Setup')
        setup.pack(fill=tk.X, padx=10, pady=(8, 4))

        row1 = ttk.Frame(setup)
        row1.pack(fill=tk.X, padx=5, pady=4)
        self._register_button(
            row1, text='Grasp Home', tab='Real Test', section='Setup',
            command=self._cmd_grasp_home,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2), ipady=3)
        self._register_button(
            row1, text='Open Gripper', tab='Real Test', section='Setup',
            command=self._cmd_gripper_open,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0), ipady=3)

        # Row 2: explicit wrist_roll set buttons. User picks +90° (markers
        # in FOV / scan pose) or −90° (jaw-clearance / drop pose) BEFORE
        # pressing Drop Scan. Drop Scan locks whatever wrist_roll is at
        # the moment it starts and preserves it through the entire sweep.
        row2 = ttk.Frame(setup)
        row2.pack(fill=tk.X, padx=5, pady=4)
        self._register_button(
            row2, text='Roll +90°  (scan pose)',
            tab='Real Test', section='Setup',
            command=self._cmd_real_wrist_roll_plus,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2), ipady=3)
        self._register_button(
            row2, text='Roll −90°  (drop pose)',
            tab='Real Test', section='Setup',
            command=self._cmd_real_wrist_roll_minus,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0), ipady=3)

        # Row 3: sweep range + direction swap.
        row3 = ttk.Frame(setup)
        row3.pack(fill=tk.X, padx=5, pady=4)
        tk.Label(row3, text='Sweep range (°):', anchor='w').pack(
            side=tk.LEFT, padx=(0, 2))
        self._register_spinbox(
            row3, label='Sweep range (°)',
            tab='Real Test', section='Setup',
            textvariable=self._real_sweep_range_deg_var,
            from_=5, to=180, increment=5, width=5,
        ).pack(side=tk.LEFT, padx=(0, 8))
        self._register_button(
            row3, text='↔  Swap direction',
            tab='Real Test', section='Setup',
            command=self._cmd_real_swap_sweep_direction,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 0), ipady=2)

        # Row 4: Drop Scan on its own row.
        row4 = ttk.Frame(setup)
        row4.pack(fill=tk.X, padx=5, pady=4)
        self._register_button(
            row4, text='▶  Drop Scan  (sweeps configured range, locks current roll)',
            tab='Real Test', section='Setup',
            command=self._cmd_real_drop_scan,
        ).pack(fill=tk.X, expand=True, ipady=3)

        # ===== Section: Calibration =====
        # Two scan-then-cache pipelines, one per detection pass:
        #   - Cups via /drop_poses_real (ArUco), scanned at drop_point_green
        #     pose where wrist camera sees all 3 cup markers in one frame.
        #   - Legos via /objects_poses_real (YOLOE), scanned at grasp_home
        #     pose where wrist camera looks down at the workspace. YOLOE
        #     only sees what's in FOV at scan time — caching freezes that
        #     view so the loop can iterate it even after the arm moves
        #     away from home for grasp/drop motions.
        #
        # Each refresh partial-merges captured detections into its cache
        # (missing markers/legos keep their previous cached value), pushes
        # the merged cache to MoveIt as collision objects, and updates the
        # corresponding live data structure (_drop_data / objects_data) +
        # listbox so existing _cmd_* paths read the cached values.
        cal = ttk.LabelFrame(frame, text='Calibration  (scan-then-cache)')
        cal.pack(fill=tk.X, padx=10, pady=4)

        self._register_button(
            cal,
            text='🔄  Refresh Cups Pose  (subscribe-once /drop_poses_real)',
            tab='Real Test', section='Calibration',
            command=self._cmd_real_refresh_cups_pose,
        ).pack(fill=tk.X, padx=5, pady=(4, 1), ipady=3)

        self._real_cups_status_var = tk.StringVar(value='Cups: (none)')
        tk.Label(
            cal, textvariable=self._real_cups_status_var,
            anchor='w', font=('TkFixedFont', 9), fg='#333',
        ).pack(fill=tk.X, padx=8, pady=(0, 4))

        self._register_button(
            cal,
            text='🔄  Refresh Legos Pose  (subscribe-once /objects_poses_real)',
            tab='Real Test', section='Calibration',
            command=self._cmd_real_refresh_legos_pose,
        ).pack(fill=tk.X, padx=5, pady=(4, 1), ipady=3)

        self._real_legos_status_var = tk.StringVar(value='Legos: (none)')
        tk.Label(
            cal, textvariable=self._real_legos_status_var,
            anchor='w', font=('TkFixedFont', 9), fg='#333',
        ).pack(fill=tk.X, padx=8, pady=(0, 4))

        clear_row = ttk.Frame(cal)
        clear_row.pack(fill=tk.X, padx=5, pady=(2, 5))
        self._register_button(
            clear_row, text='Clear all caches',
            tab='Real Test', section='Calibration',
            command=self._cmd_real_clear_cache,
        ).pack(side=tk.RIGHT, padx=(0, 3))

        # ===== Section: Steps (manual debug) =====
        # Individual step buttons that run ONE _QS_SEQUENCE step each, with
        # the cache-pose re-injection + listbox selection that the full Run
        # path does — but stop after the step. Lets you debug "why does
        # grasp_move fail?" by stepping through grasp_open → grasp_move →
        # close → ... without short-circuiting on the first failure.
        # Each button shares the dropdown's selected color (closest cached
        # lego of that color is picked) and forces real topics on entry.
        steps = ttk.LabelFrame(frame, text='Steps  (manual debug)')
        steps.pack(fill=tk.X, padx=10, pady=4)

        steps_row1 = ttk.Frame(steps)
        steps_row1.pack(fill=tk.X, padx=5, pady=(4, 2))
        for label, cmd in [
            ('Grasp Home',  self._cmd_real_grasp_home),
            ('Grasp Open',  self._cmd_real_grasp_open),
            ('Grasp Move',  self._cmd_real_grasp_move),
            ('Grasp Close', self._cmd_real_grasp_close),
        ]:
            self._register_button(
                steps_row1, text=label, tab='Real Test', section='Steps',
                command=cmd,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, ipady=2)

        steps_row2 = ttk.Frame(steps)
        steps_row2.pack(fill=tk.X, padx=5, pady=(2, 5))
        for label, cmd in [
            ('Drop Point',   self._cmd_real_drop_point),
            ('Drop Sweep',   self._cmd_real_drop_sweep),
            ('Release',      self._cmd_real_drop_release),
        ]:
            self._register_button(
                steps_row2, text=label, tab='Real Test', section='Steps',
                command=cmd,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, ipady=2)

        # ===== Section: Run =====
        # Color dropdown (populated by _real_refresh_color_dropdown from the
        # intersection of cached lego colors AND cached cup colors) + a Run
        # button that picks ONE lego of the selected color. Cycle is full
        # _QS_SEQUENCE; on success the lego is evicted from cache and the
        # dropdown shrinks accordingly.
        run_frame = ttk.LabelFrame(frame, text='Run')
        run_frame.pack(fill=tk.X, padx=10, pady=(4, 8))

        color_row = ttk.Frame(run_frame)
        color_row.pack(fill=tk.X, padx=5, pady=(4, 2))
        tk.Label(color_row, text='Drop color:', anchor='w').pack(
            side=tk.LEFT, padx=(0, 5))
        self._real_color_combobox = self._register_combobox(
            color_row, label='Real Test color',
            tab='Real Test', section='Run',
            textvariable=self._real_selected_color_var,
            values=[], width=10,
        )
        self._real_color_combobox.pack(side=tk.LEFT)

        self._real_run_button = self._register_button(
            run_frame, text='▶  Pick & Drop Selected Color',
            tab='Real Test', section='Run',
            command=self._cmd_real_run_one_color,
        )
        self._real_run_button.pack(fill=tk.X, padx=5, pady=(2, 4), ipady=4)
        # Start disabled — the first Refresh that yields a non-empty
        # intersection re-enables it via _real_refresh_color_dropdown.
        self._real_run_button.config(state='disabled')

        self._real_run_status_var = tk.StringVar(value='Idle')
        status_row = ttk.Frame(run_frame)
        status_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(status_row, text='Status:', anchor='w').pack(side=tk.LEFT)
        tk.Label(
            status_row, textvariable=self._real_run_status_var,
            anchor='w', font=('TkDefaultFont', 9, 'bold'),
        ).pack(side=tk.LEFT, padx=(5, 0))

        # Hint text — pre-flight assumption surface for the new flow.
        hint = (
            'Pre-flight: Drop Point Green → Refresh Cups Pose → Grasp Home '
            '→ Refresh Legos Pose. Then pick a color and press Run. '
            'Successful drops evict that lego from cache automatically.'
        )
        tk.Label(
            frame, text=hint, anchor='w', justify=tk.LEFT,
            fg='#555', font=('TkDefaultFont', 8),
            wraplength=900,
        ).pack(fill=tk.X, padx=12, pady=(8, 4))

        # Self-recovery on hot-reload: instance cache survives a GUI rebuild,
        # but the freshly-constructed combobox is initialized empty + button
        # disabled. Re-run the dropdown helper so the rebuilt widget reflects
        # the live cache state without forcing the user to press Refresh.
        # Wrapped in root.after so it runs after the rest of the tab finishes
        # building (helper accesses self._real_color_combobox).
        self.root.after(0, self._real_refresh_color_dropdown)

    # ------------------------------------------------------------------
    # Real Test tab — command handlers
    # ------------------------------------------------------------------

    def _cmd_real_toggle_wrist_roll_sign(self):
        """Deprecated by Roll +90° / Roll −90° buttons. Kept as stub so the
        auto-registered ~/real_toggle_wrist_roll_sign service doesn't error
        if invoked. Will be removed on next Phase 11 cleanup pass."""
        self._append_log(
            'Real: toggle_wrist_roll_sign deprecated — '
            'use Roll +90° / Roll −90° buttons instead', 'warn')

    # --- Drop Scan workflow (Phase 11-01 followup, replaces Drop Point Green) ---
    # Sweeps shoulder_pan from -90° to +90° in small steps. At each step,
    # samples /drop_poses_real briefly. When a not-yet-cached cup appears in
    # the camera's view zone (target_pan derived from cup XY is within
    # ±FOV_HALF of current sweep pan), the arm visual-servos to point at
    # that cup, then averages the cup pose over AVG_DURATION seconds of
    # nearly-stationary samples. Builds up _cached_cup_poses one cup at a
    # time. Bookended by grasp_home before/after.

    # Drop Scan tunables live at module level (DROP_SCAN_*) so hot-reload
    # picks up changes — see comment at the top of this file about
    # _patch_methods not copying class-level constants.

    def _set_yoloe_overlay(self, enable: bool, timeout_s: float = 0.5) -> bool:
        """Toggle the merged-viewer YOLOE overlay via aruco_camera_localizer
        service. Lazy-creates the client on first call so this works after
        hot-reload (which doesn't re-run __init__). Best-effort — every
        rclpy/tk failure path is swallowed because this is called from the
        `finally` block of `_real_drop_scan_thread` and must never raise
        (otherwise it masks the real exception or stalls cleanup)."""
        try:
            client = getattr(self, '_yoloe_overlay_client', None)
            if client is None:
                client = self.create_client(
                    SetBool, '/aruco_viewer/yoloe_overlay_enable')
                self._yoloe_overlay_client = client
            if not client.wait_for_service(timeout_sec=timeout_s):
                self._append_log(
                    'YOLOE overlay toggle skipped — '
                    '/aruco_viewer/yoloe_overlay_enable not advertised', 'warn')
                return False
            req = SetBool.Request()
            req.data = bool(enable)
            client.call_async(req)
            return True
        except Exception as e:
            try:
                self._append_log(
                    f'YOLOE overlay toggle errored ({type(e).__name__}): {e}',
                    'warn')
            except Exception:
                pass
            return False

    def _cmd_real_drop_scan(self):
        """Replacement for Drop Point Green: sweep -90° → +90° on
        shoulder_pan, detect each cup as it enters view, average its pose
        over 3s, cache, push to scene + dropdown. Bookended by grasp_home.

        Uses the existing Drop Topic flow (/drop_poses_real) as the
        per-cup pose source — visual servoing is "single corrective pan
        per detection," not continuous closed-loop.
        """
        self._ensure_real_state()
        if self._qs_state == 'running':
            self._append_log(
                'Real: drop_scan cannot start — a run is already in progress',
                'warn')
            return
        # Force real topics BEFORE spawning the worker thread. rclpy's
        # destroy_subscription/create_subscription calls can race with
        # MultiThreadedExecutor worker threads if they happen on a
        # daemon thread mid-spin (InvalidHandle crash). The service-
        # callback thread that's calling us here is still an executor
        # worker thread but the idempotent-by-actual-topic logic in
        # _real_ensure_real_topics now makes destroy/create a once-per-
        # process event, so the race window collapses.
        self._real_ensure_real_topics()
        threading.Thread(
            target=self._real_drop_scan_thread, daemon=True).start()

    def _real_drop_scan_thread(self):
        try:
            self._qs_abort_evt.clear()
            self._qs_resume_evt.set()  # release pause gate (default unset)
            self._qs_state = 'running'
            self._real_set_status('Drop Scan', step='preparing')
            self._append_log('Real drop_scan: starting')
            # Suppress YOLOE overlay in the wrist-camera view so the user
            # sees only ArUco drawings during the sweep. Restored in finally.
            self._set_yoloe_overlay(False)

            # --- Phase 0: pre-scan setup (gripper open + canonical scan pose) ---
            # The wrist must be at +90° wrist_roll (camera-down "scan pose")
            # AND the gripper must be fully open before the sweep starts —
            # otherwise we sweep with the previous run's residual orientation
            # / a partially-closed gripper occluding markers. Hardcoded so the
            # user can't accidentally launch a scan with the arm in drop pose.
            SCAN_WRIST_ROLL_RAD = math.pi / 2  # +90° = scan pose
            self._real_set_status('Drop Scan', step='opening gripper')
            self._append_log('Real drop_scan: opening gripper before sweep')
            self._cmd_gripper_open()
            evt = getattr(self, '_motion_event', None)
            if evt is not None:
                evt.wait(timeout=5.0)

            self._real_set_status('Drop Scan', step='flipping wrist to scan pose')
            self._append_log(
                f'Real drop_scan: flipping wrist_roll → '
                f'{math.degrees(SCAN_WRIST_ROLL_RAD):+.1f}° (scan pose)')
            if not self._real_set_wrist_roll(SCAN_WRIST_ROLL_RAD):
                self._append_log(
                    'Real drop_scan: wrist_roll flip failed', 'err')
                self._real_set_status('Halted', step='wrist_roll flip failed')
                return

            # --- Phase 1: scan_home, locking the scan-pose wrist_roll ---
            scan_wrist_roll = SCAN_WRIST_ROLL_RAD
            self._append_log(
                f'Real drop_scan: locking wrist_roll = '
                f'{math.degrees(scan_wrist_roll):+.1f}° for entire sweep')
            self._real_set_status('Drop Scan', step='scan_home')
            if not self._real_scan_home_preserve_roll(scan_wrist_roll):
                self._append_log(
                    'Real drop_scan: scan_home failed', 'err')
                self._real_set_status('Halted', step='scan_home failed')
                return

            from so_arm101_control.compute_workspace import X_PAN
            visited_ids = set()

            # --- Derive sweep parameters from UI vars ---
            # range_deg sets how far from start the sweep goes; reversed_var
            # picks which direction. Default: 90° from +90° start, going
            # right→left (so end = 0°). Clamp range to [5, 180].
            range_deg = max(5, min(180,
                int(self._real_sweep_range_deg_var.get())))
            range_rad = math.radians(range_deg)
            reversed_dir = bool(self._real_sweep_reversed_var.get())
            if reversed_dir:
                sweep_start_rad = DROP_SCAN_PAN_MIN_RAD          # -90°
                sweep_step_signed_rad = +DROP_SCAN_PAN_STEP_RAD  # +5°
                sweep_end_rad = sweep_start_rad + range_rad
            else:
                sweep_start_rad = DROP_SCAN_PAN_MAX_RAD          # +90°
                sweep_step_signed_rad = -DROP_SCAN_PAN_STEP_RAD  # -5°
                sweep_end_rad = sweep_start_rad - range_rad
            # Clamp end to absolute joint bounds (range_deg can drive it past).
            sweep_end_rad = max(DROP_SCAN_PAN_MIN_RAD,
                                min(DROP_SCAN_PAN_MAX_RAD, sweep_end_rad))
            self._append_log(
                f'Real drop_scan: range={range_deg}° '
                f'direction={"left→right" if reversed_dir else "right→left"} '
                f'start={math.degrees(sweep_start_rad):+.1f}° '
                f'end={math.degrees(sweep_end_rad):+.1f}°')

            # --- Phase 2a: move to sweep start with safe duration ---
            # The initial move from home (pan=0) to start is up to 90° of
            # travel — send with a generous duration so we don't exceed the
            # drive's tracking envelope (~30°/s sustained per CLAUDE.md).
            # Subsequent 5° increments use the short duration safely.
            self._real_set_status(
                'Drop Scan',
                step=f'moving to sweep start ({math.degrees(sweep_start_rad):+.0f}°)')
            self._append_log(
                f'Real drop_scan: moving to start pan='
                f'{math.degrees(sweep_start_rad):+.1f}°')
            if not self._real_drop_scan_motion(
                    self._real_drop_scan_pan_target(
                        sweep_start_rad, scan_wrist_roll),
                    3.5):
                self._append_log(
                    'Real drop_scan: initial move to start failed', 'err')
                self._real_set_status('Halted', step='initial pan failed')
                return

            # --- Phase 2b: sweep start → end in step_signed increments ---
            # Loop continues while signed remaining distance is non-negative
            # (works in both directions since step_signed carries the sign).
            pan = sweep_start_rad
            first_iter = True
            sweep_sign = 1.0 if sweep_step_signed_rad > 0 else -1.0
            while sweep_sign * (sweep_end_rad - pan) >= -1e-6:
                if self._qs_abort_evt.is_set():
                    self._append_log('Real drop_scan: aborted', 'warn')
                    self._real_set_status('Aborted')
                    return
                self._real_set_status(
                    'Drop Scan',
                    step=f'sweep pan={math.degrees(pan):+.1f}° '
                         f'cached={len(visited_ids)}/3')
                # Skip motion on first iter — we're already at pan_min from
                # the initial move above. Subsequent iters do the small step.
                if not first_iter:
                    target = self._real_drop_scan_pan_target(
                        pan, scan_wrist_roll)
                    if not self._real_drop_scan_motion(
                            target, DROP_SCAN_STEP_DURATION_S):
                        self._append_log(
                            f'Real drop_scan: pan motion to '
                            f'{math.degrees(pan):+.1f}° failed', 'err')
                        self._real_set_status('Halted', step='pan motion failed')
                        return
                first_iter = False
                time.sleep(DROP_SCAN_SETTLE_AFTER_STEP_S)

                # Sample briefly — looking for new cups in view zone
                samples = self._real_sample_drop_poses(
                    DROP_SCAN_SAMPLE_PER_STEP_S)
                for drop_id, pose_list in samples.items():
                    if drop_id in visited_ids:
                        continue
                    if not pose_list:
                        continue
                    # Compute target pan from latest sample
                    p = pose_list[-1]
                    target_pan = math.atan2(-p['y'], p['x'] - X_PAN)
                    # In view zone? (within ±FOV_HALF of current pan)
                    if abs(target_pan - pan) > DROP_SCAN_FOV_HALF_RAD:
                        continue
                    # Out of pan range? skip
                    if (target_pan < DROP_SCAN_PAN_MIN_RAD or
                            target_pan > DROP_SCAN_PAN_MAX_RAD):
                        continue
                    self._append_log(
                        f'Real drop_scan: detected {drop_id} → target_pan='
                        f'{math.degrees(target_pan):+.1f}° (sweep at '
                        f'{math.degrees(pan):+.1f}°)')
                    self._real_set_status(
                        'Drop Scan',
                        step=f'settling on {drop_id}')

                    # Visual servo: single corrective pan to target. Once we
                    # commit to servoing on a drop_id we never re-target it,
                    # success or failure — otherwise a flaky cup keeps
                    # pulling the sweep back and the scan never advances.
                    servo_target = self._real_drop_scan_pan_target(
                        target_pan, scan_wrist_roll)
                    if not self._real_drop_scan_motion(
                            servo_target, DROP_SCAN_STEP_DURATION_S + 0.1):
                        self._append_log(
                            f'Real drop_scan: servo to {drop_id} failed — '
                            'marking visited, sweep continues', 'warn')
                        visited_ids.add(drop_id)
                        continue
                    time.sleep(0.3)  # let arm settle before averaging

                    # 3-sec average
                    avg_samples = self._real_sample_drop_poses(
                        DROP_SCAN_AVG_DURATION_S, filter_id=drop_id)
                    drop_samples = avg_samples.get(drop_id, [])
                    if len(drop_samples) < DROP_SCAN_MIN_AVG_SAMPLES:
                        self._append_log(
                            f'Real drop_scan: {drop_id} settle failed — only '
                            f'{len(drop_samples)} samples '
                            f'(need ≥{DROP_SCAN_MIN_AVG_SAMPLES}) — '
                            'marking visited, sweep continues', 'warn')
                        visited_ids.add(drop_id)
                        # Resume sweep from servoed pan
                        pan = target_pan
                        continue
                    avg_pose, xy_var = self._real_average_pose_samples(
                        drop_samples)
                    if xy_var > DROP_SCAN_MAX_XY_VARIANCE_M2:
                        self._append_log(
                            f'Real drop_scan: {drop_id} xy variance '
                            f'{xy_var*1e6:.1f} mm² too high — marking '
                            'visited, sweep continues', 'warn')
                        visited_ids.add(drop_id)
                        pan = target_pan
                        continue
                    self._cached_cup_poses[drop_id] = avg_pose
                    visited_ids.add(drop_id)
                    self._append_log(
                        f'Real drop_scan: cached {drop_id} '
                        f'(avg {len(drop_samples)} samples, '
                        f'xy_std={1000*math.sqrt(xy_var):.2f} mm)')
                    pan = target_pan  # resume from servoed pos

                pan += DROP_SCAN_PAN_STEP_SIGNED_RAD

            # --- Phase 3: push cached to MoveIt + visual + dropdown ---
            self._real_set_status('Drop Scan', step='applying scene')
            self._remove_cup_collision_objects()
            time.sleep(0.5)  # planning-scene monitor round-trip
            if self._cached_cup_poses:
                self._add_cup_collision_objects(
                    cups_dict=dict(self._cached_cup_poses))
                self._publish_cup_visual_markers(
                    cups_dict=dict(self._cached_cup_poses))
            # Mirror into _drop_data + listbox so existing flows pick up.
            with self._drop_lock:
                self._drop_data.update(self._cached_cup_poses)
            self.root.after(0, self._populate_drop_list)
            # Status label per cup
            def _update_label():
                marks = []
                for n in ('drop_1', 'drop_2', 'drop_3'):
                    glyph = '✓' if n in self._cached_cup_poses else '✗'
                    marks.append(f'{n}:{glyph}')
                if hasattr(self, '_real_cups_status_var'):
                    self._real_cups_status_var.set(
                        'Cups: ' + '   '.join(marks))
            self.root.after(0, _update_label)
            self._real_refresh_color_dropdown()
            # Report on the AUTHORITATIVE cache (visited_ids now includes
            # cups that were attempted but rejected for high variance / low
            # sample count, so it's not a "cached" count anymore).
            cached_names = sorted(self._cached_cup_poses.keys())
            attempted_only = sorted(visited_ids - set(cached_names))
            summary = (
                f'Real drop_scan: cached {len(cached_names)} cup(s): '
                f'{cached_names}')
            if attempted_only:
                summary += f' (attempted but rejected: {attempted_only})'
            self._append_log(summary)

            # --- Phase 4: return to scan_home (preserve scan-start wrist_roll) ---
            self._real_set_status('Drop Scan', step='scan_home (return)')
            if not self._real_scan_home_preserve_roll(scan_wrist_roll):
                self._append_log('Real drop_scan: return scan_home failed',
                                 'warn')
            self._real_set_status(
                'Complete',
                step=f'✓ {len(self._cached_cup_poses)} cup(s) cached')
        finally:
            self._set_yoloe_overlay(True)
            self._qs_state = 'idle'

    # --- Drop scan helpers ---

    def _real_drop_scan_pan_target(self, pan_rad, wrist_roll_rad):
        """Build a joint target with shoulder_pan = pan_rad and wrist_roll
        LOCKED at `wrist_roll_rad` (the captured-at-scan-start value, NOT
        the live current value). Other arm joints fixed at scan-home values
        (shoulder_lift=0, elbow_flex=0, wrist_flex=π/2).

        Locking wrist_roll to a captured constant — rather than reading
        current /joint_states each step — prevents drift from compounding
        across the 36-step sweep. The controller must keep correcting back
        to the captured value rather than capturing whatever drift occurred."""
        target = {n: 0.0 for n in ARM_JOINT_NAMES}
        target['wrist_flex'] = math.pi / 2
        target['wrist_roll'] = float(wrist_roll_rad)
        target['shoulder_pan'] = float(pan_rad)
        return target

    def _real_scan_home_preserve_roll(self, wrist_roll_rad=None):
        """Move arm to a scan-ready pose (pan=0, lift=0, elbow=0, wrist_flex=π/2)
        but PRESERVE wrist_roll at `wrist_roll_rad` (defaults to current value
        from /joint_states). Used by Drop Scan so whatever wrist_roll the user
        set via the Roll +90° / Roll −90° buttons is honored throughout the
        sweep — drop_scan no longer flips the wrist itself.

        Safe duration computed from max joint delta (~30°/s sustained on the
        SO-ARM101 drives, min 1.5s).
        """
        with self.joint_lock:
            current = dict(self.joint_positions)
        if wrist_roll_rad is None:
            wrist_roll_rad = current['wrist_roll']
        target = {n: 0.0 for n in ARM_JOINT_NAMES}
        target['wrist_flex'] = math.pi / 2
        target['wrist_roll'] = float(wrist_roll_rad)
        max_delta = max(abs(target[n] - current[n]) for n in ARM_JOINT_NAMES)
        SAFE_RATE_RAD_PER_S = math.radians(30)
        duration = max(1.5, max_delta / SAFE_RATE_RAD_PER_S + 0.5)
        self._append_log(
            f'Real: scan_home (preserve wrist_roll='
            f'{math.degrees(wrist_roll_rad):+.1f}°, '
            f'max_Δ={math.degrees(max_delta):.1f}°, dur={duration:.1f}s)')
        return bool(self._send_arm_goal(
            target, duration_s=duration, blocking=True))

    def _real_set_wrist_roll(self, target_rad):
        """Move ONLY wrist_roll to target_rad; all other joints stay at
        current /joint_states values. Safe duration based on the wrist_roll
        delta (other joints have ~0 delta so they don't bound it).
        """
        self._ensure_real_state()
        with self.joint_lock:
            current = dict(self.joint_positions)
        target = dict(current)
        target['wrist_roll'] = float(target_rad)
        delta = abs(target_rad - current['wrist_roll'])
        SAFE_RATE_RAD_PER_S = math.radians(30)
        duration = max(1.5, delta / SAFE_RATE_RAD_PER_S + 0.5)
        self._append_log(
            f'Real: wrist_roll → {math.degrees(target_rad):+.1f}° '
            f'(Δ={math.degrees(delta):.1f}°, dur={duration:.1f}s)')
        return bool(self._send_arm_goal(
            target, duration_s=duration, blocking=True))

    def _cmd_real_swap_sweep_direction(self):
        """Toggle the Drop Scan sweep direction. False (default) = canonical
        right→left starting at +90°. True = swap, left→right starting at -90°.
        Read by `_real_drop_scan_thread` at scan start."""
        self._ensure_real_state()
        new_val = not self._real_sweep_reversed_var.get()
        self._real_sweep_reversed_var.set(new_val)
        direction = 'left→right (start -90°)' if new_val else 'right→left (start +90°)'
        self._append_log(f'Real: sweep direction → {direction}')

    def _cmd_real_wrist_roll_plus(self):
        """Move wrist_roll to +90° (markers-in-FOV / scan pose)."""
        threading.Thread(
            target=self._real_set_wrist_roll, args=(math.pi / 2,),
            daemon=True).start()

    def _cmd_real_wrist_roll_minus(self):
        """Move wrist_roll to −90° (jaw-clearance / drop pose)."""
        threading.Thread(
            target=self._real_set_wrist_roll, args=(-math.pi / 2,),
            daemon=True).start()

    def _real_drop_scan_motion(self, target, duration_s):
        """Send arm goal blocking. Returns True on success."""
        self._append_log(
            f'  drop_scan_motion: dispatching pan='
            f'{math.degrees(target["shoulder_pan"]):+.1f}° dur={duration_s:.1f}s')
        ok = bool(self._send_arm_goal(
            target, duration_s=duration_s, blocking=True))
        self._append_log(
            f'  drop_scan_motion: returned ok={ok}')
        return ok

    def _real_sample_drop_poses(self, duration_s, filter_id=None):
        """Subscribe to /drop_poses_real for `duration_s`, collect all
        TF transforms by child_frame_id. Returns {drop_id: [pose_dict, ...]}.
        Each pose_dict has 'x','y','z','qx','qy','qz','qw'.
        If filter_id is given, only collect that drop_id (saves work)."""
        samples = {}
        def _cb(msg):
            for tf in msg.transforms:
                fid = tf.child_frame_id
                if filter_id is not None and fid != filter_id:
                    continue
                samples.setdefault(fid, []).append({
                    'x': tf.transform.translation.x,
                    'y': tf.transform.translation.y,
                    'z': tf.transform.translation.z,
                    'qx': tf.transform.rotation.x,
                    'qy': tf.transform.rotation.y,
                    'qz': tf.transform.rotation.z,
                    'qw': tf.transform.rotation.w,
                })
        sub = self.create_subscription(
            TFMessage, '/drop_poses_real', _cb, 50,
            callback_group=self._sub_cb_group)
        try:
            time.sleep(duration_s)
        finally:
            self.destroy_subscription(sub)
        return samples

    def _real_average_pose_samples(self, samples):
        """Component-wise mean of xyz + normalized-mean quaternion. Returns
        (avg_pose_dict, xy_variance_m2). Quaternion sign-flips canonicalized
        before averaging to handle the q=-q ambiguity (otherwise opposite
        signs cancel)."""
        n = len(samples)
        sx = sum(s['x'] for s in samples) / n
        sy = sum(s['y'] for s in samples) / n
        sz = sum(s['z'] for s in samples) / n
        # xy variance (squared 2D distance from mean)
        xy_var = sum((s['x']-sx)**2 + (s['y']-sy)**2 for s in samples) / n
        # Quat mean: canonicalize sign vs first sample to avoid ±q cancellation
        ref = samples[0]
        qx = qy = qz = qw = 0.0
        for s in samples:
            sign = 1.0 if (s['qw']*ref['qw'] + s['qx']*ref['qx'] +
                           s['qy']*ref['qy'] + s['qz']*ref['qz']) >= 0 else -1.0
            qx += sign * s['qx']
            qy += sign * s['qy']
            qz += sign * s['qz']
            qw += sign * s['qw']
        norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw) or 1.0
        return ({
            'x': sx, 'y': sy, 'z': sz,
            'qx': qx/norm, 'qy': qy/norm, 'qz': qz/norm, 'qw': qw/norm,
        }, xy_var)

    def _cmd_real_drop_point_green(self):
        """Deprecated by Drop Scan (Phase 11-01 followup). Kept as a
        deprecation stub so the auto-registered ~/real_drop_point_green
        service doesn't error if invoked during the transition. Will be
        deleted on next Phase 11 cleanup pass."""
        self._append_log(
            'Real: drop_point_green is deprecated — use Drop Scan instead',
            'warn')

    def _cmd_real_refresh_cups_pose(self):
        """Subscribe-once to /drop_poses_real, partial-merge captured
        transforms into _cached_cup_poses (missing markers keep their
        previous cached value), push the merged set to MoveIt as collision
        objects, and update _drop_data + _drop_listbox so the Grasp tab's
        drop selection mirrors the cache.
        """
        self._ensure_real_state()
        # Force all relevant topics to /*_real BEFORE the refresh so live
        # callbacks during/after the refresh write real-source data into
        # _drop_data / objects_data / objects_bbox. Without this, sim subs
        # continuously overwrite _drop_data and any subsequent code path
        # that publishes visual markers without an explicit cups_dict
        # (e.g. _cmd_apply_collision_padding → _refresh_display_markers)
        # would re-publish sim cup positions.
        self._real_ensure_real_topics()
        threading.Thread(
            target=self._real_refresh_cups_thread, daemon=True).start()

    def _real_refresh_cups_thread(self):
        evt = threading.Event()
        captured = {}

        def _once_cb(msg):
            if evt.is_set():
                return
            for tf in msg.transforms:
                captured[tf.child_frame_id] = {
                    'x': tf.transform.translation.x,
                    'y': tf.transform.translation.y,
                    'z': tf.transform.translation.z,
                    'qx': tf.transform.rotation.x,
                    'qy': tf.transform.rotation.y,
                    'qz': tf.transform.rotation.z,
                    'qw': tf.transform.rotation.w,
                }
            evt.set()

        topic = '/drop_poses_real'
        self._append_log(f'Real: subscribing once to {topic} (3s timeout)')
        sub = self.create_subscription(
            TFMessage, topic, _once_cb, 10,
            callback_group=self._sub_cb_group)
        try:
            ok = evt.wait(timeout=3.0)
            if not ok:
                self._append_log(
                    f'Real: no message on {topic} within 3s — is '
                    'aruco_camera_localizer running and seeing markers?',
                    'warn')
                return
            # Partial-cache merge: only update what was captured this scan.
            self._cached_cup_poses.update(captured)
            captured_names = sorted(captured.keys())
            kept_previous = [
                n for n in sorted(self._cached_cup_poses.keys())
                if n not in captured]
            self._append_log(
                f'Real: refreshed cups: captured={captured_names}, '
                f'kept_previous={kept_previous}, '
                f'total_cached={len(self._cached_cup_poses)}')
            # Mirror into _drop_data + listbox so existing drop_point/sweep/
            # release flows (which read from _drop_data) pick up the cache.
            with self._drop_lock:
                self._drop_data.update(self._cached_cup_poses)
            self.root.after(0, self._populate_drop_list)
            # WIPE-then-PUSH mirrors _cmd_drop_refresh's working pattern —
            # MoveIt's ApplyPlanningScene ADD-on-existing-id is not cleanly
            # idempotent, so repeat refreshes silently fail without prior
            # REMOVE. 500 ms gap = planning-scene monitor round-trip.
            self._remove_cup_collision_objects()
            time.sleep(0.5)
            self._add_cup_collision_objects(
                cups_dict=dict(self._cached_cup_poses))
            # Visual markers must use the SAME cached snapshot as the
            # collision — without the explicit cups_dict, the publisher
            # would read self._drop_data which the (sim) /drop_poses sub
            # continuously overwrites, causing visual+collision divergence.
            self._publish_cup_visual_markers(
                cups_dict=dict(self._cached_cup_poses))
            self._append_log(
                f'Real: published cup visual markers from cache '
                f'({len(self._cached_cup_poses)} cups)')
            # Status label: ✓ for cached, ✗ for missing.
            def _update_label():
                marks = []
                for n in ('drop_1', 'drop_2', 'drop_3'):
                    glyph = '✓' if n in self._cached_cup_poses else '✗'
                    marks.append(f'{n}:{glyph}')
                self._real_cups_status_var.set(
                    'Cups: ' + '   '.join(marks))
            self.root.after(0, _update_label)
            self._real_refresh_color_dropdown()
        finally:
            self.destroy_subscription(sub)

    def _cmd_real_refresh_legos_pose(self):
        """Subscribe-once to /objects_poses_real, partial-merge captured
        lego TFs into _cached_lego_poses (missing legos keep their previous
        cached value), push the merged set to MoveIt as collision objects,
        and update objects_data + obj_listbox so the Grasp tab's lego
        selection mirrors the cache.

        Run this AT GRASP_HOME pose — that's the only pose where the wrist
        camera looks down at the workspace. YOLOE only sees what's in FOV
        at scan time, which is why the cache is essential for the run loop:
        once the arm leaves home for a grasp, the live /objects_poses_real
        feed only contains whatever's near tcp_link.
        """
        self._ensure_real_state()
        # Same reasoning as Refresh Cups Pose — force topics before the
        # refresh so live callbacks during the cycle write real-source data.
        self._real_ensure_real_topics()
        threading.Thread(
            target=self._real_refresh_legos_thread, daemon=True).start()

    def _real_refresh_legos_thread(self):
        evt = threading.Event()
        captured = {}

        def _once_cb(msg):
            if evt.is_set():
                return
            for tf in msg.transforms:
                captured[tf.child_frame_id] = {
                    'x': tf.transform.translation.x,
                    'y': tf.transform.translation.y,
                    'z': tf.transform.translation.z,
                    'qx': tf.transform.rotation.x,
                    'qy': tf.transform.rotation.y,
                    'qz': tf.transform.rotation.z,
                    'qw': tf.transform.rotation.w,
                }
            evt.set()

        topic = '/objects_poses_real'
        self._append_log(f'Real: subscribing once to {topic} (3s timeout)')
        sub = self.create_subscription(
            TFMessage, topic, _once_cb, 10,
            callback_group=self._sub_cb_group)
        try:
            ok = evt.wait(timeout=3.0)
            if not ok:
                self._append_log(
                    f'Real: no message on {topic} within 3s — is '
                    'localize_yoloe running and seeing legos in FOV?',
                    'warn')
                return
            # Partial-cache merge: only update what was captured this scan.
            self._cached_lego_poses.update(captured)
            captured_names = sorted(captured.keys())
            kept_previous = [
                n for n in sorted(self._cached_lego_poses.keys())
                if n not in captured]
            self._append_log(
                f'Real: refreshed legos: captured={captured_names}, '
                f'kept_previous={kept_previous}, '
                f'total_cached={len(self._cached_lego_poses)}')
            # Mirror into objects_data + listbox so existing grasp_move /
            # gripper_close_for_object flows (which read objects_data)
            # pick up the cache. Live /objects_poses_real callbacks may
            # overwrite during the run — re-injection right before the
            # cycle in _real_run_one_color_thread guards against that.
            with self.objects_lock:
                self.objects_data.update(self._cached_lego_poses)
            self.root.after(0, self._populate_object_list)
            self.root.after(0, self._qs_refresh_objects)
            # WIPE-then-PUSH mirrors _cmd_grasp_refresh's working pattern.
            # MoveIt's ApplyPlanningScene ADD on an existing id is not
            # cleanly idempotent — repeat refreshes silently fail unless
            # the prior collision objects are removed first. The 500 ms
            # gap is the round-trip the planning-scene monitor needs to
            # see the REMOVE before the ADD lands against a clean state.
            self._remove_lego_collision_objects()
            time.sleep(0.5)
            self._add_lego_collision_objects(
                legos_dict=dict(self._cached_lego_poses))
            self.root.after(0, self._real_update_legos_status_label)
            self._real_refresh_color_dropdown()
        finally:
            self.destroy_subscription(sub)

    def _real_update_legos_status_label(self):
        """Refresh the 'Legos: N cached  (red:R blue:B green:G)' label.
        Extracted so the post-drop eviction path can reuse it without
        duplicating the tally logic."""
        tally = {'red': 0, 'blue': 0, 'green': 0}
        for name in self._cached_lego_poses:
            color = name.split('_', 1)[0].lower()
            if color in tally:
                tally[color] += 1
        total = len(self._cached_lego_poses)
        breakdown = '  '.join(f'{c}:{n}' for c, n in tally.items())
        if hasattr(self, '_real_legos_status_var'):
            self._real_legos_status_var.set(
                f'Legos: {total} cached  ({breakdown})')

    def _cmd_real_clear_cache(self):
        """Clear every Real Test tab cache and reset to defaults.

        Inverse of both Refresh Cups Pose, Refresh Legos Pose, and the
        wrist-roll toggle. Walks all state mutated by this tab's handlers
        so no ghost state remains.

        Cleared:
          - _cached_cup_poses + _cached_lego_poses (the scan caches)
          - _drop_data + objects_data entries that were mirrored from the
            caches (selective by name — leaves entries from live topic
            subscriptions alone, since those refill on the next message)
          - drop_listbox + obj_listbox (re-populated from trimmed data)
          - cup + lego collision objects in MoveIt planning scene
          - cup visual markers in RViz (idempotent DELETE)
          - Status labels (Cups + Legos + Run)

        Not cleared:
          - QS state (handled separately by Quickstart Restart)
        """
        self._ensure_real_state()
        cached_cup_names = list(self._cached_cup_poses.keys())
        cached_lego_names = list(self._cached_lego_poses.keys())

        self._cached_cup_poses.clear()
        self._cached_lego_poses.clear()

        # Selectively pop the cached names from _drop_data / objects_data
        # so the listboxes don't keep showing cups/legos we no longer trust.
        # Live subscriptions will refill these on the next msg.
        if cached_cup_names:
            with self._drop_lock:
                for name in cached_cup_names:
                    self._drop_data.pop(name, None)
            self.root.after(0, self._populate_drop_list)
        if cached_lego_names:
            with self.objects_lock:
                for name in cached_lego_names:
                    self.objects_data.pop(name, None)
            self.root.after(0, self._populate_object_list)
            self.root.after(0, self._qs_refresh_objects)

        # Wipe the FULL sim-side state too — objects_data + _drop_data
        # carry residual entries from any live subscription that fired
        # since the last refresh. The user explicitly asked for clear
        # cache to clear EVERYTHING (sim + real) so subsequent refreshes
        # start from a known-empty state instead of inheriting half-stale
        # poses that mix sim and real frames.
        with self.objects_lock:
            sim_lego_count = len(self.objects_data)
            self.objects_data.clear()
        with self._drop_lock:
            sim_cup_count = len(self._drop_data)
            self._drop_data.clear()
        self.root.after(0, self._populate_object_list)
        self.root.after(0, self._populate_drop_list)
        self.root.after(0, self._qs_refresh_objects)

        self._remove_cup_collision_objects()
        self._remove_lego_collision_objects()
        # Visual markers (RViz colored cups). Idempotent — publishes DELETE
        # regardless of source (this tab's Refresh, sim's _cmd_drop_refresh,
        # or never added). Doesn't depend on _show_visual_var; clearing the
        # cache always clears the visual representation.
        self._delete_visual_markers()

        # Reset subs back to sim defaults — without this, the next refresh
        # button press in the Grasp/QS tab would still pull from /*_real
        # because subs are sticky after a Real Test session.
        self._sim_ensure_sim_topics()

        if hasattr(self, '_real_cups_status_var'):
            self._real_cups_status_var.set('Cups: (none)')
        if hasattr(self, '_real_legos_status_var'):
            self._real_legos_status_var.set('Legos: (none)')
        if hasattr(self, '_real_run_status_var'):
            self._real_run_status_var.set('Idle')

        self._append_log(
            f'Cleared ALL caches: real({len(cached_cup_names)} cups, '
            f'{len(cached_lego_names)} legos) + '
            f'sim({sim_cup_count} cups, {sim_lego_count} legos), '
            f'collisions removed, subs reset to sim, wrist_roll sign reset')
        self._real_refresh_color_dropdown()

    # --- Color-driven single-pick helper (Phase 11-01) ---
    # Color → cup mapping lives at module level (REAL_COLOR_TO_CUP) so
    # hot-reload picks up changes — class-level constants don't survive
    # _patch_methods.

    def _real_ensure_real_topics(self):
        """Force Grasp Topic + BBox Topic + Drop Topic to /*_real equivalents
        and re-subscribe so sim callbacks can't leak data into objects_data /
        objects_bbox / _drop_data during a Real Test cycle (D-07).

        Idempotent based on the SUBSCRIPTION's own `topic_name`, NOT the
        StringVar widget value. The widget can disagree with the live
        subscription (the original D-07 finding — _build_grasp_tab inits
        the sub on the sim default before _restore_gui_state flips the
        StringVar to real, no auto-re-sub). Reading sub.topic_name
        bypasses that. Re-subscribing every call is the previous
        implementation but it races destroy_subscription/create_subscription
        against MultiThreadedExecutor worker threads currently servicing
        the same sub — produces `InvalidHandle: destruction was requested`
        crashes mid-spin. The idempotent path makes the destroy/create
        race a once-per-process event instead of once-per-scan.

        Must be called from the tk thread (touches StringVars + buttons).

        Why this is essential:
        - `_objects_callback` (sim sub) overwrites `objects_data` AND auto-
          schedules `_add_lego_collision_objects` on >3 mm moves → sim
          poses leak into both objects_data AND the MoveIt planning scene.
        - `_bbox_callback` (sim sub) leaves `_lookup_bbox` returning sim
          catalog values instead of YOLOE's real catalog.
        - `_drop_callback` (sim sub) overwrites `_drop_data`, which
          `_publish_cup_visual_markers` reads → visual cups in RViz show
          sim positions instead of real-cached positions, causing
          visual-vs-collision divergence in the user's view.
        """
        REAL_OBJECTS = '/objects_poses_real'
        REAL_BBOX = '/objects_bbox_real'
        REAL_DROP = '/drop_poses_real'

        def _sub_topic(sub_attr):
            sub = getattr(self, sub_attr, None)
            if sub is None:
                return None
            try:
                return sub.topic_name.lstrip('/').rstrip('/')
            except Exception:
                return None

        def _norm(topic):
            return topic.lstrip('/').rstrip('/')

        # --- Grasp Topic: re-subscribe only if sub topic isn't already real ---
        if _sub_topic('objects_sub') != _norm(REAL_OBJECTS):
            prev_obj = self._grasp_topic_var.get().strip()
            self._grasp_topic_var.set(REAL_OBJECTS)
            self._cmd_grasp_update_topic()
            self._append_log(
                f'Real: forced grasp_topic {prev_obj!r} → {REAL_OBJECTS!r} '
                '(re-subscribed)')
        else:
            # Sub already on real — just sync the widget StringVar so the UI
            # doesn't lie about which topic is live.
            if self._grasp_topic_var.get().strip() != REAL_OBJECTS:
                self._grasp_topic_var.set(REAL_OBJECTS)

        # --- BBox Topic: re-subscribe only if sub topic isn't already real ---
        if _sub_topic('bbox_sub') != _norm(REAL_BBOX):
            prev_bbox = self._bbox_topic_var.get().strip()
            self._bbox_topic_var.set(REAL_BBOX)
            if hasattr(self, 'bbox_sub') and self.bbox_sub is not None:
                self.destroy_subscription(self.bbox_sub)
            self.bbox_sub = self.create_subscription(
                String, REAL_BBOX, self._bbox_callback, 1)
            # Clear stale sim bbox; next /objects_bbox_real msg refills it
            # within ~1 s (YOLOE publishes continuously).
            self.objects_bbox = {}
            self._append_log(
                f'Real: forced bbox_topic {prev_bbox!r} → {REAL_BBOX!r} '
                '(re-subscribed)')
        else:
            if self._bbox_topic_var.get().strip() != REAL_BBOX:
                self._bbox_topic_var.set(REAL_BBOX)

        # --- Drop Topic: re-subscribe only if sub topic isn't already real ---
        if _sub_topic('_drop_sub') != _norm(REAL_DROP):
            prev_drop = self._drop_topic_var.get().strip()
            self._drop_topic_var.set(REAL_DROP)
            # _update_drop_topic clears _drop_data + removes cup collisions —
            # those will be re-added by the next Refresh Cups Pose. The
            # call ALSO publishes "Drop topic: ..." log line for visibility.
            self._update_drop_topic(REAL_DROP)
            self._append_log(
                f'Real: forced drop_topic {prev_drop!r} → {REAL_DROP!r} '
                '(re-subscribed)')
        else:
            if self._drop_topic_var.get().strip() != REAL_DROP:
                self._drop_topic_var.set(REAL_DROP)

    def _sim_ensure_sim_topics(self):
        """Mirror of `_real_ensure_real_topics` for the SIM side.

        Force `objects_sub`, `bbox_sub`, and `_drop_sub` to the sim defaults
        (`/objects_poses_sim`, `/objects_bbox_sim`, `/drop_poses`). Idempotent
        on each sub's `topic_name` so the destroy/create race is at most a
        once-per-process event.

        Called from `_cmd_grasp_refresh` and `_cmd_drop_refresh` so that
        sim refresh buttons reliably switch back to sim topics even after a
        prior `_real_ensure_real_topics` swap. Without this, sim refresh
        leaks: it clears objects_data + cup_collision and waits for the
        live topic to refill — but the topic is still on `/*_real`, so
        sim collision objects come from real-mode poses. Closes the
        D-08 sim/real state-leak loop the user described.
        """
        SIM_OBJECTS = '/objects_poses_sim'
        SIM_BBOX = '/objects_bbox_sim'
        SIM_DROP = '/drop_poses'

        def _sub_topic(sub_attr):
            sub = getattr(self, sub_attr, None)
            if sub is None:
                return None
            try:
                return sub.topic_name.lstrip('/').rstrip('/')
            except Exception:
                return None

        def _norm(topic):
            return topic.lstrip('/').rstrip('/')

        # Grasp Topic → /objects_poses_sim
        if _sub_topic('objects_sub') != _norm(SIM_OBJECTS):
            prev = self._grasp_topic_var.get().strip() \
                if hasattr(self, '_grasp_topic_var') else ''
            if hasattr(self, '_grasp_topic_var'):
                self._grasp_topic_var.set(SIM_OBJECTS)
            self._cmd_grasp_update_topic()
            self._append_log(
                f'Sim: forced grasp_topic {prev!r} → {SIM_OBJECTS!r} '
                '(re-subscribed)')
        elif hasattr(self, '_grasp_topic_var') \
                and self._grasp_topic_var.get().strip() != SIM_OBJECTS:
            self._grasp_topic_var.set(SIM_OBJECTS)

        # BBox Topic → /objects_bbox_sim
        if _sub_topic('bbox_sub') != _norm(SIM_BBOX):
            prev = self._bbox_topic_var.get().strip() \
                if hasattr(self, '_bbox_topic_var') else ''
            if hasattr(self, '_bbox_topic_var'):
                self._bbox_topic_var.set(SIM_BBOX)
            if hasattr(self, 'bbox_sub') and self.bbox_sub is not None:
                self.destroy_subscription(self.bbox_sub)
            self.bbox_sub = self.create_subscription(
                String, SIM_BBOX, self._bbox_callback, 1)
            self.objects_bbox = {}
            self._append_log(
                f'Sim: forced bbox_topic {prev!r} → {SIM_BBOX!r} '
                '(re-subscribed)')
        elif hasattr(self, '_bbox_topic_var') \
                and self._bbox_topic_var.get().strip() != SIM_BBOX:
            self._bbox_topic_var.set(SIM_BBOX)

        # Drop Topic → /drop_poses
        if _sub_topic('_drop_sub') != _norm(SIM_DROP):
            prev = self._drop_topic_var.get().strip() \
                if hasattr(self, '_drop_topic_var') else ''
            if hasattr(self, '_drop_topic_var'):
                self._drop_topic_var.set(SIM_DROP)
            self._update_drop_topic(SIM_DROP)
            self._append_log(
                f'Sim: forced drop_topic {prev!r} → {SIM_DROP!r} '
                '(re-subscribed)')
        elif hasattr(self, '_drop_topic_var') \
                and self._drop_topic_var.get().strip() != SIM_DROP:
            self._drop_topic_var.set(SIM_DROP)

    def _real_inject_active_pair(self, lego_name, cup_name):
        """Re-inject the active lego + cup pose into objects_data and
        _drop_data from the cache. Called BEFORE EACH _QS_SEQUENCE step
        so a late real-topic callback (or any other writer) can't drift
        the pose mid-cycle. Cheap (~µs) — idempotent .update() per call.
        """
        with self._drop_lock:
            self._drop_data[cup_name] = dict(
                self._cached_cup_poses[cup_name])
        with self.objects_lock:
            self.objects_data[lego_name] = dict(
                self._cached_lego_poses[lego_name])

    # --- Per-step debug buttons (Phase 11-01 followup) ---
    # Each _cmd_real_<step> button runs ONE _QS_SEQUENCE step in real-mode
    # so the user can debug step-by-step without losing the cache-pose
    # injection guarantees. Auto-registered as ~/real_<step> services via
    # the _cmd_* convention.

    def _real_run_step(self, label, method_name, kwargs=None):
        """Common dispatcher for individual Real Test step buttons.

        Pre-flight: dropdown color is selected, at least one cached lego
        of that color exists, and (for cup steps) the matching cup is
        cached. Forces real topics, picks closest cached lego of color,
        re-injects pose for active lego+cup, dispatches via
        `_qs_execute_step` which handles obj_listbox / drop_listbox sync
        per the existing per-step policies.

        Runs the actual motion in a background thread so the GUI doesn't
        freeze. Logs ✓/✗ outcome at completion.
        """
        self._ensure_real_state()
        color = self._real_selected_color_var.get()
        if not color:
            self._append_log(
                f'Real step "{label}": no color selected — pick one first',
                'warn')
            return
        if not self._cached_lego_poses:
            self._append_log(
                f'Real step "{label}": lego cache empty — Refresh Legos first',
                'err')
            return
        candidates = [
            (math.hypot(p['x'], p['y']), name)
            for name, p in self._cached_lego_poses.items()
            if name.split('_', 1)[0].lower() == color
        ]
        if not candidates:
            self._append_log(
                f'Real step "{label}": no cached legos of color {color!r}',
                'err')
            return
        candidates.sort()
        _, lego = candidates[0]
        cup_name = self._qs_auto_drop_for_lego(lego)

        # Force real topics + re-inject (idempotent if already real).
        self._real_ensure_real_topics()
        if cup_name and cup_name in self._cached_cup_poses:
            self._real_inject_active_pair(lego, cup_name)
        else:
            # Lego-only inject for steps that don't need a cup
            with self.objects_lock:
                self.objects_data[lego] = dict(
                    self._cached_lego_poses[lego])
        self.root.after(0, self._populate_object_list)
        self.root.after(0, self._populate_drop_list)
        self.root.after(0, self._qs_refresh_objects)

        def _worker():
            time.sleep(0.6)  # let tk flush listbox repopulation
            if not self._qs_select_lego(lego):
                self._append_log(
                    f'Real step "{label}": {lego} not in qs listbox', 'err')
                return
            self._append_log(
                f'Real step "{label}": {lego}'
                + (f' → {cup_name}' if cup_name else ''))
            ok = self._qs_execute_step(method_name, kwargs or {})
            self._append_log(
                f'Real step "{label}": {"✓" if ok else "✗"}')
        threading.Thread(target=_worker, daemon=True).start()

    def _cmd_real_grasp_home(self):
        """Real-mode Grasp Home (debug step) — moves arm to home pose."""
        self._real_run_step('grasp home', '_cmd_grasp_home')

    def _cmd_real_grasp_open(self):
        """Real-mode Grasp Open (debug step) — opens gripper sized for
        the dropdown color's closest cached lego."""
        self._real_run_step('grasp open', '_cmd_gripper_open_for_object')

    def _cmd_real_grasp_move(self):
        """Real-mode Grasp Move (debug step) — IK to dropdown color's
        closest cached lego pose. Uses cached pose only, never live topic."""
        self._real_run_step('grasp move', '_cmd_grasp_move')

    def _cmd_real_grasp_close(self):
        """Real-mode Grasp Close (debug step) — closes gripper on the
        active lego, attaches to gripper if grasp succeeds."""
        self._real_run_step('grasp close', '_cmd_gripper_close_for_object')

    def _cmd_real_drop_point(self):
        """Real-mode Drop Point (debug step) — pans arm to face the
        cached cup matching the dropdown color."""
        self._real_run_step('drop point', '_cmd_drop_point')

    def _cmd_real_drop_sweep(self):
        """Real-mode Drop Sweep (debug step) — IK-plans a sweep from
        carry pose to over the cached cup. Requires a lego attached."""
        self._real_run_step('drop sweep', '_cmd_drop_sweep')

    def _cmd_real_drop_release(self):
        """Real-mode Release (debug step) — opens gripper to release
        the carried lego into the cup."""
        self._real_run_step('release', '_cmd_drop_release')

    def _real_refresh_color_dropdown(self):
        """Recompute combobox values = intersection(cached lego colors,
        cached cup colors) and update widget + Run button enable state.

        Thread-safe — mutates tk widgets via root.after(0, ...). Called on
        every cache mutation: refresh cups, refresh legos, clear cache,
        post-drop evict.

        Wraps the body in try/except so a silent failure (e.g. missing
        attribute after partial hot-reload) surfaces in the log instead of
        being swallowed by the daemon thread that called us.
        """
        try:
            if not hasattr(self, '_real_color_combobox'):
                # Tab not built yet (early hot-reload race) — no widget to update.
                return
            lego_colors = {
                n.split('_', 1)[0].lower() for n in self._cached_lego_poses
            }
            lego_colors &= set(REAL_COLOR_TO_CUP)  # drop unknowns silently
            cup_colors_present = {
                color for color, cup_name in REAL_COLOR_TO_CUP.items()
                if cup_name in self._cached_cup_poses
            }
            available = sorted(lego_colors & cup_colors_present)

            def _apply():
                cb = getattr(self, '_real_color_combobox', None)
                btn = getattr(self, '_real_run_button', None)
                if cb is None:
                    return
                cb['values'] = available
                current = self._real_selected_color_var.get()
                if available:
                    if current not in available:
                        self._real_selected_color_var.set(available[0])
                    if btn is not None:
                        btn.config(state='normal')
                else:
                    self._real_selected_color_var.set('')
                    if btn is not None:
                        btn.config(state='disabled')
            self.root.after(0, _apply)
            self._append_log(f'Real: dropdown colors = {available}')
        except Exception as e:
            self._append_log(
                f'Real: dropdown refresh failed — {type(e).__name__}: {e}',
                'err')

    def _cmd_real_run_one_color(self):
        """Single-pick handler — picks ONE lego of the dropdown's selected
        color and runs it through the full _QS_SEQUENCE with cache-only
        IK inputs (no live /objects_poses_real or /drop_poses topics
        consulted mid-cycle).

        Pre-flight checked:
          - A color is selected in the dropdown.
          - Cup pose cache is non-empty.
          - Lego pose cache is non-empty.
          - No existing QS / Real run is in progress.

        On success: evicts the dropped lego from cache + dropdown shrinks.
        On failure: cache untouched, user can Refresh + retry.
        """
        self._ensure_real_state()
        color = self._real_selected_color_var.get()
        if not color:
            self._append_log(
                'Real: no color selected — Refresh first, then pick a color',
                'warn')
            return
        if not self._cached_cup_poses:
            self._append_log(
                'Real: cup pose cache is empty — Refresh Cups Pose first',
                'err')
            return
        if not self._cached_lego_poses:
            self._append_log(
                'Real: lego pose cache is empty — Refresh Legos Pose first '
                '(at grasp_home pose, where wrist camera sees workspace)',
                'err')
            return
        if self._qs_state == 'running':
            self._append_log(
                'Real: a run is already in progress — '
                'use Quickstart Restart to abort it first', 'warn')
            return
        # Force real topics on the tk thread BEFORE spawning the worker.
        # Worker thread re-injects from cache, but only the real-topic sub
        # prevents sim callbacks from clobbering objects_data / planning
        # scene between IK steps (D-07).
        self._real_ensure_real_topics()
        threading.Thread(
            target=self._real_run_one_color_thread,
            args=(color,), daemon=True).start()

    def _real_run_one_color_thread(self, color):
        try:
            self._append_log(
                'Real: single-pick mode — using cache only, '
                'no live topic source')
            self._real_set_status(f'Running ({color})', step='preparing')
            self._qs_abort_evt.clear()
            self._qs_resume_evt.set()
            self._qs_state = 'running'

            # Pick the closest-to-base lego of the chosen color.
            # Tiebreak: alphabetical on cache key for determinism.
            candidates = [
                (math.hypot(p['x'], p['y']), name, p)
                for name, p in self._cached_lego_poses.items()
                if name.split('_', 1)[0].lower() == color
            ]
            if not candidates:
                self._append_log(
                    f'Real: no cached legos of color {color!r} '
                    '(cache may have been mutated since dropdown refresh)',
                    'err')
                self._real_set_status('Error', step='no candidates')
                return
            candidates.sort(key=lambda t: (t[0], t[1]))
            distance, lego, _pose = candidates[0]

            cup_name = self._qs_auto_drop_for_lego(lego)
            if cup_name is None or cup_name not in self._cached_cup_poses:
                # Defense-in-depth — dropdown filter should prevent this.
                self._append_log(
                    f'Real: {lego} → cup {cup_name!r} not in cache', 'err')
                self._real_set_status('Error', step='cup missing')
                return

            self._append_log(
                f'Real: picked {lego} (dist={distance:.3f} m, '
                f'{len(candidates)} candidate(s) of color {color!r}) '
                f'→ {cup_name}')

            # Re-inject cached poses into the live data structures right
            # before the cycle. _cmd_grasp_move / _cmd_drop_sweep read these,
            # and a stray live-topic callback could otherwise overwrite
            # them with whatever YOLOE last saw (which is partial — the
            # camera no longer sees the full workspace once the arm leaves
            # grasp_home). Cache is the authoritative source here.
            with self._drop_lock:
                self._drop_data[cup_name] = dict(
                    self._cached_cup_poses[cup_name])
            with self.objects_lock:
                self.objects_data.update(self._cached_lego_poses)
            self.root.after(0, self._populate_object_list)
            self.root.after(0, self._populate_drop_list)
            self.root.after(0, self._qs_refresh_objects)
            time.sleep(0.6)  # let tk flush listbox repopulation

            if not self._qs_select_lego(lego):
                self._append_log(
                    f'Real: {lego} not in qs listbox after re-inject — '
                    'check _qs_refresh_objects path', 'err')
                self._real_set_status('Error', step='select failed')
                return

            ok_all = True
            failed_label = None
            for i, (label, method_name, kwargs) in enumerate(
                    self._QS_SEQUENCE, start=1):
                if self._qs_abort_evt.is_set():
                    ok_all = False
                    break
                self._qs_resume_evt.wait()  # honor pause
                # Defense in depth: re-inject cached poses RIGHT BEFORE
                # each step so any stray callback (even from /objects_poses_real
                # — YOLOE may have re-detected with a slightly drifted pose)
                # can't influence the IK target. Cache is the only source.
                self._real_inject_active_pair(lego, cup_name)
                self._real_set_status(
                    f'Running ({color})',
                    step=f'{i}/{len(self._QS_SEQUENCE)}: {label} ({lego})')
                ok = self._qs_execute_step(method_name, kwargs)
                if not ok:
                    ok_all = False
                    failed_label = label
                    self._append_log(
                        f'Real: cycle for {lego} halted at "{label}"', 'err')
                    break

            if self._qs_abort_evt.is_set():
                self._real_set_status('Aborted', step=f'({lego})')
                # No eviction — abort during a successful sequence still
                # shouldn't strip the lego from cache (we don't know if the
                # drop completed atomically).
            elif ok_all:
                self._real_set_status(
                    'Complete', step=f'✓ {lego} → {cup_name}')
                self._append_log(
                    f'Real: cycle complete — {lego} dropped in {cup_name}')
                self._real_evict_lego_from_cache(lego)
            else:
                self._real_set_status(
                    'Halted', step=f'✗ {lego} at "{failed_label}"')
                # No eviction (D-06).
        finally:
            self._qs_state = 'idle'

    def _real_set_status(self, status, step=None):
        """Thread-safe status update for the Real Test tab.

        Mirrors `_append_log`'s `_gui_ready` guard so we never call
        `root.after` from a daemon thread when the tk mainloop has exited
        (or hasn't fully started). Without the guard, any failure inside a
        `_real_*_thread` worker that touches the status label crashes the
        worker with `RuntimeError: main thread is not in main loop` instead
        of running the `finally` cleanup."""
        if not getattr(self, '_gui_ready', False):
            return
        def _update():
            text = status if step is None else f'{status} — {step}'
            self._real_run_status_var.set(text)
        try:
            self.root.after(0, _update)
        except RuntimeError:
            pass

    def _real_evict_lego_from_cache(self, lego_name):
        """Remove a successfully-dropped lego from cache + planning scene.

        Called only on the success path of _real_run_one_color_thread (D-05).
        Failed cycles MUST NOT call this — leaving the lego cached lets
        the user retry after a Refresh (D-06).

        Steps:
          1. Pop from _cached_lego_poses + objects_data (both must drop the
             entry, else the listbox keeps showing a phantom block).
          2. Wipe-then-push lego collisions in MoveIt scene with the now-
             smaller cache. Same 500 ms gap as _real_refresh_legos_thread
             (planning-scene monitor round-trip — see HANDOFF.json
             anti-patterns: ApplyPlanningScene ADD-on-existing-id is not
             cleanly idempotent).
          3. Refresh listboxes + Legos status label + dropdown.
        """
        popped = self._cached_lego_poses.pop(lego_name, None)
        if popped is None:
            self._append_log(
                f'Real: evict noop — {lego_name} not in cache', 'warn')
            return
        with self.objects_lock:
            self.objects_data.pop(lego_name, None)
        # Wipe-then-push to reflect the smaller cache. _add_lego_collision_objects
        # writes _lego_collision_names from scratch so we don't need to
        # surgically remove just one id.
        self._remove_lego_collision_objects()
        time.sleep(0.5)
        if self._cached_lego_poses:
            self._add_lego_collision_objects(
                legos_dict=dict(self._cached_lego_poses))
        self.root.after(0, self._populate_object_list)
        self.root.after(0, self._qs_refresh_objects)
        self.root.after(0, self._real_update_legos_status_label)
        self._real_refresh_color_dropdown()
        self._append_log(
            f'Real: evicted {lego_name} from cache '
            f'(remaining: {len(self._cached_lego_poses)})')

    def _qs_select_lego(self, lego_name):
        """Select a lego in the Quickstart listbox by name. Used by both
        the Real loop and as a generalization of the per-step QS policy.
        Returns False if the listbox is empty or the name isn't present.
        """
        if not hasattr(self, '_qs_listbox') or self._qs_listbox.size() == 0:
            return False
        for i in range(self._qs_listbox.size()):
            if self._qs_listbox.get(i).split('  ')[0] == lego_name:
                self._qs_listbox.selection_clear(0, tk.END)
                self._qs_listbox.selection_set(i)
                return True
        return False

    # ==================================================================
    # Record Sim — closed-loop dataset recording in Isaac Sim
    # ==================================================================
    #
    # Orchestrates the full recording session:
    # 1) spawn lerobot-record subprocess + /joint_commands_lerobot mirror
    # 2) loop scenes (per scene = randomize → run K episodes, one per
    #    pickable lego of the chosen color)
    # 3) per episode: optional reset-to-grasp_home → drive QS pick-place →
    #    wait for "pick-and-drop cycle complete" sentinel
    # 4) on QS halt → re-randomize same scene, retry same lego (up to
    #    REC_MAX_RETRIES_PER_LEGO before skipping)
    # 5) on Stop or session done → SIGINT subprocesses, finalize dataset
    #
    # All blocking work runs on a daemon thread; UI updates marshaled via
    # root.after(0, ...). State stored in self._rec_state dict guarded by
    # self._rec_lock.
    #
    # Constants live at MODULE level (defined just above this class) so
    # hot_reload picks up changes without a full restart — class-level
    # attributes only attach to a fresh class object created on import,
    # not to the running instance's existing __class__.

    def _build_record_sim_tab(self, notebook):
        """Tab: closed-loop sim recording. Spawns lerobot-record + mirror,
        runs a configurable number of pick-and-drop episodes against the
        chosen color, re-randomizing the scene every K episodes."""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='Record Sim')

        # --- State (lazy init — survives hot-reload) ---
        if not hasattr(self, '_rec_lock'):
            self._rec_lock = threading.Lock()
        if not hasattr(self, '_rec_state'):
            self._rec_state = {
                'phase': 'idle',          # idle|starting|running|paused|stopping
                'episode': 0,             # 1-indexed during run, 0 idle
                'n_episodes': 0,
                'scene': 0,
                'last_lego': '',
                'last_verdict': '',
                'dataset_dir': '',
                'lerobot_proc': None,
                'mirror_proc': None,
                'thread': None,
                'stop_evt': threading.Event(),
                'pause_evt': threading.Event(),  # set = run, clear = pause
                'total_retries': 0,
                'resume_mode': False,
            }
            self._rec_state['pause_evt'].set()  # not paused on init

        # --- Tk vars (bound to widgets, persisted via widget registry) ---
        if not hasattr(self, '_rec_episodes_var'):
            self._rec_episodes_var = tk.IntVar(value=16)
        if not hasattr(self, '_rec_color_var'):
            self._rec_color_var = tk.StringVar(value='blue')
        if not hasattr(self, '_rec_dataset_var'):
            self._rec_dataset_var = tk.StringVar(value='')
        if not hasattr(self, '_rec_randomize_legos_var'):
            self._rec_randomize_legos_var = tk.BooleanVar(value=True)
        if not hasattr(self, '_rec_randomize_cups_var'):
            self._rec_randomize_cups_var = tk.BooleanVar(value=False)
        if not hasattr(self, '_rec_reset_arm_var'):
            self._rec_reset_arm_var = tk.BooleanVar(value=True)
        if not hasattr(self, '_rec_pause_between_var'):
            self._rec_pause_between_var = tk.BooleanVar(value=False)
        if not hasattr(self, '_rec_status_var'):
            self._rec_status_var = tk.StringVar(value='IDLE')
        if not hasattr(self, '_rec_progress_var'):
            self._rec_progress_var = tk.StringVar(value='—')
        if not hasattr(self, '_rec_dataset_path_var'):
            self._rec_dataset_path_var = tk.StringVar(value='—')

        # ===== Section: Settings =====
        settings = ttk.LabelFrame(frame, text='Settings')
        settings.pack(fill=tk.X, padx=10, pady=(8, 4))

        row1 = ttk.Frame(settings); row1.pack(fill=tk.X, padx=5, pady=4)
        tk.Label(row1, text='Target episodes:').pack(side=tk.LEFT, padx=(0, 4))
        self._register_spinbox(
            row1, label='Episodes', tab='Record Sim', section='Settings',
            textvariable=self._rec_episodes_var,
            from_=1, to=10000, increment=1, width=6,
        ).pack(side=tk.LEFT, padx=(0, 12))

        tk.Label(row1, text='Block color:').pack(side=tk.LEFT, padx=(0, 4))
        color_cb = ttk.Combobox(
            row1, textvariable=self._rec_color_var,
            values=list(REC_LEGOS_BY_COLOR.keys()),
            state='readonly', width=8,
        )
        color_cb.pack(side=tk.LEFT, padx=(0, 12))
        # Register so widget services can read/set it.
        self._widget_registry_add(
            'Block color', 'Combobox', color_cb,
            self._rec_color_var, tab='Record Sim', section='Settings',
            writable=True)

        row2 = ttk.Frame(settings); row2.pack(fill=tk.X, padx=5, pady=4)
        tk.Label(row2, text='Dataset name:').pack(side=tk.LEFT, padx=(0, 4))
        ds_entry = ttk.Entry(row2, textvariable=self._rec_dataset_var, width=40)
        ds_entry.pack(side=tk.LEFT, padx=(0, 4))
        self._widget_registry_add(
            'Dataset name', 'Entry', ds_entry,
            self._rec_dataset_var, tab='Record Sim', section='Settings',
            writable=True)
        tk.Label(row2, text='(blank → auto rec_<HHMMSS>)',
                 fg='#888').pack(side=tk.LEFT)

        # Per-episode actions
        actions = ttk.LabelFrame(frame, text='Per-episode actions')
        actions.pack(fill=tk.X, padx=10, pady=4)
        for label, var in [
            ('Randomize legos (every episode)',
             self._rec_randomize_legos_var),
            ('Randomize cups (every episode)',
             self._rec_randomize_cups_var),
            ('Reset arm to grasp_home (per episode)',
             self._rec_reset_arm_var),
            ('Pause briefly between episodes (~2.5s)',
             self._rec_pause_between_var),
        ]:
            self._register_check(
                actions, label=label, tab='Record Sim', section='Actions',
                variable=var,
            ).pack(anchor=tk.W, padx=8, pady=1)

        # ===== Section: Control =====
        control = ttk.LabelFrame(frame, text='Control')
        control.pack(fill=tk.X, padx=10, pady=4)
        ctrl_row = ttk.Frame(control); ctrl_row.pack(fill=tk.X, padx=5, pady=6)
        self._rec_btn_start = self._register_button(
            ctrl_row, text='▶ Start', tab='Record Sim', section='Control',
            command=self._cmd_rec_start,
        )
        self._rec_btn_start.pack(side=tk.LEFT, fill=tk.X, expand=True,
                                 padx=2, ipady=4)
        self._rec_btn_pause = self._register_button(
            ctrl_row, text='⏸ Pause', tab='Record Sim', section='Control',
            command=self._cmd_rec_pause,
        )
        self._rec_btn_pause.pack(side=tk.LEFT, fill=tk.X, expand=True,
                                 padx=2, ipady=4)
        self._rec_btn_stop = self._register_button(
            ctrl_row, text='⏹ Stop', tab='Record Sim', section='Control',
            command=self._cmd_rec_stop,
        )
        self._rec_btn_stop.pack(side=tk.LEFT, fill=tk.X, expand=True,
                                padx=2, ipady=4)
        self._register_button(
            ctrl_row, text='↻ Reset Scene', tab='Record Sim', section='Control',
            command=self._cmd_rec_reset_scene,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, ipady=4)
        self._register_button(
            ctrl_row, text='📁 Open Dataset', tab='Record Sim',
            section='Control', command=self._cmd_rec_open_dataset,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, ipady=4)

        # ===== Section: Status =====
        status = ttk.LabelFrame(frame, text='Status')
        status.pack(fill=tk.X, padx=10, pady=4)
        st_row1 = ttk.Frame(status); st_row1.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(st_row1, text='State:', width=10, anchor=tk.W).pack(side=tk.LEFT)
        tk.Label(st_row1, textvariable=self._rec_status_var,
                 font=('TkDefaultFont', 10, 'bold')).pack(side=tk.LEFT)

        st_row2 = ttk.Frame(status); st_row2.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(st_row2, text='Progress:', width=10, anchor=tk.W).pack(
            side=tk.LEFT)
        tk.Label(st_row2, textvariable=self._rec_progress_var).pack(
            side=tk.LEFT)

        st_row3 = ttk.Frame(status); st_row3.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(st_row3, text='Dataset:', width=10, anchor=tk.W).pack(
            side=tk.LEFT)
        tk.Label(st_row3, textvariable=self._rec_dataset_path_var,
                 fg='#0066cc').pack(side=tk.LEFT)

        # Initial button-enabled-ness
        self._rec_update_button_state()

    # -- Helpers -----------------------------------------------------------

    def _rec_set_status(self, status_text=None, progress_text=None,
                        dataset_text=None):
        """Marshal a status update onto the Tk thread."""
        def _apply():
            if status_text is not None:
                self._rec_status_var.set(status_text)
            if progress_text is not None:
                self._rec_progress_var.set(progress_text)
            if dataset_text is not None:
                self._rec_dataset_path_var.set(dataset_text)
            self._rec_update_button_state()
        try:
            self.root.after(0, _apply)
        except Exception:
            pass

    def _rec_update_button_state(self):
        """Enable/disable Start/Pause/Stop based on phase."""
        phase = self._rec_state['phase']
        is_idle = phase == 'idle'
        is_running = phase in ('running',)
        is_paused = phase == 'paused'
        try:
            self._rec_btn_start.configure(
                state=(tk.NORMAL if is_idle else tk.DISABLED))
            self._rec_btn_pause.configure(
                state=(tk.NORMAL if (is_running or is_paused) else tk.DISABLED),
                text=('▶ Resume' if is_paused else '⏸ Pause'))
            self._rec_btn_stop.configure(
                state=(tk.NORMAL if not is_idle else tk.DISABLED))
        except Exception:
            pass

    def _rec_mcp_call(self, cmd_type, params=None, timeout=120):
        """Call a soarm101-dt scene tool on the omni-kit-mcp bridge.

        ``cmd_type`` is the bare tool name (``randomize_object_poses``) and is
        sent as ``soarm101.<cmd_type>``; a dotted name passes through as-is
        (e.g. the ``run_python`` builtin is not namespaced).

        Returns the same envelope the old raw-socket server produced, so
        callers don't change: ``{"status": "success", "result": <payload>}``
        on success, ``{"status": "error", "message": ...}`` when the bridge
        or the tool reported an error, ``None`` when the bridge is
        unreachable."""
        tool = cmd_type if '.' in cmd_type else f'{REC_KIT_MCP_NAMESPACE}.{cmd_type}'
        kit_client = None
        try:
            kit_client = _import_kit_mcp_client()
            payload = kit_client.call(
                tool, params or {}, host=REC_KIT_MCP_HOST,
                port=REC_KIT_MCP_PORT, timeout=timeout)
            return {"status": "success", "result": payload}
        except Exception as exc:
            if kit_client is not None and isinstance(exc, kit_client.BridgeError):
                self._append_log(f'Record: MCP {tool} error: {exc}', 'warn')
                return {"status": "error", "message": str(exc)}
            self._append_log(f'Record: MCP {tool} failed: {exc}', 'warn')
        return None

    def _rec_call_trigger(self, srv_name, timeout=20):
        """Call a /so_arm101_control_gui/<srv_name> Trigger service from this
        node's executor. Returns the Trigger response or None."""
        cli = self.create_client(Trigger, f'~/{srv_name}')
        if not cli.wait_for_service(timeout_sec=5):
            return None
        fut = cli.call_async(Trigger.Request())
        # Spin until done — we're on a background thread, the executor runs
        # in main, so use spin_until_future_complete with the current
        # node + a timeout.
        t0 = time.time()
        while not fut.done() and (time.time() - t0) < timeout:
            time.sleep(0.05)
        if fut.done():
            return fut.result()
        return None

    def _rec_topic_has_publisher(self, topic_name):
        """Return True if any node currently publishes `topic_name`.
        Uses the running node's discovery — no daemon dependency."""
        try:
            infos = self.get_publishers_info_by_topic(topic_name)
            return len(infos) > 0
        except Exception:
            return False

    def _rec_dataset_repo_id(self):
        """Resolve dataset repo_id from the entry; auto-name if blank."""
        name = (self._rec_dataset_var.get() or '').strip()
        if not name:
            name = f"rec_{datetime.datetime.now():%H%M%S}"
        if '/' not in name:
            name = f"local/{name}"
        return name

    # -- Commands ----------------------------------------------------------

    def _cmd_rec_start(self):
        """Spawn lerobot-record + mirror, then start the orchestration thread."""
        if getattr(self, '_inf_state', {}).get('phase', 'idle') != 'idle':
            self._append_log('Record: stop inference before recording scripted motion', 'err')
            return
        if shutil.which('xdotool') is None:
            self._append_log(
                'Record: xdotool is required for episode save/discard; '
                'install with sudo apt install xdotool', 'err')
            return
        with self._rec_lock:
            if self._rec_state['phase'] != 'idle':
                self._append_log('Record: already running', 'warn')
                return
            self._rec_state['phase'] = 'starting'
            self._rec_state['stop_evt'].clear()
            self._rec_state['pause_evt'].set()
            self._rec_state['episode'] = 0
            self._rec_state['scene'] = 0
            self._rec_state['total_retries'] = 0
            self._rec_state['last_verdict'] = ''
            self._rec_state['last_lego'] = ''
            self._rec_state['n_episodes'] = int(self._rec_episodes_var.get())

        repo_id = self._rec_dataset_repo_id()
        ds_path = os.path.join(REC_DATASET_ROOT, repo_id.split('/', 1)[-1])

        # Episodes input is the TARGET TOTAL for the dataset (not per-session).
        # If the dataset already exists, compute remaining = target - existing
        # and pass --resume=true so lerobot appends with continuous episode
        # indices. If remaining <=0, no-op cleanly. Schema compatibility is
        # checked by lerobot itself via sanity_check_dataset_robot_compatibility
        # after LeRobotDataset.resume() loads the existing meta/info.json.
        target_episodes = int(self._rec_episodes_var.get())
        resume_mode = False
        info_path = os.path.join(ds_path, 'meta', 'info.json')
        if os.path.exists(ds_path):
            if not os.path.exists(info_path):
                self._append_log(
                    f'Record: {ds_path} exists but has no meta/info.json — '
                    'incomplete dataset; pick a new name or delete the folder',
                    'err')
                with self._rec_lock:
                    self._rec_state['phase'] = 'idle'
                self._rec_set_status(status_text='IDLE')
                return
            try:
                with open(info_path) as f:
                    existing_total = int(json.load(f).get('total_episodes', 0))
            except Exception as exc:
                self._append_log(
                    f'Record: failed to read {info_path}: {exc}', 'err')
                with self._rec_lock:
                    self._rec_state['phase'] = 'idle'
                self._rec_set_status(status_text='IDLE')
                return
            remaining = target_episodes - existing_total
            if remaining <= 0:
                self._append_log(
                    f'Record: {repo_id} already has {existing_total} episodes '
                    f'(target {target_episodes}); nothing to do')
                with self._rec_lock:
                    self._rec_state['phase'] = 'idle'
                self._rec_set_status(status_text='IDLE')
                return
            resume_mode = True
            with self._rec_lock:
                self._rec_state['n_episodes'] = remaining
            self._append_log(
                f'Record: resuming {repo_id} '
                f'({existing_total} existing → +{remaining} = {target_episodes} target)')
        else:
            self._append_log(f'Record: starting → {repo_id}')

        self._rec_state['resume_mode'] = resume_mode
        self._rec_state['dataset_dir'] = ds_path
        self._rec_set_status(
            status_text='STARTING (spawning subprocesses)',
            progress_text=f'0/{self._rec_state["n_episodes"]}',
            dataset_text=ds_path)

        # Mirror dedup: if /joint_commands_lerobot already has a publisher
        # (mirror left over from a prior session or started manually),
        # reuse it instead of spawning a duplicate. We check the topic
        # publisher count rather than process names because the topic is
        # the actual contract the recorder needs.
        existing_mirror = self._rec_topic_has_publisher(
            '/joint_commands_lerobot')
        if existing_mirror:
            self._append_log(
                'Record: reusing existing /joint_commands_lerobot publisher')
            self._rec_state['mirror_proc'] = None
        else:
            try:
                mp = subprocess.Popen(
                    ['python3', '-u', REC_MIRROR_SCRIPT],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True)
                self._rec_state['mirror_proc'] = mp
            except Exception as exc:
                self._append_log(
                    f'Record: mirror spawn failed: {exc}', 'err')
                self._rec_finalize(error=True)
                return

        # Spawn lerobot-record (pixi-Jazzy via bash wrapper). The task string
        # tracks Block color so the dataset's single_task field reflects what
        # actually got recorded (was hardcoded to "blue" before — drift bug).
        rec_color = self._rec_color_var.get() if hasattr(self, '_rec_color_var') else 'blue'
        single_task = f'Pick a {rec_color} lego and place it in {rec_color} cup'
        cmd = [
            'bash', REC_LEROBOT_SCRIPT,
            f'--dataset.repo_id={repo_id}',
            f'--dataset.num_episodes={self._rec_state["n_episodes"]}',
            f'--dataset.episode_time_s={REC_EPISODE_TIME_S}',
            f'--dataset.reset_time_s={REC_RESET_TIME_S}',
            f'--dataset.single_task={single_task}',
            '--dataset.push_to_hub=false',
            '--play_sounds=false',
            '--display_data=false',
        ]
        if resume_mode:
            # LeRobotDataset.resume() requires an explicit root (won't write
            # into the revision-safe Hub snapshot cache). Pass the absolute
            # path the GUI already computed.
            cmd += [
                '--resume=true',
                f'--dataset.root={ds_path}',
            ]
        try:
            log_path = f"/tmp/rec_{datetime.datetime.now():%H%M%S}.log"
            log_fd = open(log_path, 'w')
            lp = subprocess.Popen(
                cmd, stdout=log_fd, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env={**os.environ, 'LEROBOT_REQUIRE_KEYBOARD': '1'},
                start_new_session=True)
            self._rec_state['lerobot_proc'] = lp
            self._rec_state['lerobot_log'] = log_path
        except Exception as exc:
            self._append_log(f'Record: lerobot spawn failed: {exc}', 'err')
            self._rec_finalize(error=True)
            return

        # Start orchestration thread.
        t = threading.Thread(target=self._rec_loop_thread, daemon=True)
        self._rec_state['thread'] = t
        t.start()

    def _cmd_rec_pause(self):
        """Toggle pause/resume the orchestration loop. lerobot keeps running."""
        with self._rec_lock:
            phase = self._rec_state['phase']
            if phase == 'running':
                self._rec_state['phase'] = 'paused'
                self._rec_state['pause_evt'].clear()
                self._append_log('Record: paused (lerobot still recording)')
                self._rec_set_status(status_text='PAUSED')
            elif phase == 'paused':
                self._rec_state['phase'] = 'running'
                self._rec_state['pause_evt'].set()
                self._append_log('Record: resumed')
                self._rec_set_status(status_text='RUNNING')

    def _cmd_rec_stop(self):
        """Signal the loop to stop, then SIGINT subprocesses."""
        with self._rec_lock:
            if self._rec_state['phase'] == 'idle':
                return
            self._rec_state['phase'] = 'stopping'
            self._rec_state['stop_evt'].set()
            self._rec_state['pause_evt'].set()  # release pause if blocked
        self._rec_set_status(status_text='STOPPING')
        self._append_log('Record: stopping (waiting for clean tear-down…)')
        # Run the rest on a thread so we don't block tk.
        threading.Thread(target=self._rec_finalize_thread, daemon=True).start()

    def _cmd_rec_reset_scene(self):
        """Trigger sim_reset.sh in a subprocess. Idempotent — safe any time."""
        self._append_log('Record: sim_reset')
        try:
            subprocess.Popen(
                ['bash', REC_SIM_RESET_SCRIPT],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL, start_new_session=True)
        except Exception as exc:
            self._append_log(f'Record: reset failed: {exc}', 'err')

    def _cmd_rec_open_dataset(self):
        """xdg-open the most recent dataset folder (or the configured one)."""
        target = self._rec_state.get('dataset_dir') or REC_DATASET_ROOT
        if not os.path.exists(target):
            self._append_log(f'Record: no dataset at {target}', 'warn')
            return
        try:
            subprocess.Popen(['xdg-open', target],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        except Exception as exc:
            self._append_log(f'Record: open failed: {exc}', 'err')

    # -- Subprocess teardown ----------------------------------------------

    def _rec_finalize_thread(self):
        """Background: SIGINT subprocesses, wait for clean exit, then idle."""
        for key in ('lerobot_proc', 'mirror_proc'):
            proc = self._rec_state.get(key)
            if proc and proc.poll() is None:
                try:
                    # SIGINT propagates through start_new_session
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                except Exception:
                    try:
                        proc.send_signal(signal.SIGTERM)
                    except Exception:
                        pass
        # Wait up to ~10s for clean exit, then escalate.
        deadline = time.time() + 10.0
        for key in ('lerobot_proc', 'mirror_proc'):
            proc = self._rec_state.get(key)
            if not proc:
                continue
            while proc.poll() is None and time.time() < deadline:
                time.sleep(0.2)
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
        self._rec_finalize(error=False)

    def _rec_finalize(self, error=False):
        """Reset state to idle. Safe to call from any thread."""
        with self._rec_lock:
            ep_done = self._rec_state['episode']
            ep_target = self._rec_state['n_episodes']
            self._rec_state['phase'] = 'idle'
            self._rec_state['lerobot_proc'] = None
            self._rec_state['mirror_proc'] = None
            self._rec_state['stop_evt'].clear()
            self._rec_state['pause_evt'].set()
        msg = 'IDLE (error)' if error else 'IDLE'
        # Final summary stays in the progress label so the user can see
        # how many episodes landed before they stopped / it completed.
        if ep_target > 0:
            summary = f'last run: {ep_done}/{ep_target} episodes'
        else:
            summary = '—'
        self._rec_set_status(status_text=msg, progress_text=summary)
        self._append_log(f'Record: finalized ({msg}, {summary})')

    # -- Main loop --------------------------------------------------------

    def _rec_loop_thread(self):
        """Orchestration loop. Runs on a daemon background thread.

        Episode contract (Phase 10.2 v3 — parity item #6 fix):
          - 1 episode = 1 random blue lego pick-place.
          - Phase B (lerobot's non-recording reset window) does ALL prep:
            sim_reset (cups + legos + arm) PLUS ik_target + qs_select
            for the NEXT episode's lego. So when Phase A starts, the
            next call into _rec_run_one_qs_cycle is just qs_play + poll.
          - Real episodes are pure motion (~278 frames @ 30fps); pre-
            v3 sim recorded refresh+select inside Phase A and clocked
            ~459 frames per episode. v3 closes that gap.
          - On any failure: lerobot's left-arrow signal discards the
            captured buffer; orchestrator does NOT retry the same lego,
            just re-resets, picks a new random lego, and continues.
          - No retry budget — loop continues until `n_target` SUCCESSES
            are recorded, or stop_evt fires, or lerobot subprocess exits.

        Lerobot phase sync: Phase A (episode_time_s=120s) → Phase B
        (reset_time_s=15s). Each iteration of the main loop does:
          1. Phase A: qs_play + poll  (lerobot is recording)
          2. Send key (Right=save / Left=discard)
          3. Pick next_lego
          4. Phase B: sim_reset + ik_target + qs_select(next_lego)
          5. Wait for new Phase A
          6. current_lego = next_lego, repeat
        """
        st = self._rec_state
        legos = list(REC_LEGOS_BY_COLOR[self._rec_color_var.get()])
        n_target = st['n_episodes']

        # Warm-up wait — block until lerobot logs "Recording episode 0".
        # Lerobot enters Phase A immediately on spawn; we need the buffer
        # to be live before our discard signal lands.
        for _ in range(600):
            if st['stop_evt'].is_set():
                return
            lp = st.get('lerobot_proc')
            if lp and lp.poll() is not None:
                self._append_log(
                    f'Record: recorder exited during startup (rc={lp.returncode}); '
                    f'see {st.get("lerobot_log", "")}', 'err')
                self._cmd_rec_stop()
                return
            if lp:
                log_path = st.get('lerobot_log', '')
                if log_path and os.path.exists(log_path):
                    try:
                        with open(log_path) as f:
                            if 'Recording episode 0' in f.read():
                                break
                    except Exception:
                        pass
            time.sleep(0.2)
        else:
            self._append_log('Record: recorder did not become ready in 120 s', 'err')
            self._cmd_rec_stop()
            return

        # Step 1: discard the just-opened Phase A immediately. Sending
        # Left makes lerobot exit Phase A early (captured ~0s of data),
        # run Phase B (reset_time_s=15s), clear the buffer, then start
        # a fresh Phase A. This is what shifts our initial prep OUT of
        # any saved episode — Phase B is non-recording.
        prev_phase_a = self._rec_count_phase_a_starts()
        self._append_log(
            'Record: discarding initial Phase A so first pick-place lands '
            'in a clean episode')
        self._rec_lerobot_discard()

        # Step 2: pick the FIRST lego now so we can pre-select it during
        # the initial Phase B prep. Subsequent legos are chosen at the
        # tail of each loop iteration.
        current_lego = random.choice(legos)
        st['last_lego'] = current_lego

        # Step 3: do the initial pre-episode setup DURING Phase B.
        # ~12-15s of arm + cup + lego + ik_target + qs_select work;
        # Phase B has 15s budget — tight but reset_time_s can be raised
        # via record_sim_isaac.sh if needed.
        self._rec_set_status(status_text='RESETTING (initial)')
        self._rec_pre_episode_setup(current_lego)

        # Step 4: if initial prep raced past Phase B's 15s budget,
        # lerobot's first post-discard Phase A is already live and has
        # been capturing the tail of our prep — that's how ep1 ended
        # up at 750 frames in v7 vs the 388-frame steady-state. Detect
        # this race by checking whether Phase A advanced DURING prep,
        # and discard the contaminated buffer if so. Steady-state
        # iterations don't pay this cost — only fires when the race
        # actually happened.
        cur_phase_a = self._rec_count_phase_a_starts()
        if cur_phase_a > prev_phase_a:
            self._append_log(
                'Record: initial prep raced past Phase B — '
                'discarding contaminated Phase A so first saved episode '
                'starts clean')
            self._rec_lerobot_discard()
            prev_phase_a = cur_phase_a

        # Step 5: wait for lerobot to enter the fresh Phase A. Same
        # gate the loop uses between iterations — keeps us from racing
        # the next pick-place into Phase B's tail.
        if not self._rec_wait_for_new_phase_a(prev_phase_a, timeout_s=60.0):
            self._cmd_rec_stop()
            return

        with self._rec_lock:
            if st['phase'] != 'starting':
                return
            st['phase'] = 'running'
        self._rec_set_status(status_text='RUNNING')

        success_count = 0
        attempt = 0
        while not st['stop_evt'].is_set():
            # Stop if lerobot subprocess exited (it manages its own
            # n_target and exits when reached or on error).
            lp = st.get('lerobot_proc')
            if lp and lp.poll() is not None:
                self._append_log(
                    f'Record: lerobot exited (rc={lp.returncode})')
                break

            # Belt-and-suspenders: also stop after our own success count
            # hits n_target — covers the case where lerobot's still alive
            # but should have saved enough.
            if success_count >= n_target:
                self._append_log(
                    f'Record: reached n_target={n_target} successes')
                break

            st['pause_evt'].wait()  # block while paused
            attempt += 1
            st['scene'] = attempt
            st['episode'] = success_count

            self._rec_set_status(
                status_text='RUNNING',
                progress_text=(
                    f'episode {success_count + 1}/{n_target}, '
                    f'attempt {attempt}, lego {current_lego}'))

            # Phase A: qs_play + poll only — prep was done in last
            # Phase B (or initial setup for the first iteration).
            verdict = self._rec_run_one_qs_cycle(current_lego)

            if verdict == 'ABORT':
                break
            st['last_verdict'] = verdict

            # Snapshot lerobot's current Phase A and B indices BEFORE signalling.
            # Phase A index gates the next pick-place on lerobot reaching a NEW
            # Phase A (so an early-failing next cycle can't send Left during
            # the current Phase B and discard the previously-successful save —
            # see commit history for the ep6 loss bug). Phase B index lets us
            # wait for lerobot to definitively transition out of Phase A
            # before we start our scene reset (otherwise our randomize call
            # races the tail of the still-recording Phase A).
            prev_phase_a = self._rec_count_phase_a_starts()
            prev_phase_b = self._rec_count_phase_b_starts()

            # Settle window: arm just returned to grasp_home — give lerobot
            # ~2s of clean idle-at-home frames at the END of the episode so
            # the policy sees an unambiguous "task complete" signal in the
            # final observations. Without this, the recorded episode ends
            # while the arm is still decelerating and the last frame's
            # joint state vs target diverges visibly.
            time.sleep(2.0)

            # Signal lerobot via global keyboard event.
            if verdict == 'PASS':
                self._rec_lerobot_advance()
                success_count += 1
                st['episode'] = success_count
                self._append_log(
                    f'Record: ✓ episode {success_count}/{n_target} saved '
                    f'(lego={current_lego}, attempt {attempt})')
            else:
                self._rec_lerobot_discard()
                self._append_log(
                    f'Record: ✗ episode discarded '
                    f'(verdict={verdict}, lego={current_lego}, '
                    f'attempt {attempt})')

            if success_count >= n_target:
                break

            # Wait for lerobot to confirm it has actually exited Phase A.
            # On PASS this means "Reset the environment" line appeared (lerobot
            # is now in its non-recording reset_time_s window — safe to
            # randomize the scene). On discard the rerecord branch in lerobot
            # skips the reset window, so we don't gate on Phase B for that.
            if verdict == 'PASS':
                if not self._rec_wait_for_new_phase_b(prev_phase_b, timeout_s=10.0):
                    self._cmd_rec_stop()
                    return

            # Choose NEXT lego before prep so qs_select can target it.
            current_lego = random.choice(legos)
            st['last_lego'] = current_lego

            # Scene reset: now safe — lerobot is in its non-recording reset_time_s
            # window (or has discarded the buffer). Randomize legos, update cups,
            # send arm to data-home, pre-select next target. Runs in parallel
            # with the remainder of lerobot's reset_time_s budget.
            self._rec_pre_episode_setup(current_lego)

            # Block until lerobot starts its next Phase A. Without this gate
            # an early-failing next cycle could send Left during lerobot's
            # post-Phase-B rerecord-decision window and drop the previous
            # success buffer.
            if not self._rec_wait_for_new_phase_a(prev_phase_a, timeout_s=60.0):
                self._cmd_rec_stop()
                return

        # The last episode has no reset window. Wait for its encoder/finalizer
        # to exit instead of preparing an unused scene or interrupting a save.
        if success_count >= n_target and not st['stop_evt'].is_set():
            self._rec_set_status(status_text='SAVING (final episode)')
            lp = st.get('lerobot_proc')
            if lp:
                try:
                    lp.wait(timeout=120.0)
                except subprocess.TimeoutExpired:
                    self._append_log('Record: final save did not finish in 120 s', 'err')

        # Clean exit
        if not st['stop_evt'].is_set():
            self._append_log(
                f'Record: completed {success_count} successes '
                f'({attempt} attempts, {attempt - success_count} discarded)')
            self._cmd_rec_stop()

    def _rec_count_phase_a_starts(self):
        """Count "Recording episode " lines in lerobot's stdout log.
        Used as a phase-tracking signal — every Phase A start emits one."""
        log_path = self._rec_state.get('lerobot_log', '')
        if not log_path or not os.path.exists(log_path):
            return 0
        try:
            with open(log_path) as f:
                return f.read().count('Recording episode ')
        except Exception:
            return 0

    def _rec_wait_for_new_phase_a(self, prev_count, timeout_s=25.0):
        """Block until lerobot's log shows MORE Phase A starts than
        prev_count, or until timeout_s elapses. Returns True on
        new Phase A detected, False on timeout."""
        st = self._rec_state
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if st['stop_evt'].is_set():
                return False
            cur = self._rec_count_phase_a_starts()
            if cur > prev_count:
                return True
            time.sleep(0.5)
        self._append_log(
            f'Record: WARN — timeout waiting for new Phase A '
            f'(prev={prev_count}, cur={self._rec_count_phase_a_starts()})',
            'warn')
        return False

    def _rec_count_phase_b_starts(self):
        """Count "Reset the environment" lines in lerobot's stdout log.
        Lerobot logs this when it transitions from the recording inner-loop
        (Phase A) into the non-recording reset_time_s inner-loop (Phase B).
        Used as the orchestrator's "Phase A is definitively done" signal —
        we wait for this count to advance after sending Right arrow before
        starting our own scene reset, otherwise the randomize call can race
        the tail of the still-recording Phase A and contaminate the saved
        episode with a teleporting-lego frame.
        """
        log_path = self._rec_state.get('lerobot_log', '')
        if not log_path or not os.path.exists(log_path):
            return 0
        try:
            with open(log_path) as f:
                return f.read().count('Reset the environment')
        except Exception:
            return 0

    def _rec_wait_for_new_phase_b(self, prev_count, timeout_s=10.0):
        """Block until lerobot's log shows MORE Phase B starts than prev_count,
        or until timeout_s elapses. Returns True on new Phase B detected,
        False on timeout. Phase B follows quickly (~50ms) after Right arrow,
        so the default 10s budget is generous."""
        st = self._rec_state
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if st['stop_evt'].is_set():
                return False
            cur = self._rec_count_phase_b_starts()
            if cur > prev_count:
                return True
            time.sleep(0.2)
        self._append_log(
            f'Record: WARN — timeout waiting for new Phase B '
            f'(prev={prev_count}, cur={self._rec_count_phase_b_starts()})',
            'warn')
        return False

    # -- Helpers for new orchestration ------------------------------------

    def _rec_pre_episode_setup(self, lego):
        """Phase-B prep: full sim_reset + ik_target + qs_select for `lego`.

        Runs in lerobot's reset_time_s window (NOT recorded), so the
        next Phase A starts already pointing at a chosen target — the
        recorded episode contains only qs_play motion. This is what
        keeps sim episodes ~278 frames (real-equivalent) instead of
        ~459 frames (the prior layout that did all this prep inside
        the recording window).

        Sequence:
          1. Force-detach any held lego (safe no-op if none attached).
          2. MCP match_real_world → reset CUP_LAYOUT to real-world defaults.
          3. MCP update_cups → re-snap cup prims.
          4. MCP randomize_object_poses → scatter legos for diversity.
          5. qs_refresh_all → resync MoveIt collision scene.
          6. grasp_home_data → arm to calibrated data-collection home.
          7. Set ik_target param to `lego`.
          8. qs_select → choose pick + drop poses for `lego`.
        """
        from rcl_interfaces.srv import SetParameters
        from rcl_interfaces.msg import (
            Parameter, ParameterValue, ParameterType)

        st = self._rec_state
        if st['stop_evt'].is_set():
            return
        self._rec_set_status(status_text='RESETTING')
        self._rec_call_trigger('detach_lego', timeout=5)
        self._rec_mcp_call('match_real_world', {'reposition': True})
        self._rec_mcp_call('update_cups', {})
        # Force at least one lego of the active Block color to land in the
        # workspace camera's FOV — without this, episodes record an empty
        # observation feed (the policy has no target object to ground on).
        rec_color = self._rec_color_var.get() if hasattr(self, '_rec_color_var') else None
        randomize_params = {}
        if rec_color in REC_LEGOS_BY_COLOR:
            randomize_params['required_visible_color'] = rec_color
            randomize_params['required_visible_count'] = 1
        self._rec_mcp_call('randomize_object_poses', randomize_params)
        time.sleep(1.0)  # physics settle
        self._rec_call_trigger('qs_refresh_all', timeout=10)
        time.sleep(2.0)  # listbox repop + collision-scene rebuild
        self._rec_call_trigger('grasp_home_data', timeout=20)

        # Pre-select the next lego — keeps Phase A pure motion.
        if st['stop_evt'].is_set():
            return
        params_cli = self.create_client(
            SetParameters, '~/set_parameters')
        params_cli.wait_for_service(timeout_sec=5)
        p = Parameter(name='ik_target',
                      value=ParameterValue(
                          type=ParameterType.PARAMETER_STRING,
                          string_value=lego))
        fut = params_cli.call_async(SetParameters.Request(parameters=[p]))
        t0 = time.time()
        while not fut.done() and time.time() - t0 < 10:
            time.sleep(0.05)

        if st['stop_evt'].is_set():
            return
        self._rec_call_trigger('qs_select', timeout=15)
        time.sleep(0.5)

    def _rec_lerobot_advance(self):
        """Send Right-arrow to lerobot's keyboard listener → save current
        episode, advance to next one. Picked up by pynput's global X11
        listener regardless of window focus."""
        self._rec_send_key('Right')

    def _rec_lerobot_discard(self):
        """Send Left-arrow to lerobot's keyboard listener → discard the
        episode buffer, redo (which we'll start fresh on the next
        iteration). Picked up by pynput's global X11 listener."""
        self._rec_send_key('Left')

    def _rec_send_key(self, key):
        """Inject a global key event via xdotool. Lerobot's
        init_keyboard_listener (pynput) catches arrow keys with no
        focus dependency. --clearmodifiers prevents stuck Shift/Ctrl
        from prior interactive use."""
        try:
            subprocess.run(
                ['xdotool', 'key', '--clearmodifiers', key],
                check=True, timeout=2,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            self._append_log(
                f'Record: xdotool key {key} failed: {exc}', 'err')
            self._cmd_rec_stop()

    def _rec_run_episode_with_retry(self, lego, episode_idx, n_target):
        """Run one episode for `lego`, re-randomizing on QS halt up to
        REC_MAX_RETRIES_PER_LEGO times. Returns 'PASS' / 'SKIP' / 'ABORT'."""
        st = self._rec_state
        for attempt in range(REC_MAX_RETRIES_PER_LEGO):
            if st['stop_evt'].is_set():
                return 'ABORT'
            st['pause_evt'].wait()

            self._rec_set_status(
                status_text='RUNNING',
                progress_text=(
                    f'Ep {episode_idx + 1}/{n_target}, scene {st["scene"]}, '
                    f'lego {lego} (try {attempt + 1})'))

            # Optional reset to data-collection home before each episode.
            # Uses grasp_home_data (calibrated to real-world ep7 frame 277)
            # so sim recordings start from the same pose as real captures.
            if self._rec_reset_arm_var.get():
                self._rec_call_trigger('grasp_home_data', timeout=20)

            verdict = self._rec_run_one_qs_cycle(lego)
            if verdict == 'PASS':
                return 'PASS'
            if verdict == 'ABORT':
                return 'ABORT'
            # FAIL: re-randomize and retry
            st['total_retries'] += 1
            self._append_log(
                f'Record: episode {episode_idx + 1} failed on {lego} '
                f'(attempt {attempt + 1}); re-randomizing')
            # Same FOV constraint on the retry randomize.
            rec_color_retry = (self._rec_color_var.get()
                               if hasattr(self, '_rec_color_var') else None)
            retry_params = {}
            if rec_color_retry in REC_LEGOS_BY_COLOR:
                retry_params['required_visible_color'] = rec_color_retry
                retry_params['required_visible_count'] = 1
            self._rec_mcp_call('randomize_object_poses', retry_params)
            time.sleep(1.0)

        # Exhausted retries — log and skip.
        self._append_log(
            f'Record: lego {lego} unreachable after '
            f'{REC_MAX_RETRIES_PER_LEGO} retries; advancing', 'warn')
        return 'SKIP'

    def _rec_run_one_qs_cycle(self, lego, timeout_s=90):
        """Phase-A-only motion: qs_play + sentinel poll.

        All scene prep (ik_target, qs_refresh_all, qs_select) is
        performed by _rec_pre_episode_setup during Phase B before this
        runs, so the lerobot-recorded window contains only the
        physical pick-and-drop motion. This is what makes sim episode
        length match real (~278 frames vs the previous ~459).

        Returns 'PASS' / 'FAIL' / 'ABORT'. Per Q1 (Always PASS — never
        verify), success is the 'pick-and-drop cycle complete' sentinel
        in get_log. No pose check.
        """
        st = self._rec_state
        if st['stop_evt'].is_set():
            return 'ABORT'

        # Snapshot pre-cycle log line count.
        # NOTE: log_resp.message contains REAL newlines (\n char), not the
        # escaped two-character literal "\n" that an earlier version of
        # this code split on. The literal-string split produced len=1
        # always, so the "new lines since pre-cycle" slice was empty and
        # the success sentinel was never matched — every successful cycle
        # got mislabeled FAIL after the 90 s timeout, which then fired the
        # discard path on a perfectly valid pick-place. Fixed to split on
        # actual '\n'.
        log_resp = self._rec_call_trigger('get_log', timeout=10)
        if log_resp is None:
            return 'FAIL'
        pre_lines = len(log_resp.message.split('\n'))

        # Fire the motion. This is the ONLY step inside Phase A.
        if st['stop_evt'].is_set():
            return 'ABORT'
        self._rec_call_trigger('qs_play', timeout=15)

        # Poll get_log for terminal sentinel.
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if st['stop_evt'].is_set():
                return 'ABORT'
            st['pause_evt'].wait()
            time.sleep(2.0)
            log_resp = self._rec_call_trigger('get_log', timeout=10)
            if log_resp is None:
                continue
            new = '\n'.join(log_resp.message.split('\n')[pre_lines:])
            if 'pick-and-drop cycle complete' in new:
                return 'PASS'
            if ('Quickstart halted' in new or
                    'Quickstart aborted' in new):
                return 'FAIL'
        return 'FAIL'

    # ==================================================================
    # End Record Sim
    # ==================================================================

    # ==================================================================
    # Inference — closed-loop SmolVLA deployment in Isaac Sim
    # ==================================================================
    #
    # Spawns linux-env/scripts/inference_smolvla.sh as a subprocess (mirrors
    # Record Sim's two-process architecture — control_gui runs system Humble
    # Python 3.10, the inference script runs pixi-Jazzy Python 3.12 because
    # SmolVLA + lerobot need 3.12+).
    #
    # The subprocess is the source of truth for the closed loop: it
    # subscribes to /joint_states + /wrist_camera_rgb_sim + /workspace_camera_sim,
    # runs SmolVLA, and dispatches FollowJointTrajectory action goals
    # directly to the controllers — control_gui is NOT on the critical path
    # (avoids /joint_commands executor saturation).
    #
    # The tab is a launcher. Status comes from log tail + subprocess poll;
    # we don't try to mirror per-step state into the GUI (would couple the
    # GUI to inference cadence).

    def _build_inference_tab(self, notebook):
        """Tab: closed-loop SmolVLA inference. Launches inference_smolvla.sh
        with the chosen model + task; subprocess publishes FJT action goals
        to ros2_control. Hot-reload safe (state lazy-init below)."""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='Inference')

        # --- State (lazy init — survives hot-reload) ---
        if not hasattr(self, '_inf_lock'):
            self._inf_lock = threading.Lock()
        if not hasattr(self, '_inf_state'):
            self._inf_state = {
                'phase': 'idle',     # idle | starting | running | stopping
                'proc': None,
                'log_path': '',
                'started_at': 0.0,
            }

        # --- Tk vars ---
        if not hasattr(self, '_inf_preset_var'):
            self._inf_preset_var = tk.StringVar(
                value=next(iter(INF_MODEL_PRESETS)))
        if not hasattr(self, '_inf_model_path_var'):
            self._inf_model_path_var = tk.StringVar(value='')
        if not hasattr(self, '_inf_checkpoint_var'):
            self._inf_checkpoint_var = tk.StringVar(value='')
        if not hasattr(self, '_inf_task_var'):
            self._inf_task_var = tk.StringVar(value=INF_DEFAULT_TASK)
        if not hasattr(self, '_inf_rename_var'):
            self._inf_rename_var = tk.StringVar(value='{}')
        if not hasattr(self, '_inf_inf_rate_var'):
            self._inf_inf_rate_var = tk.DoubleVar(
                value=INF_DEFAULT_INFERENCE_RATE)
        if not hasattr(self, '_inf_pub_rate_var'):
            self._inf_pub_rate_var = tk.DoubleVar(
                value=INF_DEFAULT_PUBLISH_RATE)
        if not hasattr(self, '_inf_status_var'):
            self._inf_status_var = tk.StringVar(value='IDLE')
        if not hasattr(self, '_inf_log_path_var'):
            self._inf_log_path_var = tk.StringVar(value='—')
        # Default-on convenience: after the inference subprocess exits, drive
        # the arm back to the calibrated data-collection home pose so the
        # next inference / record run starts from a consistent baseline.
        if not hasattr(self, '_inf_return_home_var'):
            self._inf_return_home_var = tk.BooleanVar(value=True)
        # Recording-while-inferring: when ON, _cmd_inf_start additionally
        # spawns the JTC-reference mirror + lerobot-record (same
        # subprocesses Record Sim uses), so the rollout produces a
        # standard v3 dataset. The subprocesses get torn down with
        # inference on Stop (or on inference crash via _inf_watch_proc).
        if not hasattr(self, '_inf_record_var'):
            self._inf_record_var = tk.BooleanVar(value=False)
        if not hasattr(self, '_inf_record_dataset_var'):
            self._inf_record_dataset_var = tk.StringVar(value='')
        # Inference method radio — see INF_METHOD_LABELS for the two options.
        # Default to async_rtc since async inference overlaps inference with execution
        # (no queue-starvation gaps the custom 1Hz replan suffers).
        if not hasattr(self, '_inf_method_var'):
            self._inf_method_var = tk.StringVar(value='async_rtc')

        if not hasattr(self, '_inf_rtc_var'):
            self._inf_rtc_var = tk.BooleanVar(value=True)

        # Apply current preset to the path/checkpoint/rename fields on first
        # build (and on every preset change).
        self._inf_apply_preset()

        # ===== Section: Model =====
        model = ttk.LabelFrame(frame, text='Model')
        model.pack(fill=tk.X, padx=10, pady=(8, 4))

        row1 = ttk.Frame(model); row1.pack(fill=tk.X, padx=5, pady=4)
        tk.Label(row1, text='Preset:').pack(side=tk.LEFT, padx=(0, 4))
        preset_cb = ttk.Combobox(
            row1, textvariable=self._inf_preset_var,
            values=list(INF_MODEL_PRESETS.keys()),
            state='readonly', width=30,
        )
        preset_cb.pack(side=tk.LEFT, padx=(0, 8))
        preset_cb.bind('<<ComboboxSelected>>',
                       lambda _e: self._inf_apply_preset())
        self._widget_registry_add(
            'Inference preset', 'Combobox', preset_cb,
            self._inf_preset_var, tab='Inference', section='Model',
            writable=True)

        row2 = ttk.Frame(model); row2.pack(fill=tk.X, padx=5, pady=4)
        tk.Label(row2, text='Model path:', width=12, anchor='w').pack(side=tk.LEFT)
        path_entry = ttk.Entry(row2, textvariable=self._inf_model_path_var,
                               width=50)
        path_entry.pack(side=tk.LEFT, padx=(0, 4), fill=tk.X, expand=True)
        self._widget_registry_add(
            'Model path', 'Entry', path_entry,
            self._inf_model_path_var, tab='Inference', section='Model',
            writable=True)

        row3 = ttk.Frame(model); row3.pack(fill=tk.X, padx=5, pady=4)
        tk.Label(row3, text='Checkpoint:', width=12, anchor='w').pack(side=tk.LEFT)
        ckpt_entry = ttk.Entry(row3, textvariable=self._inf_checkpoint_var,
                               width=12)
        ckpt_entry.pack(side=tk.LEFT, padx=(0, 4))
        self._widget_registry_add(
            'Checkpoint', 'Entry', ckpt_entry,
            self._inf_checkpoint_var, tab='Inference', section='Model',
            writable=True)
        tk.Label(row3, text='(e.g. 050000; blank → top-level path)',
                 fg='#888').pack(side=tk.LEFT)

        row4 = ttk.Frame(model); row4.pack(fill=tk.X, padx=5, pady=4)
        tk.Label(row4, text='Rename map:', width=12, anchor='w').pack(side=tk.LEFT)
        rename_entry = ttk.Entry(row4, textvariable=self._inf_rename_var,
                                 width=50)
        rename_entry.pack(side=tk.LEFT, padx=(0, 4), fill=tk.X, expand=True)
        self._widget_registry_add(
            'Rename map', 'Entry', rename_entry,
            self._inf_rename_var, tab='Inference', section='Model',
            writable=True)

        # Register a real radio widget and its finite choices, never its LabelFrame.
        method = ttk.LabelFrame(frame, text='Method')
        method.pack(fill=tk.X, padx=10, pady=4)
        for key, label in INF_METHOD_LABELS.items():
            rb = ttk.Radiobutton(
                method, text=label, variable=self._inf_method_var, value=key)
            rb.pack(anchor=tk.W, padx=8, pady=1)
        self._widget_registry_add(
            'Inference method', 'Radiobutton', rb,
            self._inf_method_var, tab='Inference', section='Method',
            writable=True, choices=INF_METHOD_LABELS)
        self._register_check(
            method, label='Real RTC (async method)', tab='Inference',
            section='Method', variable=self._inf_rtc_var,
        ).pack(anchor=tk.W, padx=8, pady=2)

        # ===== Section: Task =====
        task = ttk.LabelFrame(frame, text='Task')
        task.pack(fill=tk.X, padx=10, pady=4)
        task_row = ttk.Frame(task); task_row.pack(fill=tk.X, padx=5, pady=4)
        tk.Label(task_row, text='Instruction:', width=12, anchor='w').pack(side=tk.LEFT)
        task_entry = ttk.Entry(task_row, textvariable=self._inf_task_var, width=60)
        task_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._widget_registry_add(
            'Task instruction', 'Entry', task_entry,
            self._inf_task_var, tab='Inference', section='Task',
            writable=True)

        # ===== Section: Rates =====
        rates = ttk.LabelFrame(frame, text='Rates')
        rates.pack(fill=tk.X, padx=10, pady=4)
        rate_row = ttk.Frame(rates); rate_row.pack(fill=tk.X, padx=5, pady=4)
        tk.Label(rate_row, text='Inference (Hz):').pack(side=tk.LEFT, padx=(0, 4))
        self._register_spinbox(
            rate_row, label='Inference rate', tab='Inference', section='Rates',
            textvariable=self._inf_inf_rate_var,
            from_=0.5, to=10.0, increment=0.5, width=6,
        ).pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(rate_row, text='Client FPS:').pack(side=tk.LEFT, padx=(0, 4))
        self._register_spinbox(
            rate_row, label='Publish rate', tab='Inference', section='Rates',
            textvariable=self._inf_pub_rate_var,
            from_=5.0, to=50.0, increment=0.1, width=6,
        ).pack(side=tk.LEFT)

        # ===== Section: Control =====
        control = ttk.LabelFrame(frame, text='Control')
        control.pack(fill=tk.X, padx=10, pady=4)
        ctrl_row = ttk.Frame(control); ctrl_row.pack(fill=tk.X, padx=5, pady=6)
        self._inf_btn_start = self._register_button(
            ctrl_row, text='▶ Start', tab='Inference', section='Control',
            command=self._cmd_inf_start,
        )
        self._inf_btn_start.pack(side=tk.LEFT, fill=tk.X, expand=True,
                                 padx=2, ipady=4)
        self._inf_btn_stop = self._register_button(
            ctrl_row, text='⏹ Stop', tab='Inference', section='Control',
            command=self._cmd_inf_stop,
        )
        self._inf_btn_stop.pack(side=tk.LEFT, fill=tk.X, expand=True,
                                padx=2, ipady=4)
        self._register_button(
            ctrl_row, text='🔍 Dry-run (no ROS)', tab='Inference',
            section='Control', command=self._cmd_inf_dry_run,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, ipady=4)
        self._register_button(
            ctrl_row, text='📄 Tail log', tab='Inference', section='Control',
            command=self._cmd_inf_tail_log,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, ipady=4)

        # ===== Section: Recovery (post-stop arm pose) =====
        recovery = ttk.LabelFrame(frame, text='Recovery')
        recovery.pack(fill=tk.X, padx=10, pady=4)
        self._register_check(
            recovery, label='Return to grasp_home (data) on Stop',
            tab='Inference', section='Recovery',
            variable=self._inf_return_home_var,
        ).pack(anchor=tk.W, padx=8, pady=2)

        # ===== Section: Record this run (optional) =====
        record = ttk.LabelFrame(frame, text='Record this run')
        record.pack(fill=tk.X, padx=10, pady=4)
        self._register_check(
            record, label='Record observation + action to a v3 dataset',
            tab='Inference', section='Record',
            variable=self._inf_record_var,
        ).pack(anchor=tk.W, padx=8, pady=2)
        rec_row = ttk.Frame(record); rec_row.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(rec_row, text='Dataset name:', width=12, anchor='w').pack(side=tk.LEFT)
        rec_entry = ttk.Entry(
            rec_row, textvariable=self._inf_record_dataset_var, width=30,
        )
        rec_entry.pack(side=tk.LEFT, padx=(0, 4))
        self._widget_registry_add(
            'Inference dataset name', 'Entry', rec_entry,
            self._inf_record_dataset_var, tab='Inference', section='Record',
            writable=True)
        tk.Label(rec_row, text='(blank → auto inf_<HHMMSS>)',
                 fg='#888').pack(side=tk.LEFT)

        # ===== Section: Status =====
        status = ttk.LabelFrame(frame, text='Status')
        status.pack(fill=tk.X, padx=10, pady=4)
        st_row1 = ttk.Frame(status); st_row1.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(st_row1, text='State:', width=10, anchor=tk.W).pack(side=tk.LEFT)
        tk.Label(st_row1, textvariable=self._inf_status_var,
                 font=('TkDefaultFont', 10, 'bold')).pack(side=tk.LEFT)
        st_row2 = ttk.Frame(status); st_row2.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(st_row2, text='Log:', width=10, anchor=tk.W).pack(side=tk.LEFT)
        tk.Label(st_row2, textvariable=self._inf_log_path_var,
                 fg='#0066cc').pack(side=tk.LEFT)

    # -- Helpers ------------------------------------------------------------

    def _inf_apply_preset(self):
        """Copy the selected preset's defaults into the editable fields.
        Custom preset leaves them blank for the user to fill."""
        preset = self._inf_preset_var.get()
        path, ckpt, rename = INF_MODEL_PRESETS.get(preset, ('', '', '{}'))
        if preset != 'Custom':
            self._inf_model_path_var.set(os.path.expanduser(path))
            self._inf_checkpoint_var.set(ckpt)
        # Rename map is always overwritten — it's tightly coupled to the
        # checkpoint's training-time camera-key convention.
        self._inf_rename_var.set(rename)

    def _inf_set_status(self, status_text=None, log_text=None):
        """Marshal a status update onto the Tk thread."""
        def _apply():
            if status_text is not None:
                self._inf_status_var.set(status_text)
            if log_text is not None:
                self._inf_log_path_var.set(log_text)
        try:
            self.root.after(0, _apply)
        except Exception:
            pass

    def _inf_build_cmd(self, dry_run=False):
        """Assemble the bash invocation. Returns (cmd_list, ok, err).

        The chosen Method radio dispatches to the right wrapper:
          * custom    → linux-env/scripts/inference_smolvla.sh
                        (smolvla_inference.py, our chunk-replan node)
          * async_rtc → linux-env/scripts/inference_async_lerobot.sh
                        (run_async_inference_sim.py wrapping
                         lerobot.async_inference.{policy_server,robot_client})

        Both scripts accept --task and a model path, but their flag
        conventions differ — custom uses `--model.path`/`--model.checkpoint`,
        async uses `--pretrained` (HF repo or local dir; checkpoint subfolders
        not supported by lerobot's async inference loader, so we resolve
        them locally here when needed).
        """
        path = os.path.expanduser(self._inf_model_path_var.get().strip())
        if not path:
            return None, False, 'Model path is empty — fill it in.'
        task = self._inf_task_var.get().strip()
        if not task:
            return None, False, 'Task instruction is empty.'
        method = self._inf_method_var.get() if hasattr(self, '_inf_method_var') else 'custom'

        if method == 'async_rtc':
            # Async inference path. lerobot's async loader expects a single
            # path that points DIRECTLY at a `pretrained_model` dir or HF
            # repo root. If the user specified a checkpoint subfolder via
            # the Checkpoint field, splice it onto the path (works for HF
            # repos with `checkpoints/<step>/pretrained_model` layouts).
            ckpt = self._inf_checkpoint_var.get().strip()
            full_path = path
            if ckpt:
                full_path = f'{path}/checkpoints/{ckpt}/pretrained_model'
            cmd = [
                'bash', INF_ASYNC_SCRIPT,
                f'--pretrained={full_path}',
                f'--task={task}',
                f'--fps={float(self._inf_pub_rate_var.get())}',
                '--rtc.enabled' if self._inf_rtc_var.get() else '--no-rtc',
            ]
            if dry_run:
                cmd.append('--dry-run')
            return cmd, True, ''

        # Default: custom chunk-replan path.
        cmd = [
            'bash', INF_SCRIPT,
            f'--model.path={path}',
            f'--task={task}',
            f'--inference-rate={float(self._inf_inf_rate_var.get())}',
            f'--publish-rate={float(self._inf_pub_rate_var.get())}',
        ]
        ckpt = self._inf_checkpoint_var.get().strip()
        if ckpt:
            cmd.append(f'--model.checkpoint={ckpt}')
        rename = self._inf_rename_var.get().strip()
        if rename and rename != '{}':
            cmd.append(f'--rename-map={rename}')
        if dry_run:
            cmd.append('--dry-run')
        return cmd, True, ''

    # -- Commands -----------------------------------------------------------

    def _inf_record_repo_id(self):
        """Resolve dataset repo_id from the record entry; auto-name if blank.
        Mirrors Record Sim's _rec_dataset_repo_id convention so both tabs
        write to the same `local/<name>` namespace under REC_DATASET_ROOT."""
        name = (self._inf_record_dataset_var.get() or '').strip()
        if not name:
            name = f"inf_{datetime.datetime.now():%H%M%S}"
        if '/' not in name:
            name = f"local/{name}"
        return name

    def _cmd_inf_start(self):
        """Spawn the inference subprocess (and optionally mirror +
        lerobot-record). Idempotent — refuses if already running."""
        with self._inf_lock:
            if self._inf_state['phase'] != 'idle':
                self._append_log('Inference: already running', 'warn')
                return
            self._inf_state['phase'] = 'starting'

        cmd, ok, err = self._inf_build_cmd(dry_run=False)
        if not ok:
            self._append_log(f'Inference: {err}', 'err')
            with self._inf_lock:
                self._inf_state['phase'] = 'idle'
            self._inf_set_status(status_text='IDLE')
            return

        if getattr(self, '_rec_state', {}).get('phase', 'idle') != 'idle':
            self._append_log('Inference: stop Record Sim before starting inference', 'err')
            with self._inf_lock:
                self._inf_state['phase'] = 'idle'
            self._inf_set_status(status_text='IDLE')
            return

        # ----- If recording is requested, spin up mirror + lerobot-record
        # BEFORE the inference subprocess so the recorder's 2 s action-
        # warmup sees /joint_commands_lerobot publishing. Mirrors the
        # ordering in _cmd_rec_start (mirror first, then lerobot-record).
        record_on = False
        try:
            record_on = bool(self._inf_record_var.get())
        except Exception:
            record_on = False
        # The recorder has its own ROS subscriptions in a separate process and
        # actuate=false. Only inference sends goals; the mirror reads JTC references.
        rec_repo_id = ''
        rec_ds_path = ''
        if record_on:
            if shutil.which('xdotool') is None:
                self._append_log('Inference: recording needs sudo apt install xdotool', 'err')
                with self._inf_lock:
                    self._inf_state['phase'] = 'idle'
                self._inf_set_status(status_text='IDLE')
                return
            rec_repo_id = self._inf_record_repo_id()
            rec_ds_path = os.path.join(
                REC_DATASET_ROOT, rec_repo_id.split('/', 1)[-1])
            if os.path.exists(rec_ds_path):
                self._append_log(
                    f'Inference: dataset {rec_ds_path} already exists — '
                    'pick a new name (no resume support in inference path)',
                    'err')
                with self._inf_lock:
                    self._inf_state['phase'] = 'idle'
                self._inf_set_status(status_text='IDLE')
                return

            # Mirror dedup — same logic as _cmd_rec_start. The mirror's
            # output topic /joint_commands_lerobot is the actual contract
            # the recorder needs; reuse if a publisher already exists.
            mirror_proc = None
            if self._rec_topic_has_publisher('/joint_commands_lerobot'):
                self._append_log(
                    'Inference: reusing existing /joint_commands_lerobot '
                    'publisher')
            else:
                try:
                    mirror_proc = subprocess.Popen(
                        ['python3', '-u', REC_MIRROR_SCRIPT],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True)
                except Exception as e:
                    self._append_log(
                        f'Inference: mirror spawn failed: {e}', 'err')
                    with self._inf_lock:
                        self._inf_state['phase'] = 'idle'
                    self._inf_set_status(status_text='IDLE')
                    return

            # Spawn lerobot-record. Single long episode — user closes via
            # Stop sends Right so save_episode runs before finalization.
            # num_episodes=1 + episode_time_s=600 gives a 10-minute ceiling.
            single_task = self._inf_task_var.get().strip() or 'inference rollout'
            rec_log = (
                f"/tmp/inf_rec_{datetime.datetime.now():%H%M%S}.log"
            )
            rec_cmd = [
                'bash', REC_LEROBOT_SCRIPT,
                f'--dataset.repo_id={rec_repo_id}',
                '--robot.actuate=false',
                '--dataset.num_episodes=1',
                '--dataset.episode_time_s=600',
                '--dataset.reset_time_s=0',
                f'--dataset.single_task={single_task}',
                '--dataset.push_to_hub=false',
                '--play_sounds=false',
                '--display_data=false',
            ]
            try:
                rec_log_fd = open(rec_log, 'w')
                lerobot_proc = subprocess.Popen(
                    rec_cmd, stdout=rec_log_fd, stderr=subprocess.STDOUT,
                    env={**os.environ, 'LEROBOT_REQUIRE_KEYBOARD': '1'},
                    stdin=subprocess.DEVNULL, start_new_session=True)
            except Exception as e:
                self._append_log(
                    f'Inference: lerobot-record spawn failed: {e}', 'err')
                if mirror_proc is not None:
                    try:
                        os.killpg(os.getpgid(mirror_proc.pid), signal.SIGINT)
                    except Exception:
                        pass
                with self._inf_lock:
                    self._inf_state['phase'] = 'idle'
                self._inf_set_status(status_text='IDLE')
                return

            with self._inf_lock:
                self._inf_state['mirror_proc'] = mirror_proc
                self._inf_state['lerobot_proc'] = lerobot_proc
                self._inf_state['lerobot_log'] = rec_log
                self._inf_state['dataset_dir'] = rec_ds_path
            self._append_log(
                f'Inference: recording → {rec_ds_path}\n'
                f'  lerobot log: {rec_log}'
            )

        # ----- Now spawn the inference subprocess.
        log_path = f'/tmp/inf_{datetime.datetime.now():%H%M%S}.log'
        self._append_log(
            f'Inference: starting subprocess → {log_path}\n'
            f'  cmd = {" ".join(cmd)}'
        )
        try:
            log_fd = open(log_path, 'w')
            proc = subprocess.Popen(
                cmd, stdout=log_fd, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
        except Exception as e:
            self._append_log(f'Inference: spawn failed: {e}', 'err')
            # Tear down mirror + lerobot-record if we already started them.
            self._inf_kill_recorders()
            with self._inf_lock:
                self._inf_state['phase'] = 'idle'
            self._inf_set_status(status_text='IDLE')
            return

        with self._inf_lock:
            self._inf_state['proc'] = proc
            self._inf_state['log_path'] = log_path
            self._inf_state['phase'] = 'running'
            self._inf_state['started_at'] = time.time()
        status = ('RUNNING (recording)' if record_on else 'RUNNING')
        self._inf_set_status(status_text=status, log_text=log_path)

        # Watcher thread: on subprocess exit, transition phase back to idle.
        threading.Thread(target=self._inf_watch_proc, daemon=True).start()

    def _inf_kill_recorders(self):
        """Save the live episode, wait for encoding, then stop the passive mirror.

        Safe when neither process exists; used on Stop and spawn failure.
        """
        with self._inf_lock:
            lp = self._inf_state.get('lerobot_proc')
            mp = self._inf_state.get('mirror_proc')

        if lp is not None and lp.poll() is None:
            try:
                # SIGINT only finalizes already-saved episodes. Right ends the
                # live buffer through lerobot's normal save_episode path.
                subprocess.run(['xdotool', 'key', '--clearmodifiers', 'Right'],
                               check=True, timeout=2,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self._append_log('Inference: saving recorded episode…')
                lp.wait(timeout=120.0)
            except (subprocess.SubprocessError, OSError) as exc:
                self._append_log(f'Inference: recorder save failed: {exc}', 'err')
                try:
                    os.killpg(os.getpgid(lp.pid), signal.SIGINT)
                    lp.wait(timeout=10.0)
                except (OSError, subprocess.TimeoutExpired):
                    if lp.poll() is None:
                        os.killpg(os.getpgid(lp.pid), signal.SIGTERM)
        if lp is not None:
            info_path = os.path.join(self._inf_state.get('dataset_dir', ''), 'meta', 'info.json')
            try:
                with open(info_path) as f:
                    info = json.load(f)
                if int(info.get('total_episodes', 0)) < 1:
                    raise ValueError('no episode saved')
                self._append_log(f'Inference: saved {info["total_frames"]} frames to '
                                 f'{self._inf_state["dataset_dir"]}')
            except (OSError, ValueError, KeyError) as exc:
                self._append_log(f'Inference: recorded dataset incomplete: {exc}; '
                                 f'see {self._inf_state.get("lerobot_log", "")}', 'err')

        if mp is not None and mp.poll() is None:
            try:
                os.killpg(os.getpgid(mp.pid), signal.SIGINT)
            except Exception:
                try:
                    mp.terminate()
                except Exception:
                    pass
            try:
                mp.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                pass

        with self._inf_lock:
            self._inf_state['lerobot_proc'] = None
            self._inf_state['mirror_proc'] = None

    def _inf_watch_proc(self):
        with self._inf_lock:
            proc = self._inf_state.get('proc')
        if proc is None:
            return
        try:
            rc = proc.wait()
        except Exception:
            rc = -1
        with self._inf_lock:
            if self._inf_state.get('phase') in ('running', 'stopping'):
                self._inf_state['phase'] = 'stopping'
        # Surface non-zero exits since the subprocess is the source of truth
        # for inference health (model load failures, OOM, etc. show up here).
        if rc != 0:
            self._append_log(
                f'Inference: subprocess exited with code {rc} '
                f'(see {self._inf_state.get("log_path", "?")})', 'warn')
        else:
            self._append_log('Inference: subprocess exited cleanly')

        # Tear down recording subprocesses (no-op if recording wasn't on).
        # Done BEFORE grasp_home so the dataset captures the final inference
        # state, not the recovery motion (which would pollute the episode).
        self._inf_kill_recorders()
        with self._inf_lock:
            self._inf_state['phase'] = 'idle'
            self._inf_state['proc'] = None

        # Optional: drive arm back to calibrated data-collection home so the
        # next run starts from a known pose. _cmd_grasp_home_data is a Tk-
        # thread method (uses self._motion_event + threading), so marshal
        # via root.after; matches the marshaling pattern in _inf_set_status.
        # The grasp_home action is fire-and-forget here — we don't block the
        # watcher waiting for arm motion (would prevent the watcher from
        # being recycled if the user clicks Start again before the home
        # finishes). The motion still runs to completion in the background.
        try:
            return_home = bool(self._inf_return_home_var.get())
        except Exception:
            return_home = False
        if return_home:
            self._append_log(
                'Inference: dispatching gripper open + grasp_home (data) recovery')
            try:
                # Two-phase recovery:
                #   1. open the gripper (separate controller — _cmd_grasp_home
                #      only commands the arm group, leaves gripper wherever
                #      inference left it, often closed around nothing).
                #   2. drive the arm to the calibrated data home pose.
                # Both are fire-and-forget; the gripper goal completes in
                # ~0.5s while the arm motion takes ~3s, so they overlap
                # cleanly without a planning-group conflict (separate
                # controllers, separate FJT actions).
                #
                # 1.4 rad ≈ 80° — neutral-open jaw matching the
                # GRIPPER_OPEN_RAD value in linux-env/scripts/drive_pick_place.py.
                self.root.after(0, lambda: self._send_gripper_goal(
                    1.4, duration_s=0.6, blocking=False))
                self.root.after(0, self._cmd_grasp_home_data)
            except Exception as e:
                self._append_log(
                    f'Inference: recovery dispatch failed: {e}', 'warn')
        self._inf_set_status(status_text='IDLE')

    def _cmd_inf_stop(self):
        """SIGINT the subprocess; rclpy + torch teardown runs in our signal
        handler. Fall back to SIGTERM after a few seconds if it's stuck."""
        with self._inf_lock:
            proc = self._inf_state.get('proc')
            if proc is None or proc.poll() is not None:
                self._append_log('Inference: nothing to stop')
                self._inf_state['phase'] = 'idle'
                self._inf_set_status(status_text='IDLE')
                return
            self._inf_state['phase'] = 'stopping'
        self._inf_set_status(status_text='STOPPING')
        # SIGINT propagates to the whole process group thanks to
        # start_new_session=True at spawn time.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        except Exception as e:
            self._append_log(f'Inference: SIGINT failed: {e}', 'warn')
            try:
                proc.terminate()
            except Exception:
                pass

    def _cmd_inf_dry_run(self):
        """Run the inference pipeline once without ROS — load the model,
        run a single forward pass on synthetic obs, print the action chunk,
        exit. Useful to verify model load + processor wiring without
        spinning up the closed loop."""
        cmd, ok, err = self._inf_build_cmd(dry_run=True)
        if not ok:
            self._append_log(f'Inference: {err}', 'err')
            return
        log_path = f'/tmp/inf_dryrun_{datetime.datetime.now():%H%M%S}.log'
        self._append_log(f'Inference DRY-RUN → {log_path}\n  cmd = {" ".join(cmd)}')
        try:
            log_fd = open(log_path, 'w')
            subprocess.Popen(
                cmd, stdout=log_fd, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
        except Exception as e:
            self._append_log(f'Inference: dry-run spawn failed: {e}', 'err')
            return
        self._inf_set_status(log_text=log_path)

    def _cmd_inf_tail_log(self):
        """Open the current inference log in xdg-open (or just print location)."""
        log_path = self._inf_state.get('log_path', '')
        if not log_path or not os.path.exists(log_path):
            self._append_log('Inference: no log to tail yet')
            return
        try:
            subprocess.Popen(['xdg-open', log_path],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        except Exception as e:
            self._append_log(f'Inference: tail open failed: {e}', 'warn')

    # ==================================================================
    # End Inference
    # ==================================================================

    def _build_display_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='RViz')

        # --- Cups ---
        cup_frame = ttk.LabelFrame(frame, text='Cups')
        cup_frame.pack(fill=tk.X, padx=10, pady=5)

        self._show_visual_var = tk.BooleanVar(value=True)
        self._register_check(cup_frame, label='Visual (colored cups)',
                             tab='RViz', section='Cups',
                             variable=self._show_visual_var,
                             command=self._toggle_visual_markers).pack(anchor='w', padx=5, pady=2)

        pad_row = tk.Frame(cup_frame)
        pad_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(pad_row, text='Collision padding %:', anchor='w').pack(side=tk.LEFT)
        # 0% default — matches the global _CUP_COLLISION_PADDING=1.0. Any
        # padding > 0 with the current cup positions caused drop_sweep
        # carried-block path to clip the inflated cup mesh sub-mm during
        # pan-across, falling through to OMPL RNG fallback. Bare geometry
        # keeps tier-1 deterministic for grasp_move + drop_sweep both;
        # truly unreachable spawns get rejected and the recording pipeline
        # discards them as failed episodes.
        self._collision_padding_var = tk.DoubleVar(value=0.0)
        self._register_spinbox(pad_row, label='Collision padding %',
                               tab='RViz', section='Cups',
                               textvariable=self._collision_padding_var,
                               from_=0, to=50, increment=0.5, width=5).pack(side=tk.LEFT, padx=(5, 0))
        self._register_button(pad_row, text='Apply', tab='RViz', section='Cups',
                              command=self._cmd_apply_collision_padding).pack(side=tk.LEFT, padx=(5, 0))

        # --- Planning ---
        plan_frame = ttk.LabelFrame(frame, text='Planning')
        plan_frame.pack(fill=tk.X, padx=10, pady=5)

        grip_row = tk.Frame(plan_frame)
        grip_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(grip_row, text='Drop grip angle (deg):', anchor='w').pack(side=tk.LEFT)
        self._drop_grip_angle_var = tk.IntVar(value=45)
        self._register_spinbox(grip_row, label='Drop grip angle (deg)',
                               tab='RViz', section='Planning',
                               textvariable=self._drop_grip_angle_var,
                               from_=0, to=90, increment=5, width=5).pack(side=tk.LEFT, padx=(5, 0))

        attempts_row = tk.Frame(plan_frame)
        attempts_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(attempts_row, text='Planning attempts:', anchor='w').pack(side=tk.LEFT)
        self._planning_attempts_var = tk.IntVar(value=50)
        self._register_spinbox(attempts_row, label='Planning attempts',
                               tab='RViz', section='Planning',
                               textvariable=self._planning_attempts_var,
                               from_=1, to=200, increment=10, width=5).pack(side=tk.LEFT, padx=(5, 0))

        # --- Info ---
        info_frame = ttk.LabelFrame(frame, text='Info')
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(info_frame, text='Collision planning is always ON regardless of display',
                 anchor='w', fg='#555555').pack(anchor='w', padx=5, pady=1)
        tk.Label(info_frame, text='Changes to planning params take effect on next motion',
                 anchor='w', fg='#555555').pack(anchor='w', padx=5, pady=1)

    # ------------------------------------------------------------------
    # IK tab: tab-change, FK, spinbox IK, buttons
    # ------------------------------------------------------------------

    @staticmethod
    def _quat_to_rpy_deg(qx, qy, qz, qw):
        """Convert quaternion to Roll/shoulder_lift/Yaw in degrees."""
        sinr = 2.0 * (qw * qx + qy * qz)
        cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr, cosr)
        sinp = 2.0 * (qw * qy - qz * qx)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))
        siny = 2.0 * (qw * qz + qx * qy)
        cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny, cosy)
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


    def _get_ik_target_quat(self):
        """Read XYZ + RPY from spinboxes, return (x, y, z, qx, qy, qz, qw)."""
        x = self.xyz_vars['X'].get()
        y = self.xyz_vars['Y'].get()
        z = self.xyz_vars['Z'].get()
        qx, qy, qz, qw = self._rpy_deg_to_quat(
            self.rpy_vars['Roll'].get(),
            self.rpy_vars['shoulder_lift'].get(),
            self.rpy_vars['Yaw'].get())
        return x, y, z, qx, qy, qz, qw

    def _on_tab_changed(self, event):
        """Auto-populate IK spinboxes from current EE pose when switching to IK tab."""
        tab_text = self._notebook.tab(self._notebook.select(), 'text')
        if tab_text != 'IK':
            return
        self._compute_fk_to_spinboxes()
        with self.joint_lock:
            self._ik_planned_target = {n: self.joint_positions[n] for n in ARM_JOINT_NAMES}
        self._ik_valid = True

    def _compute_fk_to_spinboxes(self, joint_positions=None):
        """Compute FK and populate IK spinboxes. Uses current joints if None."""
        if not MOVEIT_AVAILABLE or not hasattr(self, 'fk_client'):
            return
        if not self.fk_client.service_is_ready():
            return

        if joint_positions is None:
            with self.joint_lock:
                positions = [self.joint_positions[n] for n in ARM_JOINT_NAMES]
        else:
            positions = [joint_positions[n] for n in ARM_JOINT_NAMES]

        def _call_fk():
            with self._ik_solve_lock:
                request = GetPositionFK.Request()
                request.header.frame_id = 'base'
                request.header.stamp = self.get_clock().now().to_msg()
                request.fk_link_names = ['tcp_link']
                request.robot_state.joint_state.name = list(ARM_JOINT_NAMES)
                request.robot_state.joint_state.position = list(positions)

                future = self.fk_client.call_async(request)
                self._wait_future(future, timeout_sec=2.0)

            if future.result() is None:
                return
            resp = future.result()
            if resp.error_code.val != 1 or not resp.pose_stamped:
                return

            p = resp.pose_stamped[0].pose.position
            o = resp.pose_stamped[0].pose.orientation
            if getattr(self, '_gui_ready', False):
                self.root.after(0, self._populate_ik_spinboxes,
                               p.x, p.y, p.z, o.x, o.y, o.z, o.w)

        threading.Thread(target=_call_fk, daemon=True).start()

    def _populate_ik_spinboxes(self, x, y, z, qx, qy, qz, qw):
        """Set IK spinbox values from FK result and mark state as valid."""
        self._ik_trace_active = False  # suppress IK solves during batch update
        self.xyz_vars['X'].set(round(x, 3))
        self.xyz_vars['Y'].set(round(y, 3))
        self.xyz_vars['Z'].set(round(z, 3))
        r, p, ya = self._quat_to_rpy_deg(qx, qy, qz, qw)
        self.rpy_vars['Roll'].set(round(r, 1))
        self.rpy_vars['shoulder_lift'].set(round(p, 1))
        self.rpy_vars['Yaw'].set(round(ya, 1))
        self._ik_trace_active = True
        # Store as last valid
        for key in ['X', 'Y', 'Z']:
            self._ik_last_valid[key] = self.xyz_vars[key].get()
        for key in ['Roll', 'shoulder_lift', 'Yaw']:
            self._ik_last_valid[key] = self.rpy_vars[key].get()
        # Mark valid, clear red
        self._ik_valid = True
        for spin in self._ik_spinboxes.values():
            spin.config(fg='black')

    def _cmd_ik_reset(self):
        """Reset IK tab: zero arm and populate spinboxes from resulting FK."""
        self._cmd_zero_arm()
        def _after_zero():
            with self.joint_lock:
                pos = {n: self.joint_positions[n] for n in ARM_JOINT_NAMES}
            self._ik_planned_target = dict(pos)
            self._ik_valid = True
            self._compute_fk_to_spinboxes(pos)
        if getattr(self, '_gui_ready', False):
            self.root.after(500, _after_zero)

    # --- Spinbox IK: debounced, serialized ---

    def _on_ik_var_changed(self, field):
        """Any IK spinbox variable changed. Debounce then solve IK."""
        if not self._ik_trace_active:
            self.get_logger().debug(f'IK trace suppressed: {field}')
            return  # suppress during programmatic batch updates
        self.get_logger().info(f'IK var changed: {field}')
        if self._ik_debounce_id is not None:
            self.root.after_cancel(self._ik_debounce_id)
        self._ik_debounce_id = self.root.after(
            150, lambda: self._ik_solve_interactive(field))

    def _ik_solve_interactive(self, changed_field, revert_on_fail=False):
        """Compute IK from current spinbox values. Serialized via lock."""
        if not MOVEIT_AVAILABLE or self.ik_client is None:
            self.get_logger().warn('IK solve skipped: MoveIt not available')
            return
        if not self.ik_client.service_is_ready():
            self.get_logger().warn('IK solve skipped: /compute_ik not ready')
            return

        self._ik_debounce_id = None
        x, y, z, qx, qy, qz, qw = self._get_ik_target_quat()
        self.get_logger().info(
            f'IK solve starting: {changed_field} -> ({x:.3f}, {y:.3f}, {z:.3f})')

        with self.joint_lock:
            current_joints = [self.joint_positions[n] for n in ARM_JOINT_NAMES]

        seeds = [
            list(current_joints),
            [math.atan2(-y, x) if abs(x) + abs(y) > 0.001 else 0.0,
             0.0, 0.0, 0.0, 0.0],
            [0.0] * len(ARM_JOINT_NAMES),
        ]

        self._ik_solve_gen += 1
        gen = self._ik_solve_gen

        def _make_ik_request(seed, avoid_collisions):
            request = GetPositionIK.Request()
            ik_req = PositionIKRequest()
            ik_req.group_name = 'arm'
            ik_req.avoid_collisions = avoid_collisions
            robot_state = RobotState()
            robot_state.joint_state.name = list(ARM_JOINT_NAMES)
            robot_state.joint_state.position = list(seed)
            ik_req.robot_state = robot_state
            pose = PoseStamped()
            pose.header.frame_id = 'base'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            ik_req.pose_stamped = pose
            ik_req.timeout.sec = 0
            ik_req.timeout.nanosec = 500_000_000  # 0.5s cap per request
            request.ik_request = ik_req
            return request

        def _extract_joints(result):
            sol = result.solution.joint_state
            target = {}
            for i, name in enumerate(sol.name):
                if name in ARM_JOINT_NAMES and i < len(sol.position):
                    target[name] = sol.position[i]
            return target

        def _solve():
            if not self._ik_solve_lock.acquire(blocking=False):
                self.get_logger().info('IK solve: lock busy, skipping')
                return  # another solve running, debounce will retry
            try:
                # Pass 1: collision-aware IK (valid + safe)
                for i, seed in enumerate(seeds):
                    request = _make_ik_request(seed, avoid_collisions=True)
                    future = self.ik_client.call_async(request)
                    result = self._wait_future(future, timeout_sec=2.0)
                    code = result.error_code.val if result else 'N/A'
                    self.get_logger().info(
                        f'IK pass1 seed{i}: done={future.done()}, code={code}')
                    if result is not None and result.error_code.val == 1:
                        target = _extract_joints(result)
                        self.get_logger().info('IK SUCCESS (collision-free)')
                        if getattr(self, '_gui_ready', False):
                            self.root.after(0, self._ik_interactive_success,
                                           target, gen)
                        return

                # Pass 2: collision-ignored IK (reachable but colliding)
                for i, seed in enumerate(seeds):
                    request = _make_ik_request(seed, avoid_collisions=False)
                    future = self.ik_client.call_async(request)
                    result = self._wait_future(future, timeout_sec=2.0)
                    code = result.error_code.val if result else 'N/A'
                    self.get_logger().info(
                        f'IK pass2 seed{i}: done={future.done()}, code={code}')
                    if result is not None and result.error_code.val == 1:
                        target = _extract_joints(result)
                        self.get_logger().info('IK COLLISION (goal shown red)')
                        if getattr(self, '_gui_ready', False):
                            self.root.after(0, self._ik_interactive_fail_with_goal,
                                           target, changed_field, gen)
                        return

                # Truly unreachable — no solution even ignoring collisions
                self.get_logger().info('IK UNREACHABLE (all seeds failed)')
                if getattr(self, '_gui_ready', False):
                    self.root.after(0, self._ik_interactive_fail,
                                   changed_field, revert_on_fail, gen)
            finally:
                self._ik_solve_lock.release()

        threading.Thread(target=_solve, daemon=True).start()

    def _ik_interactive_success(self, target, gen):
        """IK succeeded — update goal state, clear red."""
        if gen != self._ik_solve_gen:
            return  # stale

        self._ik_valid = True
        self._ik_planned_target = dict(target)

        for key in ['X', 'Y', 'Z']:
            self._ik_last_valid[key] = self.xyz_vars[key].get()
        for key in ['Roll', 'shoulder_lift', 'Yaw']:
            self._ik_last_valid[key] = self.rpy_vars[key].get()

        # Clear all red
        for spin in self._ik_spinboxes.values():
            spin.config(fg='black')

        # Update sliders + goal state
        self._slider_driven = True
        self._select_planning_group('arm')
        with self.joint_lock:
            for name in ARM_JOINT_NAMES:
                if name in target:
                    self.joint_positions[name] = target[name]
        for name in ARM_JOINT_NAMES:
            if name in target and name in self.sliders:
                self.sliders[name].set(target[name])
                self.slider_labels[name].config(text=f'{target[name]:.3f}')
        self._publish_goal_state()

    def _ik_interactive_fail_with_goal(self, target, changed_field, gen):
        """IK collision — publish colliding goal (RViz shows red), mark field red."""
        if gen != self._ik_solve_gen:
            return  # stale

        self._ik_valid = False
        self._ik_planned_target = None  # not executable

        # Mark the offending field red
        self._ik_spinboxes[changed_field].config(fg='red')

        # Publish the colliding solution so goal robot shows RED in RViz
        self._slider_driven = True
        self._select_planning_group('arm')
        with self.joint_lock:
            for name in ARM_JOINT_NAMES:
                if name in target:
                    self.joint_positions[name] = target[name]
        for name in ARM_JOINT_NAMES:
            if name in target and name in self.sliders:
                self.sliders[name].set(target[name])
                self.slider_labels[name].config(text=f'{target[name]:.3f}')
        self._publish_goal_state()

        self._append_log(
            f'IK collision — {changed_field} causes collision', 'warn')

    def _ik_interactive_fail(self, changed_field, revert_on_fail, gen):
        """IK truly unreachable — mark field red, log warning."""
        if gen != self._ik_solve_gen:
            return  # stale

        self._ik_valid = False

        # Mark the offending field red
        self._ik_spinboxes[changed_field].config(fg='red')
        self._append_log(
            f'IK unreachable — {changed_field} value out of workspace', 'warn')

    # --- IK buttons (always enabled, show warning if invalid) ---

    def _ik_btn_set_joints(self):
        """Send currently planned IK joints directly to controllers."""
        if not self._ik_valid or self._ik_planned_target is None:
            self._append_log('IK solution not found — adjust target first', 'warn')
            return
        self._cmd_set_joints()

    def _ik_btn_plan_execute(self):
        """Plan & Execute from currently planned IK joints via MoveIt."""
        if not self._ik_valid or self._ik_planned_target is None:
            self._append_log('IK solution not found — adjust target first', 'warn')
            return
        self._cmd_plan_execute()

    # --- IK services (for programmatic access) ---

    def _cmd_ik_set_joints(self):
        """Service: send IK joints to controllers."""
        if self._ik_valid and self._ik_planned_target is not None:
            self._cmd_set_joints()
        else:
            self._compute_ik_full(mode='set_joints')

    def _cmd_ik_plan_execute(self):
        """Service: plan & execute IK joints."""
        if self._ik_valid and self._ik_planned_target is not None:
            self._cmd_plan_execute()
        else:
            self._compute_ik_full(mode='plan_execute')

    def _cmd_ik_randomize(self):
        """Randomize arm goal state, compute FK, populate spinboxes."""
        self._cmd_randomize_arm()
        def _fk_after_randomize():
            with self.joint_lock:
                pos = {n: self.joint_positions[n] for n in ARM_JOINT_NAMES}
            self._ik_planned_target = dict(pos)
            self._ik_valid = True
            joints_str = ', '.join(f'{n}: {pos[n]:.3f}' for n in ARM_JOINT_NAMES)
            self._append_log(f'Randomized goal: {joints_str}')
            self._compute_fk_to_spinboxes(pos)
        if getattr(self, '_gui_ready', False):
            self.root.after(1500, _fk_after_randomize)

    # --- Full IK solver (for services, 6 seeds + 2 passes) ---

    # ------------------------------------------------------------------
    # IK pipeline — three entry points (Phase 7 Plan 07-04 / D-10..D-13):
    #
    #   Solvers (return candidate joint dicts):
    #     _solve_grasp_ik   — geometric, top-down grasp poses only (~100µs)
    #     _solve_free_ik    — MoveIt KDL multi-seed for freeform xyz+quaternion
    #
    #   Execution wrapper (solve + collision-check + execute):
    #     _plan_collision_free_execute — geometric + MoveIt path planning
    #                                    (used by drop sweep, planned grasp)
    #
    #   Post-solve helpers (shared by both solvers):
    #     _check_state_valid  — collision validity via MoveIt planning scene
    #     _ik_apply_and_act   — sync sliders, publish goal state, dispatch mode
    #
    # Motion entry points:
    #   _execute_trajectory — direct-send for trajectory-shaped moves
    #   _cmd_plan_execute   — MoveIt-planned collision-aware execution
    #   _cmd_set_joints     — 0.5s slider-quick-move (bypasses trajectory by
    #                         design — quick-move semantics differ)
    # ------------------------------------------------------------------

    def _solve_grasp_ik(self, x, y, z, grasp_yaw, mode=None):
        """Geometric top-down grasp IK + collision check (no MoveIt fallback).

        Fast analytical solver (~100µs). Validates each solution against the
        MoveIt planning scene. On success, dispatches to _ik_apply_and_act(mode).
        Runs on a background daemon thread (matches historic _try_geometric).
        """
        def _try_geometric():
            with self._ik_solve_lock:
                ground_z = (self._ground_z_var.get()
                            if hasattr(self, '_ground_z_var') else None)
                ok, reason = check_grasp_reachable(x, y, z, ground_z=ground_z)
                if not ok:
                    self._append_log(
                        f'Grasp rejected: {reason} '
                        f'({x:.3f}, {y:.3f}, {z:.3f})', 'warn')
                    return

                yaw_used, stage_sols, dbg_lines = find_reachable_grasp_yaw(
                    [('grasp', x, y, z)], grasp_yaw)
                if yaw_used is None:
                    self._append_log(
                        f'Grasp unreachable: no geometric IK at any yaw '
                        f'({x:.3f}, {y:.3f}, {z:.3f})', 'warn')
                    for line in dbg_lines:
                        self._append_log(line, 'warn')
                    return
                solutions = stage_sols['grasp']
                if yaw_used != grasp_yaw:
                    self._append_log(
                        f'Grasp yaw fallback: '
                        f'{math.degrees(grasp_yaw):.1f}° → '
                        f'{math.degrees(yaw_used):.1f}° '
                        f'(Δ{math.degrees(yaw_used - grasp_yaw):+.1f}°)')

                for i, sol in enumerate(solutions):
                    config = 'elbow-up' if i == 0 else 'elbow-down'
                    if self._check_state_valid(sol):
                        self._append_log(
                            f'Geometric IK: {config}, '
                            f'wrist_roll='
                            f'{math.degrees(sol["wrist_roll"]):.1f}°')
                        self._ik_apply_and_act(sol, mode)
                        return
                    self._append_log(
                        f'Geometric IK: {config} collides', 'warn')

                self._append_log(
                    f'Grasp unreachable: all geometric IK solutions collide '
                    f'at yaw {math.degrees(yaw_used):.1f}° '
                    f'({x:.3f}, {y:.3f}, {z:.3f})', 'warn')

        threading.Thread(target=_try_geometric, daemon=True).start()

    def _compute_ik_full(self, mode=None, target_pose=None, grasp_yaw=None):
        """IK routing layer: geometric for grasps, MoveIt KDL for freeform.

        See _solve_grasp_ik (grasp_yaw path) and _solve_free_ik (freeform
        xyz+quaternion path). Both solvers call _ik_apply_and_act on success
        to sync sliders and dispatch the execution mode.

        target_pose: optional (x, y, z, qx, qy, qz, qw) to bypass spinboxes.
        grasp_yaw: desired jaw-line direction (rad) for top-down alignment.
        """
        if target_pose is not None:
            x, y, z, qx, qy, qz, qw = target_pose
        else:
            x, y, z, qx, qy, qz, qw = self._get_ik_target_quat()

        if grasp_yaw is not None:
            self._solve_grasp_ik(x, y, z, grasp_yaw, mode=mode)
            return

        if not MOVEIT_AVAILABLE or self.ik_client is None:
            self._append_log('moveit_msgs not installed', 'warn')
            return
        if not self.ik_client.service_is_ready():
            self._append_log('compute_ik service not ready', 'warn')
            return
        self._solve_free_ik(x, y, z, qx, qy, qz, qw, mode, grasp_yaw)

    def _solve_free_ik(self, x, y, z, qx, qy, qz, qw, mode, grasp_yaw):
        """MoveIt multi-seed IK solver — used for non-grasp moves only."""
        if not MOVEIT_AVAILABLE or self.ik_client is None:
            self._append_log('moveit_msgs not installed', 'warn')
            return
        if not self.ik_client.service_is_ready():
            self._append_log('compute_ik service not ready', 'warn')
            return

        with self.joint_lock:
            current_joints = [self.joint_positions[n] for n in ARM_JOINT_NAMES]

        bearing = math.atan2(-y, x) if abs(x) + abs(y) > 0.001 else 0.0

        seeds = [
            list(current_joints),
            [bearing, 0.0, 0.0, 0.0, 0.0],
            [0.0] * len(ARM_JOINT_NAMES),
        ]
        for _ in range(3):
            seeds.append([random.uniform(*JOINT_LIMITS[n]) for n in ARM_JOINT_NAMES])

        def _try_seeds():
            with self._ik_solve_lock:
                for avoid_col in [True, False]:
                    for seed in seeds:
                        request = GetPositionIK.Request()
                        ik_req = PositionIKRequest()
                        ik_req.group_name = 'arm'
                        ik_req.avoid_collisions = avoid_col

                        robot_state = RobotState()
                        robot_state.joint_state.name = list(ARM_JOINT_NAMES)
                        robot_state.joint_state.position = list(seed)
                        ik_req.robot_state = robot_state

                        pose = PoseStamped()
                        pose.header.frame_id = 'base'
                        pose.header.stamp = self.get_clock().now().to_msg()
                        pose.pose.position.x = x
                        pose.pose.position.y = y
                        pose.pose.position.z = z
                        pose.pose.orientation.x = qx
                        pose.pose.orientation.y = qy
                        pose.pose.orientation.z = qz
                        pose.pose.orientation.w = qw
                        ik_req.pose_stamped = pose
                        ik_req.timeout.sec = 0
                        ik_req.timeout.nanosec = 500_000_000  # 0.5s cap
                        request.ik_request = ik_req

                        future = self.ik_client.call_async(request)
                        self._wait_future(future, timeout_sec=2.0)

                        if future.result() is None:
                            continue
                        resp = future.result()
                        if resp.error_code.val == 1:
                            sol = resp.solution.joint_state
                            target = {}
                            for i, name in enumerate(sol.name):
                                if name in ARM_JOINT_NAMES \
                                        and i < len(sol.position):
                                    target[name] = sol.position[i]
                            self._ik_apply_and_act(target, mode)
                            return

            self._append_log(
                f'IK failed for ({x:.3f}, {y:.3f}, {z:.3f})', 'warn')

        threading.Thread(target=_try_seeds, daemon=True).start()




    def _set_ik_status(self, text):
        self.ik_status_var.set(text)
        self._append_log(text)


    def _update_ee_display(self, x, y, z, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        self.ee_labels['X'].set(f'{x:.4f}')
        self.ee_labels['Y'].set(f'{y:.4f}')
        self.ee_labels['Z'].set(f'{z:.4f}')
        self.ee_labels['qx'].set(f'{qx:.4f}')
        self.ee_labels['qy'].set(f'{qy:.4f}')
        self.ee_labels['qz'].set(f'{qz:.4f}')
        self.ee_labels['qw'].set(f'{qw:.4f}')





    # ------------------------------------------------------------------
    # Drop target subscription + helpers
    # ------------------------------------------------------------------


    def _update_drop_topic(self, topic='/drop_poses'):
        """Switch drop subscription to topic."""
        if self._drop_sub is not None:
            self.destroy_subscription(self._drop_sub)
        with self._drop_lock:
            self._drop_data.clear()
        self._remove_cup_collision_objects()
        self._drop_sub = self.create_subscription(
            TFMessage, topic, self._drop_callback, 10,
            callback_group=self._sub_cb_group)
        self._append_log(f'Drop topic: {topic}')

    def _populate_drop_list(self):
        """Refresh drop_listbox from _drop_data (call on GUI thread)."""
        if not hasattr(self, '_drop_listbox'):
            return
        self._drop_listbox.delete(0, tk.END)
        with self._drop_lock:
            for name, pos in self._drop_data.items():
                label = DROP_ID_LABELS.get(name, "")
                label_str = f" [{label}]" if label else ""
                self._drop_listbox.insert(
                    tk.END,
                    f'{name}{label_str}  ({pos["x"]:.3f}, {pos["y"]:.3f}, {pos["z"]:.3f})')
        count = self._drop_listbox.size()
        if count > 0:
            self._append_log(f'Drop targets refreshed: {count} found')




    # ------------------------------------------------------------------
    # Phase 9: lego block collision tracking in MoveIt planning scene
    # ------------------------------------------------------------------
    # Blocks enter the scene at grasp_refresh as world CollisionObjects.
    # On successful gripper_close_for_object they convert to AttachedCollisionObject
    # on tcp_link — MoveIt then tracks the carried block's world pose via FK, so
    # plans during drop_sweep respect the full carried-block envelope.
    # On drop_release the attached object is removed from the scene entirely.









    # ------------------------------------------------------------------
    # AttachedCollisionObject lifecycle — standard MoveIt 2.5.9 path.
    # Pre-2.5.9 required a 10 Hz world-tracker workaround (see Phase 9
    # notes in prior revisions). The bug is fixed: /check_state_validity
    # correctly reports attached ↔ world collision pairs, so the workaround
    # is gone and touch_links on the AttachedCollisionObject handles the
    # gripper-vs-held-block allow semantics.
    # ------------------------------------------------------------------

    @scene_transaction
    def _refresh_attached_pose(self):
        """Re-snapshot the attached lego's tcp_link-local offset from Isaac's
        current /objects_poses_sim reading. The attached body is rigid once
        set, but the physical block slides/settles in the jaws over time
        (gravity + compliance in Isaac). Calling this right before plan
        operations keeps MoveIt's collision geometry aligned with reality.
        One-shot — no timer.

        Returns True if the attached pose was updated, False if there was
        nothing to refresh (no lego attached, no bbox, no fresh pose, or
        the block appears to have slipped out of the gripper).
        """
        if not getattr(self, '_attached_lego_name', None):
            return False
        if not MOVEIT_AVAILABLE or not hasattr(self, '_apply_scene_client'):
            return False
        obj_name = self._attached_lego_name
        bbox = self._lookup_bbox(obj_name)
        if not bbox:
            return False
        # Synchronize TF and block pose streams — same rationale as
        # _attach_lego_to_gripper. If arm still has residual velocity,
        # refresh produces an offset-biased pose identical to the original
        # attach bug we're fixing.
        self._wait_arm_at_rest(timeout_sec=0.8)
        with self.objects_lock:
            block = dict(self.objects_data.get(obj_name, {}))
        if not block:
            return False

        import numpy as np
        try:
            t = self._tf_buffer.lookup_transform(
                'base', 'tcp_link', rclpy.time.Time())
        except Exception:
            return False
        tr = t.transform
        tcp_mat = self._pose7_to_mat(
            tr.translation.x, tr.translation.y, tr.translation.z,
            tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w)
        block_mat = self._pose7_to_mat(
            float(block['x']), float(block['y']), float(block['z']),
            float(block.get('qx', 0.0)), float(block.get('qy', 0.0)),
            float(block.get('qz', 0.0)), float(block.get('qw', 1.0)))
        block_in_tcp = np.linalg.inv(tcp_mat) @ block_mat
        ax, ay, az, aqx, aqy, aqz, aqw = self._mat_to_pose7(block_in_tcp)
        dist_mm = float((ax*ax + ay*ay + az*az) ** 0.5) * 1000.0
        if dist_mm > 80.0:
            self._append_log(
                f'Refresh skipped: block {dist_mm:.0f}mm from tcp_link '
                f'(likely slipped from gripper)', 'warn')
            return False

        name_id = f'lego_{obj_name}'
        # Humble quirk: a single diff with REMOVE+ADD of the same attached
        # id is a no-op (success=True but the old attached stays put). Two
        # sequential diffs — REMOVE then ADD — actually re-seats the pose.
        rm_scene = PlanningSceneMsg()
        rm_scene.is_diff = True
        rm_scene.robot_state.is_diff = True
        aco_rm = AttachedCollisionObject()
        aco_rm.link_name = 'tcp_link'
        aco_rm.object.id = name_id
        aco_rm.object.operation = CollisionObject.REMOVE
        rm_scene.robot_state.attached_collision_objects.append(aco_rm)

        add_scene = PlanningSceneMsg()
        add_scene.is_diff = True
        add_scene.robot_state.is_diff = True
        kind, geom = self._build_lego_geometry(obj_name)
        if kind is None:
            return False
        aco = AttachedCollisionObject()
        aco.link_name = 'tcp_link'
        aco.object.id = name_id
        aco.object.header.frame_id = 'tcp_link'
        aco.object.operation = CollisionObject.ADD
        pose = Pose()
        pose.position.x = float(ax); pose.position.y = float(ay)
        pose.position.z = float(az)
        pose.orientation.x = float(aqx); pose.orientation.y = float(aqy)
        pose.orientation.z = float(aqz); pose.orientation.w = float(aqw)
        if kind == 'mesh':
            aco.object.meshes.append(geom)
            aco.object.mesh_poses.append(pose)
        else:  # 'box'
            aco.object.primitives.append(geom)
            aco.object.primitive_poses.append(pose)
        aco.touch_links = ['tcp_link', 'gripper', 'jaw']
        add_scene.robot_state.attached_collision_objects.append(aco)

        try:
            fut = self._apply_scene_client.call_async(
                ApplyPlanningScene.Request(scene=rm_scene))
            self._wait_future(fut, timeout_sec=3.0)
            # Ignore REMOVE result: returns False if the attached didn't
            # exist yet (e.g., first refresh right after grasp).
            fut = self._apply_scene_client.call_async(
                ApplyPlanningScene.Request(scene=add_scene))
            self._wait_future(fut, timeout_sec=3.0)
            res = fut.result()
            if not (res and res.success):
                return False
        except Exception:
            return False
        tracer.event('attached_pose_refreshed',
                     obj_name=obj_name,
                     block_in_tcp=(ax, ay, az),
                     block_in_tcp_quat=(aqx, aqy, aqz, aqw),
                     dist_mm=dist_mm)
        return True

    def _cmd_detach_lego(self):
        """Manual/test-harness detach — force-remove every known lego from the
        robot's attached list, regardless of _attached_lego_name's current value.
        Useful after a crashed cycle left a stale AttachedCollisionObject.
        """
        self._ensure_lego_state()
        if not MOVEIT_AVAILABLE or not hasattr(self, '_apply_scene_client'):
            return

        def _apply():
            if not self._wait_scene_service(self._apply_scene_client, timeout_sec=5.0):
                return
            get_req = GetPlanningSceneSrv.Request()
            get_req.components.components = 4  # ROBOT_STATE_ATTACHED_OBJECTS
            native = self._wait_future(self._get_scene_client.call_async(get_req), timeout_sec=5.)
            if native is None:
                raise RuntimeError('Cannot query MoveIt attachments for manual detach')
            names = [a.object.id for a in native.scene.robot_state.attached_collision_objects
                     if a.object.id.startswith('lego_')]
            scene = PlanningSceneMsg()
            scene.is_diff = True
            scene.robot_state.is_diff = True
            for name in names:
                aco = AttachedCollisionObject()
                aco.link_name = 'tcp_link'
                aco.object.id = name
                aco.object.operation = CollisionObject.REMOVE
                scene.robot_state.attached_collision_objects.append(aco)
                rm = CollisionObject()
                rm.id = name
                rm.operation = CollisionObject.REMOVE
                scene.world.collision_objects.append(rm)
            if names:
                req = ApplyPlanningScene.Request()
                req.scene = scene
                self._wait_future(self._apply_scene_client.call_async(req), timeout_sec=5.)
            observed = self._wait_future(self._get_scene_client.call_async(get_req), timeout_sec=5.)
            if observed is None or any(a.object.id.startswith('lego_') for a in observed.scene.robot_state.attached_collision_objects):
                raise RuntimeError('MoveIt still reports a lego attachment after manual detach')
            self._attached_lego_name = None
            self._attached_lego_tcp_offset = None
            self._append_log('Detached all legos (force cleanup)')

        self._dispatch_scene_update(_apply)
        # Repopulate world legos shortly after
        self.root.after(300, self._add_lego_collision_objects)

    def _detach_lego(self):
        """Non-blocking detach wrapper. Spawns the apply in a thread so
        callers that don't need to wait (e.g. _cmd_grasp_refresh) aren't
        blocked. For drop_release, use `_detach_lego_sync` instead — it
        waits for the planning-scene apply to commit BEFORE returning,
        which is required so the subsequent grasp_home plans against a
        scene without the (now phantom) attached block.
        """
        threading.Thread(target=self._detach_lego_sync, daemon=True).start()


    def _publish_cup_visual_markers(self, cups_dict=None):
        """Publish colored cup meshes as RViz visual markers.

        Reads cup poses from `self._drop_data` by default. If `cups_dict`
        is provided (e.g. real-mode cached cups from Refresh Cups Pose),
        uses that snapshot instead — bypasses the live /drop_poses
        subscription so the visual matches the user-frozen scan, not
        whatever the live (potentially sim) topic is currently publishing.

        Mirrors the cups_dict pattern in `_add_cup_collision_objects` so
        visual + collision share the same source-of-truth in real mode.
        """
        if not hasattr(self, '_cup_visual_pub'):
            from rclpy.qos import QoSProfile, DurabilityPolicy
            self._cup_visual_pub = self.create_publisher(
                MarkerArray, '/cup_visual_markers_array',
                QoSProfile(depth=5, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        if cups_dict is not None:
            drop_items = dict(cups_dict)
        else:
            with self._drop_lock:
                drop_items = dict(self._drop_data)
        if not drop_items:
            return
        scale = _CUP_STL_SCALE * _CUP_COLLISION_PADDING
        ma = MarkerArray()
        for i, (name, pos) in enumerate(drop_items.items()):
            r, g, b, a = _CUP_VISUAL_COLORS.get(name, (0.5, 0.5, 0.5, 0.99))
            m = VisMarker()
            m.header.frame_id = 'base'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'cup_visual'
            m.id = i
            m.type = VisMarker.MESH_RESOURCE
            m.action = VisMarker.ADD
            m.mesh_resource = _get_cup_stl_uri()
            m.pose.position.x = pos['x']
            m.pose.position.y = pos['y']
            # Cup STL origin is at base. /drop_poses publishes body-center,
            # so shift down by half height to render the mesh base on ground.
            m.pose.position.z = pos['z'] - CUP_BODY_HEIGHT_M / 2.0
            # Phase 9: honor cup orientation from /drop_poses (was identity)
            m.pose.orientation.x = float(pos.get('qx', 0.0))
            m.pose.orientation.y = float(pos.get('qy', 0.0))
            m.pose.orientation.z = float(pos.get('qz', 0.0))
            m.pose.orientation.w = float(pos.get('qw', 1.0))
            m.scale = Vector3(x=scale, y=scale, z=scale)
            from std_msgs.msg import ColorRGBA
            m.color = ColorRGBA(r=r, g=g, b=b, a=a)
            m.lifetime = Duration(sec=0)
            m.frame_locked = True
            ma.markers.append(m)
        self._cup_visual_pub.publish(ma)

    def _cmd_apply_collision_padding(self):
        """Update collision padding and re-add collision objects + visual markers."""
        global _CUP_COLLISION_PADDING
        pct = self._collision_padding_var.get()
        _CUP_COLLISION_PADDING = 1.0 + pct / 100.0
        # Clear cached mesh so it rebuilds with new padding
        if hasattr(_load_cup_mesh, '_cached'):
            del _load_cup_mesh._cached
        self._append_log(f'Collision padding: {pct}% ({_CUP_COLLISION_PADDING:.2f}x)')
        # Re-add collision objects and visual markers with new padding
        self._add_cup_collision_objects()
        self.root.after(300, self._refresh_display_markers)

    def _cmd_apply_home_speed(self):
        """Sync _home_speed_var into the home_velocity_scale ROS2 param so
        _cmd_grasp_home picks it up on the next motion. 0.0 = no override
        (use the planning default velocity_scale_var); values in (0, 1.0]
        override velocity_scale for the grasp_home plan only.
        """
        from rclpy.parameter import Parameter as RclParam
        v = float(self._home_speed_var.get())
        self.set_parameters([RclParam('home_velocity_scale', RclParam.Type.DOUBLE, v)])
        label = 'no override' if v <= 0.0 else f'{v:.2f}'
        self._append_log(f'Home speed scale: {label}')

    def _refresh_display_markers(self):
        """Republish visual markers based on current toggle state."""
        if getattr(self, '_show_visual_var', None) and self._show_visual_var.get():
            self._publish_cup_visual_markers()

    def _toggle_visual_markers(self):
        """Toggle colored cup visual markers on/off."""
        if self._show_visual_var.get():
            self._publish_cup_visual_markers()
        else:
            self._delete_visual_markers()


    def _delete_visual_markers(self):
        """Delete cup visual markers."""
        if not hasattr(self, '_cup_visual_pub'):
            return
        ma = MarkerArray()
        for i in range(3):
            m = VisMarker()
            m.header.frame_id = 'base'
            m.ns = 'cup_visual'
            m.id = i
            m.action = VisMarker.DELETE
            ma.markers.append(m)
        self._cup_visual_pub.publish(ma)


    def _get_selected_drop_pose(self):
        """Return (name, x, y, z) for the selected drop listbox entry, or None."""
        if not hasattr(self, '_drop_listbox'):
            return None
        sel = self._drop_listbox.curselection()
        if not sel:
            self._append_log('No drop target selected', 'warn')
            return None
        text = self._drop_listbox.get(sel[0])
        # Strip label annotation: "drop_3 [red]  (...)" → "drop_3"
        raw = text.split('  ')[0]  # "drop_3 [red]" or "drop_3"
        name = raw.split(' [')[0]  # "drop_3"
        with self._drop_lock:
            pose = self._drop_data.get(name)
        if pose is None:
            self._append_log(f'Drop target {name!r} not in data', 'warn')
            return None
        return name, pose['x'], pose['y'], pose['z']

    def _cmd_grasp_update_topic(self):
        """Switch object subscription to topic from GUI entry and auto-refresh."""
        new_topic = self._grasp_topic_var.get().strip()
        if not new_topic:
            return
        # Destroy old subscription and create new one
        if hasattr(self, 'objects_sub') and self.objects_sub is not None:
            self.destroy_subscription(self.objects_sub)
        with self.objects_lock:
            self.objects_data.clear()
        self.objects_sub = self.create_subscription(
            TFMessage, new_topic, self._objects_callback, 10,
            callback_group=self._sub_cb_group)
        # Update button text
        if hasattr(self, '_grasp_move_btn'):
            if new_topic == '/drop_poses':
                self._grasp_move_btn.config(text='Move to Drop')
            else:
                self._grasp_move_btn.config(text='Move to Grab')
        self._append_log(f'Grasp topic: {new_topic}')

    def _cmd_grasp_refresh(self):
        """Resync object state from /objects_poses_sim. Async — schedules
        listbox repopulate on the tk thread via root.after(500, ...).
        Callers that need guaranteed-fresh state should sleep ~1s after.

        Forces sub topics back to sim before pulling — without this, a prior
        Real Test session leaves `objects_sub` on /objects_poses_real and
        the 500 ms refill grabs real-mode poses into the sim collision
        scene (the D-08 leak the user reported)."""
        if not hasattr(self, 'obj_listbox'):
            return
        if tracer.is_active():
            tracer.close_cycle('user_canceled',
                               note='grasp_refresh while cycle open')
        # Reset subs to sim BEFORE clearing — otherwise the next live
        # callback could refill objects_data with real-mode data before
        # the timed re-populate fires.
        self._sim_ensure_sim_topics()
        with self.objects_lock:
            self.objects_data.clear()
        self.obj_listbox.delete(0, tk.END)
        self._ensure_lego_state()
        if self._attached_lego_name:
            self._detach_lego_sync()
        self._remove_lego_collision_objects()
        if getattr(self, '_gui_ready', False):
            self.root.after(500, self._populate_object_list)
            self.root.after(700, self._add_lego_collision_objects)

    def _populate_object_list(self):
        if not hasattr(self, 'obj_listbox'):
            return
        self.obj_listbox.delete(0, tk.END)
        with self.objects_lock:
            for name, pos in self.objects_data.items():
                self.obj_listbox.insert(
                    tk.END,
                    f'{name}  ({pos["x"]:.3f}, {pos["y"]:.3f}, {pos["z"]:.3f})')
        count = self.obj_listbox.size()
        if count > 0:
            self._append_log(f'Objects refreshed: {count} found')



    # Tier-2 intermediate pose — all non-pan joints lifted above the cup
    # plane. shoulder_pan is held at whatever the current segment needs
    # (start_pan for the retract, target_pan for the settle).




    def _joint_space_collision_free_execute_LEGACY_TIER1(
            self, target, on_complete_event, duration_s=3.0, waypoints=50):
        """Legacy tier1-first path kept for reference. Do not call — see the
        new version above for the always-OMPL discipline."""
        with self.joint_lock:
            current = {n: self._actual_positions.get(n, self.joint_positions.get(n, 0.0))
                       for n in ARM_JOINT_NAMES}

        target_full = {n: target.get(n, current[n]) for n in ARM_JOINT_NAMES}

        # Short-circuit: if current is already within 0.5° of target on every
        # joint, there's nothing to do. Previously this would still dispatch
        # a zero-delta 51-waypoint trajectory, return success instantly, and
        # confuse the user into thinking a motion completed ("robot didn't
        # move but service said OK"). Now we explicitly flag the no-op case
        # in _last_motion_status so the caller can distinguish "moved to
        # target" from "already at target".
        ALREADY_AT_TARGET_TOL = math.radians(0.5)  # 0.0087 rad
        max_delta = max(abs(target_full[n] - current[n]) for n in ARM_JOINT_NAMES)
        if max_delta < ALREADY_AT_TARGET_TOL:
            self._append_log(
                f'  already at target (max Δ={math.degrees(max_delta):.2f}° '
                f'< 0.5°) — no motion dispatched')
            tracer.event('tier1_noop',
                         max_delta_deg=math.degrees(max_delta),
                         target=dict(target_full), current=dict(current))
            self._last_motion_status = {
                'ok': True, 'outcome': 'already_at_target',
                'msg': (f'already at target '
                        f'(max Δ={math.degrees(max_delta):.2f}°)')}
            on_complete_event.set()
            return True

        def _segment_valid(a, b, N):
            """Return (ok, invalid_info, contacts). ok=True if all wps valid."""
            for i in range(N + 1):
                alpha = i / N
                q = {n: a[n] + alpha * (b[n] - a[n]) for n in ARM_JOINT_NAMES}
                valid, contacts = self._check_state_valid_with_contacts(q)
                if not valid:
                    return False, (i, alpha), contacts
            return True, None, []

        # Step 1: try direct joint-space interpolation (fast, deterministic).
        ok, bad, bad_contacts = _segment_valid(current, target_full, waypoints)
        tracer.event('tier1_check_done',
                     ok=bool(ok),
                     bad_wp=(None if ok else bad[0]),
                     bad_alpha=(None if ok else bad[1]),
                     waypoints=waypoints,
                     target=dict(target_full),
                     current=dict(current),
                     contacts=([{
                         'a': c.contact_body_1, 'b': c.contact_body_2,
                         'depth_mm': c.depth * 1000,
                     } for c in bad_contacts[:10]] if not ok else []))
        if ok:
            self._append_log(
                f'  joint-space direct: {waypoints+1} wps clean, executing')
            # DIAG: dump the exact waypoints tier1 validated
            try:
                import json as _json
                _wps = []
                for i in range(waypoints + 1):
                    alpha = i / waypoints
                    q = {n: current[n] + alpha * (target_full[n] - current[n])
                         for n in ARM_JOINT_NAMES}
                    _wps.append({'t': alpha * duration_s,
                                 'positions': [q[n] for n in ARM_JOINT_NAMES]})
                with open('/tmp/last_arm_trajectory.json', 'w') as _f:
                    _json.dump({'source': 'tier1',
                                'joint_names': list(ARM_JOINT_NAMES),
                                'points': _wps}, _f)
            except Exception:
                pass
            def _tier1_done():
                # _execute_trajectory runs a UI animation thread and calls
                # on_complete after it returns. It doesn't currently wire
                # into the action-client result callback, so we can only
                # report "animation finished" — NOT that the physical
                # controller converged. For now, treat finishing the
                # animation as success; _send_arm_goal handles the actual
                # controller dispatch internally.
                # TODO: if we need controller-level verdict for tier1, wire
                # _send_arm_goal to surface the action result.
                self._last_motion_status = {
                    'ok': True, 'outcome': 'completed',
                    'msg': f'tier1 joint-space ({waypoints+1} wps) complete'}
                on_complete_event.set()
            self._execute_trajectory(
                target_full, duration_s=duration_s,
                on_complete=_tier1_done)
            return True

        # Step 2: direct path collides — fall back to OMPL, one-shot.
        self._append_log(
            f'  joint-space direct: wp[{bad[0]}]@α={bad[1]:.2f} collides — '
            f'falling back to OMPL (one-shot)')
        return self._ompl_plan_validate_execute(
            target_full, on_complete_event)



    def _trajectory_all_valid(self, joint_trajectory):
        """Return True if every waypoint of joint_trajectory is collision-free.

        Uses /check_state_validity (same service as _check_state_valid, but
        operates on a trajectory point instead of a dict).
        """
        if not joint_trajectory.points:
            return False
        names = list(joint_trajectory.joint_names)
        for pt in joint_trajectory.points:
            state = {n: p for n, p in zip(names, pt.positions)}
            if not self._check_state_valid(state):
                return False
        return True


    def _cmd_validate_last_trajectory(self):
        """Re-run the sub-segment post-check on the most recently executed
        trajectory (/tmp/last_arm_trajectory.json) against the CURRENT
        /planning_scene and log detailed results.

        Purpose: close the blind spot where the pre-execution post-check
        said "all clear" but the physical motion still contacted a cup.
        This service replays the same check so we can (a) reproduce the
        verdict after the fact, (b) test with different subsample counts,
        (c) confirm the scene contains the cups we expect before blaming
        the checker.

        Reports to GUI log:
          - trajectory file path, wps, n_sub checked
          - first invalid state (if any): waypoint index + sub-t + contacts
          - full contact list written to /tmp/trajectory_validation_report.json
        """
        import json as _json
        import os as _os
        path = getattr(self, '_last_trajectory_path',
                       '/tmp/last_arm_trajectory.json')
        if not _os.path.isfile(path):
            self._append_log(f'validate_last_trajectory: no file at {path}', 'warn')
            self._cmd_error = 'no last trajectory'
            return
        try:
            dump = _json.load(open(path))
        except Exception as e:
            self._append_log(f'validate_last_trajectory: load failed: {e}', 'err')
            self._cmd_error = f'load failed: {e}'
            return

        # Rebuild a minimal JointTrajectory for the existing checker.
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        jt = JointTrajectory()
        jt.joint_names = list(dump.get('joint_names', []))
        for pt in dump.get('points', []):
            jp = JointTrajectoryPoint()
            jp.positions = [float(x) for x in pt['positions']]
            jt.points.append(jp)

        chk = self._trajectory_first_invalid_with_contacts(jt)
        report = {
            'file': path,
            'n_wps': chk.n_wps,
            'n_sub_checked': chk.n_sub,
            'subsamples_per_segment': TRAJ_SUBSEGMENT_SAMPLES,
            'ok': chk.ok,
        }
        if chk.ok:
            self._append_log(
                f'validate_last_trajectory OK: {chk.n_wps} wps + {chk.n_sub} '
                f'sub-states clear (file={_os.path.basename(path)})')
        else:
            where = (f'wp[{chk.bad_wp}]' if chk.bad_subidx is None
                     else f'wp[{chk.bad_wp}→{chk.bad_wp + 1}] sub-t={chk.sub_t:.2f}')
            contacts = [{'a': c.contact_body_1, 'b': c.contact_body_2,
                         'depth_mm': float(c.depth) * 1000.0}
                        for c in chk.contacts]
            summary = '; '.join(
                f'{c["a"]}↔{c["b"]}(d={c["depth_mm"]:.1f}mm)'
                for c in contacts[:5]) or 'no contact info'
            report['first_invalid'] = {
                'bad_wp': chk.bad_wp, 'bad_subidx': chk.bad_subidx,
                'sub_t': chk.sub_t, 'contacts': contacts}
            self._append_log(
                f'validate_last_trajectory FAIL: {where} '
                f'(checked {chk.n_wps} wps + {chk.n_sub} sub-states): '
                f'{summary} (file={_os.path.basename(path)})', 'warn')
        try:
            with open('/tmp/trajectory_validation_report.json', 'w') as f:
                _json.dump(report, f, indent=2)
        except Exception:
            pass


    # ------------------------------------------------------------------
    # Hot-reload services (so agents can trigger reload without GUI focus)
    # ------------------------------------------------------------------

    def _cmd_hot_reload(self):
        """Hot-reload logic only (same as Ctrl+R). Updates methods + constants."""
        self._hot_reload_logic()

    def _cmd_hot_reload_gui(self):
        """Hot-reload logic + rebuild GUI (same as Ctrl+Shift+R)."""
        self._hot_reload_gui()

    def _build_dump_services_output(self):
        """Build the markdown button↔service mapping + service inventory.

        Used by both:
          - _srv_dump_services (Trigger service — returns the string in response.message)
          - scripts/regen_agent_debug_guide.sh (pastes the string into docs/AGENT_DEBUG_GUIDE.md)
        """
        lines = []
        lines.append('# Button ↔ Service Mapping (auto-generated from dump_services)')
        lines.append('')
        lines.append('> Regenerated from control_gui.py via '
                     '`ros2 service call /so_arm101_control_gui/dump_services '
                     'std_srvs/srv/Trigger {}`. Do NOT hand-edit — rerun the '
                     'service after code changes.')
        lines.append('')

        cmd_methods = {n for n in dir(self) if n.startswith('_cmd_')}
        wrapper_methods = {n for n in dir(self)
                           if '_btn_' in n and n.startswith('_')}

        # Table 1 — Button map
        lines.append('## Buttons')
        lines.append('')
        lines.append('| Tab | Section | Button | Bound Method | Registered Service | Mapping OK |')
        lines.append('|-----|---------|--------|--------------|--------------------|------------|')
        bound_cmds = set()
        for entry in self._button_registry:
            cmd_name = entry.get('command_name', '')
            tab = entry.get('tab') or '-'
            section = entry.get('section') or '-'
            text = entry.get('text', '')
            if cmd_name.startswith('_cmd_') and cmd_name in cmd_methods:
                svc = '~/' + cmd_name[5:]
                ok = '✓'
                bound_cmds.add(cmd_name)
            elif cmd_name in wrapper_methods:
                svc = '(wrapper)'
                ok = 'WRAPPER'
            else:
                svc = '-'
                ok = '❌ INLINE LAMBDA'
            lines.append(f'| {tab} | {section} | {text} | `{cmd_name}` | {svc} | {ok} |')

        # Table 2 — Services without buttons (agent-only)
        lines.append('')
        lines.append('## Agent-only services (no button binding)')
        lines.append('')
        lines.append('| Service | Method |')
        lines.append('|---------|--------|')
        for name in sorted(cmd_methods):
            if name in bound_cmds:
                continue
            lines.append(f'| `~/{name[5:]}` | `{name}` |')

        return '\n'.join(lines)

    def _srv_dump_services(self, request, response):
        """Manual service — returns the button↔service mapping as markdown."""
        try:
            response.message = self._build_dump_services_output()
            response.success = True
        except Exception as e:
            response.message = f'dump_services error: {e}'
            response.success = False
        return response

    # ------------------------------------------------------------------
    # Drop motion commands (auto-registered as ~/drop_refresh, ~/drop_select,
    # ~/drop_point, ~/drop_sweep, ~/drop_release Trigger services)
    # ------------------------------------------------------------------

    def _drop_btn_update_topic(self):
        """GUI-only wrapper: read the drop topic entry and switch subscription.

        This is a thin _*_btn_* wrapper (not auto-registered as a service)
        because the agent-callable path for changing topics is the parameter
        + _cmd_drop_refresh pattern, not a Trigger service.
        """
        topic = self._drop_topic_var.get().strip()
        self._update_drop_topic(topic)

    def _cmd_drop_refresh(self):
        """Resync cup state from /drop_poses. Mirrors _cmd_grasp_refresh:
        clear _drop_data, remove current collision objects, schedule
        repopulate + re-add + marker republish at fixed delays. The
        /drop_poses subscription refills _drop_data at ~12 Hz so ≥6
        messages arrive within the 500 ms scheduled read window — safe
        to clear without leaving the listbox blank for long.

        Clearing matters because /drop_poses child_frame_ids changed
        (drop_0/1/2 → drop_1/2/3 after the cup-ArUco-ID remap); without
        a clear, the stale drop_0 entry from a previous run lingers in
        _drop_data forever and shows up in every listbox refresh.

        Forces sub topics back to sim before pulling so a prior Real
        Test session can't leak /drop_poses_real data into the sim cup
        collision scene.
        """
        if not hasattr(self, '_drop_listbox'):
            return
        if tracer.is_active():
            tracer.close_cycle('user_canceled',
                               note='drop_refresh while cycle open')
        self._sim_ensure_sim_topics()
        with self._drop_lock:
            self._drop_data.clear()
        self._drop_listbox.delete(0, tk.END)
        self._remove_cup_collision_objects()
        if getattr(self, '_gui_ready', False):
            self.root.after(500, self._populate_drop_list)
            self.root.after(700, self._add_cup_collision_objects)
            self.root.after(900, self._refresh_display_markers)

    def _cmd_drop_select(self):
        """Select a drop target by name (via ik_target param).
        Usage: ros2 param set ... ik_target "drop_1" then call this service.
        Errors if no name provided (matches GUI behavior — must explicitly choose).
        """
        if not hasattr(self, '_drop_listbox') or self._drop_listbox.size() == 0:
            self._append_log('No drop targets to select', 'warn')
            return
        name_hint = self.get_parameter('ik_target').get_parameter_value().string_value.strip()
        if not name_hint or not name_hint.startswith('drop_'):
            self._append_log(
                'No drop target specified. Set param first: '
                'ros2 param set /so_arm101_control_gui ik_target "drop_1"',
                'warn')
            return
        for i in range(self._drop_listbox.size()):
            entry = self._drop_listbox.get(i)
            entry_name = entry.split(' [')[0]
            if entry_name == name_hint:
                self._drop_listbox.selection_clear(0, tk.END)
                self._drop_listbox.selection_set(i)
                self._append_log(f'Selected drop target: {self._drop_listbox.get(i)}')
                return
        self._append_log(f'Drop target {name_hint!r} not found in list', 'warn')




    def _cmd_mtc_pick_place(self):
        """MTC pick-and-place spike. Runs full cycle for hardcoded target pair
        (green_2x3_1 -> drop_0) via the so_arm101_mtc C++ node.
        Requires control.launch.py mtc:=true. See /home/aaugus11/.claude/plans/
        abstract-swinging-sundae.md for the hypothesis being validated.
        """
        self._ensure_lego_state()
        if self._attached_lego_name:
            self._append_log(
                f'MTC refusing to run: lego "{self._attached_lego_name}" is already attached. '
                'Call /drop_release first.', 'warn')
            return
        # Refresh scene so MTC sees current lego + cup collision objects.
        try:
            self._add_lego_collision_objects()
        except Exception as exc:
            self._append_log(f'MTC pre-step (lego scene refresh) errored: {exc}', 'warn')
        try:
            self._add_cup_collision_objects()
        except Exception as exc:
            self._append_log(f'MTC pre-step (cup scene refresh) errored: {exc}', 'warn')

        client = getattr(self, '_mtc_run_client', None)
        if client is None:
            self._append_log('MTC client not initialised (launch with mtc:=true).', 'err')
            return
        if not client.wait_for_service(timeout_sec=2.0):
            self._append_log(
                'MTC service /so_arm101_mtc/run unavailable — is the node running '
                '(launch with mtc:=true)?', 'err')
            return

        self._append_log('MTC: calling /so_arm101_mtc/run ...')
        future = client.call_async(Trigger.Request())
        deadline = time.time() + 60.0
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)
        if not future.done():
            self._append_log('MTC: timeout waiting for /so_arm101_mtc/run', 'err')
            return
        try:
            resp = future.result()
        except Exception as exc:
            self._append_log(f'MTC call failed: {exc}', 'err')
            return
        level = 'info' if resp.success else 'err'
        self._append_log(f'MTC result: success={resp.success} msg="{resp.message}"', level)

    def _cmd_grasp_select(self):
        """Select an object in the listbox by name (via ik_target param) or first item.
        Usage: ros2 param set ... ik_target "green_1" then call this service.
        """
        if not hasattr(self, 'obj_listbox') or self.obj_listbox.size() == 0:
            self._append_log('No objects to select', 'warn')
            return
        # Check if a name was specified via the ik_target parameter
        name_hint = self.get_parameter('ik_target').get_parameter_value().string_value.strip()
        target_idx = 0
        if name_hint and '=' not in name_hint:
            for i in range(self.obj_listbox.size()):
                if self.obj_listbox.get(i).split('  ')[0] == name_hint:
                    target_idx = i
                    break
        self.obj_listbox.selection_clear(0, tk.END)
        self.obj_listbox.selection_set(target_idx)
        text = self.obj_listbox.get(target_idx)
        self._append_log(f'Selected: {text.split("  ")[0]}')


    def _get_selected_object_name(self):
        """Return the object name selected in the grasp listbox, or None."""
        sel = self.obj_listbox.curselection()
        if not sel:
            return None
        return self.obj_listbox.get(sel[0]).split('  ')[0]







    # ------------------------------------------------------------------
    # Tab 3: Gripper Control
    # ------------------------------------------------------------------

    def _cmd_gripper_close(self):
        self._gripper_command(JOINT_LIMITS['gripper_joint'][0], execute=True)

    def _cmd_gripper_open(self):
        """Open gripper to JOINT_LIMITS max. Mirrors `_cmd_gripper_open_for_object`'s
        motion-event pattern (background thread, blocking goal, evt.set()) so
        callers can wait on completion. After the motion lands, switches the
        planning group back to 'arm' so RViz's MotionPlanning panel shows the
        orange arm goal-state ghost on the next arm command instead of leaving
        it stuck on the gripper group."""
        open_angle = JOINT_LIMITS['gripper_joint'][1]
        duration = 1.0
        self._append_log(f'Open Gripper: {math.degrees(open_angle):.1f}°')
        self._gripper_command(open_angle, execute=False, duration_s=duration)
        evt = threading.Event()
        self._motion_event = evt
        def _send():
            self._send_gripper_goal(open_angle, duration_s=duration, blocking=True)
            evt.set()
            self._select_planning_group('arm')
        threading.Thread(target=_send, daemon=True).start()

    def _cmd_gripper_open_range(self):
        """Open gripper to the range spinbox upper value (grasp tab)."""
        angle = math.radians(self._grasp_grip_open_var.get())
        self._gripper_command(angle, execute=True,
                              duration_s=self._grasp_grip_duration_var.get())

    def _cmd_gripper_close_range(self):
        """Close gripper to the range spinbox lower value (grasp tab)."""
        angle = math.radians(self._grasp_grip_close_var.get())
        self._gripper_command(angle, execute=True,
                              duration_s=self._grasp_grip_duration_var.get())

    def _cmd_set_jaw_open_clearance(self):
        """Set jaw open clearance: ros2 param set ... jaw_open_clearance_mm 5.0"""
        val = self.get_parameter('jaw_open_clearance_mm').get_parameter_value().double_value
        if hasattr(self, '_jaw_open_clearance_var'):
            self._jaw_open_clearance_var.set(val)
        self._append_log(f'Jaw open clearance set to {val:.1f}mm')

    def _cmd_set_jaw_close_clearance(self):
        """Set jaw close clearance: ros2 param set ... jaw_close_clearance_mm 0.0
        +ve = more gap, -ve = tighter"""
        val = self.get_parameter('jaw_close_clearance_mm').get_parameter_value().double_value
        if hasattr(self, '_jaw_close_clearance_var'):
            self._jaw_close_clearance_var.set(val)
        self._append_log(f'Jaw close clearance set to {val:+.1f}mm')

    def _cmd_set_tcp_clearance(self):
        """Set TCP IK clearance from param: ros2 param set ... tcp_clearance_mm 1.0"""
        val = self.get_parameter('tcp_clearance_mm').get_parameter_value().double_value
        if hasattr(self, '_tcp_clearance_var'):
            self._tcp_clearance_var.set(val)
        self._append_log(f'TCP clearance set to {val:.1f}mm')

    def _cmd_check_grasp_reachable(self):
        """Check if the selected object is within the top-down grasp workspace."""
        obj_name = self._get_selected_object_name()
        if not obj_name:
            self._append_log('No object selected for reachability check', 'warn')
            return
        with self.objects_lock:
            pos = self.objects_data.get(obj_name)
        if pos is None:
            self._append_log(f'Object "{obj_name}" not found', 'warn')
            return
        obj_z_override = self._grasp_obj_z_var.get()
        z = obj_z_override if abs(obj_z_override) > 1e-4 else pos['z']
        ground_z = self._ground_z_var.get() if hasattr(self, '_ground_z_var') else None
        ok, reason = check_grasp_reachable(pos['x'], pos['y'], z, ground_z=ground_z)
        r = math.sqrt(pos['x']**2 + pos['y']**2)
        if ok:
            self._append_log(
                f'Grasp reachable: "{obj_name}" r={r:.3f}m z={z:.3f}m '
                f'[R: {GRASP_WORKSPACE_BOUNDS["R_MIN"]:.3f}-'
                f'{GRASP_WORKSPACE_BOUNDS["R_MAX"]:.3f}m, '
                f'Z: {GRASP_WORKSPACE_BOUNDS["Z_MIN"]:.3f}-'
                f'{GRASP_WORKSPACE_BOUNDS["Z_MAX"]:.3f}m]')
        else:
            self._append_log(
                f'Grasp unreachable: "{obj_name}" — {reason}', 'warn')


    # ------------------------------------------------------------------
    # Trajectory execution (arm joints via action interface)
    # ------------------------------------------------------------------

    def _execute_trajectory(self, target_positions, duration_s=2.0, on_complete=None,
                            blocking=False):
        """Send trajectory to arm_controller via action interface.
        Source: trajectory logic adapted from JETANK_description/jetank_control_gui.py

        If blocking=True, sends with blocking and waits for both the
        controller result and the UI animation to finish before returning.
        Must be called from a background thread when blocking.
        """
        if not self._traj_lock.acquire(blocking=False):
            self._append_log('Trajectory already in progress', 'warn')
            return

        self._slider_driven = True

        # Send the full trajectory via action (controller handles interpolation)
        self._send_arm_goal(target_positions, duration_s=duration_s, blocking=blocking)

        # Animate the UI sliders to show progress
        with self.joint_lock:
            start = {n: self.joint_positions[n] for n in ARM_JOINT_NAMES}

        def _run():
            try:
                steps = int(duration_s * 50)
                for i in range(steps + 1):
                    if not self.running:
                        break
                    t = i / max(steps, 1)
                    t = t * t * (3.0 - 2.0 * t)

                    positions = {}
                    for name in ARM_JOINT_NAMES:
                        s = start.get(name, 0.0)
                        e = target_positions.get(name, s)
                        lo, hi = JOINT_LIMITS[name]
                        positions[name] = max(lo, min(hi, s + (e - s) * t))

                    with self.joint_lock:
                        self.joint_positions.update(positions)

                    if getattr(self, '_gui_ready', False):
                        self.root.after(0, self._sync_arm_sliders, dict(positions))

                    self._publish_goal_state()

                    if self.use_real_hardware:
                        self._send_hw_command(positions)

                    time.sleep(1.0 / 50)

                self._append_log('Trajectory complete')
                self._slider_driven = False
            finally:
                self._traj_lock.release()
            if on_complete is not None:
                on_complete()

        threading.Thread(target=_run, daemon=True).start()

    def _sync_arm_sliders(self, positions):
        for name, val in positions.items():
            if name in self.sliders:
                self.sliders[name].set(val)
                self.slider_labels[name].config(text=f'{val:.3f}')

    # ------------------------------------------------------------------
    # Hardware mode
    # ------------------------------------------------------------------

    def _toggle_hardware(self):
        use_real = self.hw_var.get() == 'real'
        with self.hw_lock:
            if use_real == self.use_real_hardware:
                return
            self.use_real_hardware = use_real

        if use_real:
            self._append_log('Switching to REAL hardware mode', 'warn')
            self.status_var.set('Mode: Real Hardware')
            self.set_joints_btn.config(state=tk.DISABLED)
        else:
            self._append_log('Switching to SIMULATION mode')
            self.status_var.set('Mode: Simulation')
            self.set_joints_btn.config(state=tk.NORMAL)

        # Update grasp topic to match hardware mode
        if hasattr(self, '_grasp_topic_var'):
            new_topic = '/objects_poses_real' if use_real else '/objects_poses_sim'
            self._grasp_topic_var.set(new_topic)
            self._cmd_grasp_update_topic()


    def _real_js_callback(self, msg):
        if not self.use_real_hardware:
            return
        with self.joint_lock:
            for i, name in enumerate(msg.name):
                if name in self.joint_positions and i < len(msg.position):
                    self.joint_positions[name] = msg.position[i]
            positions = dict(self.joint_positions)

        if getattr(self, '_gui_ready', False):
            self.root.after(0, self._sync_all_sliders, positions)

    def _sync_all_sliders(self, positions):
        for name, val in positions.items():
            if name in self.sliders:
                self.sliders[name].set(val)
                if name in self.slider_labels:
                    self.slider_labels[name].config(text=f'{val:.3f}')

    def _send_hw_command(self, positions):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(positions.keys())
        msg.position = list(positions.values())
        self.joint_cmd_pub.publish(msg)

    def _ext_cmd_callback(self, msg):
        """Handle external joint commands — update sliders and send to controllers."""
        self._slider_driven = True
        with self.joint_lock:
            for i, name in enumerate(msg.name):
                if name in self.joint_positions and i < len(msg.position):
                    lo, hi = JOINT_LIMITS.get(name, (-math.pi, math.pi))
                    self.joint_positions[name] = max(lo, min(hi, msg.position[i]))
            positions = dict(self.joint_positions)
        # External commands directly move the robot
        self._send_arm_goal(positions, duration_s=0.2)
        jaw = positions.get(GRIPPER_JOINT_NAME, 0.0)
        self._send_gripper_goal(jaw, duration_s=0.2)
        self._publish_goal_state()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _on_close(self):
        self.running = False
        self.root.quit()
        self.root.destroy()

    def destroy_node(self):
        self.running = False
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args, signal_handler_options=rclpy.SignalHandlerOptions.NO)
    node = SOArm101ControlGUI()
    # Explicit thread count — default is multiprocessing.cpu_count() which can
    # leave too few free threads when service handlers block on .wait()
    # (Phase 9 sync-refresh pattern). 4 threads is ample headroom.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    # Tracks whether a prior signal already armed the force-exit timer, so
    # a repeated SIGTERM doesn't double-arm and so we can distinguish
    # "handler never fired" (cosmetic script race) from "handler fired but
    # Tk deadlocked" (real bug — force-exit wins).
    _shutdown_state = {'armed': False}

    def _shutdown_handler(signum, frame):
        node.running = False
        # Tell tkinter to quit from its own event loop. root.after(0, ...)
        # is the only thread-safe way to drive the Tk interpreter from a
        # signal handler (which runs on the main thread, not the Tk thread).
        if hasattr(node, 'root'):
            try:
                node.root.after(0, node._on_close)
            except Exception:
                pass

        # BELT-AND-SUSPENDERS: historically the restart script would send
        # SIGTERM, this handler would fire, Tk's event loop would NOT pick
        # up the after() callback in time (blocked on an X server round-trip
        # or a slow widget repaint), the script would time out, warn "X11
        # processes still alive (NOT force-killing)," and the next launch's
        # control_gui would SIGSEGV trying to take over the occupied display
        # + duplicate ROS2 node name. We were leaking processes on every
        # restart and producing the "RViz without GUI / GUI without RViz"
        # failure modes. This timer forces the process to die within ~2.5s
        # of the first signal even if Tk is wedged — the restart script
        # can then trust that SIGTERM actually killed the process.
        if _shutdown_state['armed']:
            return
        _shutdown_state['armed'] = True

        def _force_exit():
            # Give the Tk path ~2.5s to land cleanly. After that, os._exit
            # bypasses atexit/destructors but that's ACCEPTABLE here: the
            # process is already being torn down by SIGTERM and we own all
            # its external state (no uncommitted files, sockets handed off
            # to the OS). Clean-exit would be nicer but deadlocked-forever
            # is strictly worse.
            import os
            os._exit(0)

        import threading as _t
        # daemon: a clean exit must not wait for this guard. As a non-daemon
        # thread it kept every process alive for the full 2.5 s and then
        # os._exit()-ed it, even when the Tk + rclpy shutdown had finished.
        _guard = _t.Timer(2.5, _force_exit)
        _guard.daemon = True
        _guard.start()

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    # Spin ROS2 executor in background thread
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # Run tkinter on main thread (required by macOS Cocoa, safe on Linux)
    try:
        node._run_gui()
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        executor.shutdown()
        # Let the spin thread leave rcl before teardown: a daemon thread still
        # inside the executor's wait set when the interpreter finalizes aborts
        # the process ("terminate called without an active exception").
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
