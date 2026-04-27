#!/usr/bin/env python3
"""Per-joint Homing_Offset (register 31) trim for the SO-ARM101 follower.

When lerobot-calibrate gets a joint *almost* right but visually drifts
1-2 ticks, re-running the full wizard is overkill. This script bumps
Homing_Offset by an integer delta per joint and (optionally) updates
lerobot's calibration JSON so the change persists across power cycles.

  1 tick = 0.088° on the Feetech 4096-tick encoder.
  Recommended delta range: ±50 ticks (~±4.4°). Larger deltas mean the
  lerobot-calibrate sweep limits are wrong — re-run the wizard instead.

Procedure (mirrors arm_reset.py — same kill+respawn dance because the
Feetech bus is half-duplex; see jsr's torque_reset service in a future
commit for the in-process alternative):

  1. Kill jsr_write   → release bus
  2. Open serial      → 1 Mbaud
  3. Torque OFF       → safe EEPROM-mutation state
  4. Per joint:
       - read Homing_Offset (reg 31, signed int16)
       - new = current + delta
       - write back
  5. Anti-jolt        → Goal_Position = Present_Position per servo
  6. Torque ON        → re-engage with zero PID error
  7. (optional) persist → mutate lerobot JSON to match
  8. (optional) restart_jsr → resume /joint_states

Usage:
  arm_trim.py --shoulder-lift -3 --elbow-flex +2 --persist --restart-jsr
  arm_trim.py --joint shoulder_pan --delta -1   # (alt syntax)
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import time

import serial


PORT = '/dev/tty.usbmodem5A680088871'
JSR_WRAPPER = '/tmp/run_jsr_write.sh'

JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
               'wrist_flex', 'wrist_roll', 'gripper']

LEROBOT_CALIB_DIR = os.path.expanduser(
    '~/.cache/huggingface/lerobot/calibration/robots/so101_follower/')

REG_HOMING_OFFSET = 31
REG_TORQUE_ENABLE = 40
REG_GOAL_POSITION = 42
REG_PRESENT_POSITION = 56

DELTA_LIMIT = 50  # tick magnitude per call; refuse larger values.


# --- Feetech bus primitives (mirrors arm_reset.py) ---

def _chk(p):
    return (~sum(p)) & 0xFF


def write_byte(ser, sid, addr, value):
    payload = [sid, 4, 0x03, addr, value & 0xFF]
    pkt = bytes([0xFF, 0xFF] + payload + [_chk(payload)])
    ser.reset_input_buffer()
    ser.write(pkt)
    time.sleep(0.005)
    r = ser.read(6)
    return len(r) >= 6 and r[2] == sid and r[4] == 0


def write_word(ser, sid, addr, value):
    """Write 2-byte little-endian. Negative values get masked to uint16."""
    v = value & 0xFFFF
    lo = v & 0xFF
    hi = (v >> 8) & 0xFF
    payload = [sid, 5, 0x03, addr, lo, hi]
    pkt = bytes([0xFF, 0xFF] + payload + [_chk(payload)])
    ser.reset_input_buffer()
    ser.write(pkt)
    time.sleep(0.005)
    r = ser.read(6)
    return len(r) >= 6 and r[2] == sid and r[4] == 0


def read_word(ser, sid, addr):
    payload = [sid, 4, 0x02, addr, 2]
    pkt = bytes([0xFF, 0xFF] + payload + [_chk(payload)])
    ser.reset_input_buffer()
    ser.write(pkt)
    time.sleep(0.003)
    r = ser.read(8)
    if len(r) < 8 or r[0] != 0xFF or r[1] != 0xFF or r[2] != sid or r[4] != 0:
        return None
    return struct.unpack('<H', r[5:7])[0]


def read_signed_word(ser, sid, addr):
    raw = read_word(ser, sid, addr)
    if raw is None:
        return None
    return raw - 65536 if raw >= 32768 else raw


# --- Process management ---

def kill_jsr():
    subprocess.run(['pkill', '-9', '-f', 'run_jsr_write.sh'],
                   check=False, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    subprocess.run(['pkill', '-9', '-f', 'joint_state_reader'],
                   check=False, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    time.sleep(1.5)


def start_jsr():
    if not os.path.exists(JSR_WRAPPER):
        raise RuntimeError(f"missing wrapper: {JSR_WRAPPER}")
    log = open('/tmp/jsr_write.log', 'a')
    subprocess.Popen(
        ['/bin/bash', JSR_WRAPPER],
        stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True, close_fds=True,
        preexec_fn=os.setpgrp,
    )
    time.sleep(4.0)


# --- lerobot JSON helpers ---

def find_lerobot_json():
    """Return the lerobot calibration JSON path (most-recent .json in
    so101_follower/), or None if not found."""
    if not os.path.isdir(LEROBOT_CALIB_DIR):
        return None
    json_files = [f for f in os.listdir(LEROBOT_CALIB_DIR)
                  if f.endswith('.json')]
    if not json_files:
        return None
    return os.path.join(LEROBOT_CALIB_DIR, sorted(json_files)[-1])


def update_lerobot_json(path, deltas, verbose=True):
    """Add `delta` to each joint's homing_offset in the lerobot JSON.

    Mutates in place; backs up to <path>.bak first. Skips joints not
    present in the JSON (warns) so a partial config doesn't get a NULL
    field added.
    """
    with open(path) as f:
        cal = json.load(f)
    backup = path + '.bak'
    with open(backup, 'w') as f:
        json.dump(cal, f, indent=2)
    if verbose:
        print(f"[trim] backup → {backup}")

    for joint, delta in deltas.items():
        if delta == 0:
            continue
        if joint not in cal:
            print(f"  WARN: '{joint}' missing from lerobot JSON; skipping",
                  file=sys.stderr)
            continue
        old = int(cal[joint].get('homing_offset', 0))
        cal[joint]['homing_offset'] = old + delta
        if verbose:
            print(f"  {joint}: homing_offset {old} → {old + delta} "
                  f"(Δ{delta:+d})")

    with open(path, 'w') as f:
        json.dump(cal, f, indent=2)
    if verbose:
        print(f"[trim] wrote → {path}")


# --- Main trim procedure ---

def trim_arm(deltas, port=PORT, persist=False, restart_jsr=False,
             verbose=True):
    """Apply per-joint Homing_Offset deltas. `deltas` is a dict
    {joint_name: int_ticks} — only joints with non-zero values matter."""
    nonzero = {j: d for j, d in deltas.items() if d != 0}
    if not nonzero:
        if verbose:
            print("[trim] all deltas are zero — nothing to do")
        return 0

    for j, d in nonzero.items():
        if abs(d) > DELTA_LIMIT:
            print(f"ERROR: {j} delta={d} exceeds ±{DELTA_LIMIT} ticks. "
                  f"Re-run lerobot-calibrate instead.", file=sys.stderr)
            return 2

    if verbose:
        print(f"[trim] applying deltas: {nonzero}")
        print("[trim] kill jsr_write → release bus")
    kill_jsr()

    ser = serial.Serial(port, 1000000, timeout=0.1)
    ser.reset_input_buffer()
    time.sleep(0.05)

    failed_joints = []
    try:
        # Torque OFF before EEPROM mutation. The servo's behavior at
        # torque=on while changing Homing_Offset is undefined per Feetech's
        # protocol notes — safer to gate the write.
        if verbose:
            print("[trim] Torque_Enable=0  (safe EEPROM-mutation state)")
        for sid in range(1, 7):
            write_byte(ser, sid, REG_TORQUE_ENABLE, 0)
            time.sleep(0.003)
        time.sleep(0.5)  # let arm settle under gravity

        # Apply deltas
        for joint, delta in nonzero.items():
            sid = JOINT_NAMES.index(joint) + 1
            current = read_signed_word(ser, sid, REG_HOMING_OFFSET)
            if current is None:
                print(f"  WARN: {joint} read Homing_Offset failed",
                      file=sys.stderr)
                failed_joints.append(joint)
                continue
            new_val = current + delta
            ok = write_word(ser, sid, REG_HOMING_OFFSET, new_val)
            verify = read_signed_word(ser, sid, REG_HOMING_OFFSET)
            if not ok or verify != new_val:
                print(f"  WARN: {joint} write Homing_Offset {current}→"
                      f"{new_val} failed (verify={verify})",
                      file=sys.stderr)
                failed_joints.append(joint)
                continue
            if verbose:
                print(f"  {joint}: Homing_Offset {current} → {new_val} "
                      f"(Δ{delta:+d})")
            time.sleep(0.005)

        # Anti-jolt before torque re-enable
        if verbose:
            print("[trim] anti-jolt: Goal_Position ← Present_Position")
        for sid in range(1, 7):
            present = read_word(ser, sid, REG_PRESENT_POSITION)
            if present is not None:
                write_word(ser, sid, REG_GOAL_POSITION, present)
            time.sleep(0.005)

        # Torque ON
        if verbose:
            print("[trim] Torque_Enable=1  (re-engage)")
        for sid in range(1, 7):
            write_byte(ser, sid, REG_TORQUE_ENABLE, 1)
            time.sleep(0.003)
    finally:
        ser.close()

    # Persist to lerobot JSON
    if persist and not failed_joints:
        json_path = find_lerobot_json()
        if json_path is None:
            print(f"WARN: no lerobot calibration JSON in "
                  f"{LEROBOT_CALIB_DIR} — EEPROM updated, JSON drift",
                  file=sys.stderr)
        else:
            update_lerobot_json(json_path, nonzero, verbose=verbose)

    if restart_jsr:
        if verbose:
            print("[trim] restart jsr_write → /joint_states resumes")
        start_jsr()

    if failed_joints:
        print(f"FAILED joints (EEPROM not updated): {failed_joints}",
              file=sys.stderr)
        return 1

    if verbose:
        print("[trim] done")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', default=PORT)
    for j in JOINT_NAMES:
        ap.add_argument(f'--{j.replace("_", "-")}', type=int, default=0,
                        metavar='N',
                        help=f'{j} Homing_Offset delta in ticks '
                             f'(±{DELTA_LIMIT} max). 1 tick = 0.088°.')
    ap.add_argument('--persist', action='store_true',
                    help='Mutate lerobot JSON to match EEPROM (recommended).')
    ap.add_argument('--restart-jsr', action='store_true',
                    help='Restart jsr_write after the trim. Default false '
                         'so callers can chain multiple operations.')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    deltas = {j: getattr(args, j.replace('-', '_')) for j in JOINT_NAMES}
    rc = trim_arm(deltas, port=args.port,
                  persist=args.persist,
                  restart_jsr=args.restart_jsr,
                  verbose=not args.quiet)
    sys.exit(rc)


if __name__ == '__main__':
    main()
