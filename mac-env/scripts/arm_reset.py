#!/usr/bin/env python3
"""Reliable arm reset / vibration-stop primitive for SO-ARM101 follower.

Procedure (single bus-access window, ~3 sec total downtime on /joint_states):
  1. Kill jsr_write to release the half-duplex Feetech bus.
  2. Per servo:
       - Read Present_Position
       - Write Goal_Position = Present_Position   (anti-jolt for re-enable)
  3. Per servo: Torque_Enable = 0   (motor goes limp, vibration stops in <50ms)
  4. Sleep 300 ms (let any residual inertia decay).
  5. Per servo: optionally write new PID gains (--set-lerobot-defaults flag).
  6. Per servo: Torque_Enable = 1   (re-engage; zero PID error → no startup motion).
  7. Restart jsr_write so /joint_states resumes flowing.

Use this between every test iteration so vibration state never carries forward.

Feetech STS3215 register map (relevant subset):
  21  P_Coefficient        (1 byte)
  22  D_Coefficient        (1 byte)
  23  I_Coefficient        (1 byte)
  40  Torque_Enable        (1 byte: 0=off, 1=on)
  42  Goal_Position        (2 bytes, little-endian)
  56  Present_Position     (2 bytes, little-endian)

References:
  matthieuvigne/STS_servos register map
  lerobot/src/lerobot/robots/so_follower/so_follower.py:160-164 (P=16 default)
"""
import argparse
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

LEROBOT_P = 16
LEROBOT_D = 32
LEROBOT_I = 0


# --- Feetech bus protocol primitives ---

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
    """Write 2-byte little-endian word."""
    lo = value & 0xFF
    hi = (value >> 8) & 0xFF
    payload = [sid, 5, 0x03, addr, lo, hi]
    pkt = bytes([0xFF, 0xFF] + payload + [_chk(payload)])
    ser.reset_input_buffer()
    ser.write(pkt)
    time.sleep(0.005)
    r = ser.read(6)
    return len(r) >= 6 and r[2] == sid and r[4] == 0


def read_word(ser, sid, addr):
    """Read 2-byte little-endian word."""
    payload = [sid, 4, 0x02, addr, 2]
    pkt = bytes([0xFF, 0xFF] + payload + [_chk(payload)])
    ser.reset_input_buffer()
    ser.write(pkt)
    time.sleep(0.003)
    r = ser.read(8)
    if len(r) < 8 or r[0] != 0xFF or r[1] != 0xFF or r[2] != sid or r[4] != 0:
        return None
    return struct.unpack('<H', r[5:7])[0]


def read_byte(ser, sid, addr):
    payload = [sid, 4, 0x02, addr, 1]
    pkt = bytes([0xFF, 0xFF] + payload + [_chk(payload)])
    ser.reset_input_buffer()
    ser.write(pkt)
    time.sleep(0.002)
    r = ser.read(7)
    if len(r) < 7 or r[0] != 0xFF or r[1] != 0xFF or r[2] != sid or r[4] != 0:
        return None
    return r[5]


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
    proc = subprocess.Popen(
        ['/bin/bash', JSR_WRAPPER],
        stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True, close_fds=True,
        # Detach from parent to survive parent exit
        preexec_fn=os.setpgrp,
    )
    time.sleep(4.0)
    return proc


# --- The main reset procedure ---

def reset_arm(port=PORT, set_lerobot_defaults=False, verbose=True):
    if verbose:
        print(f"[reset] kill jsr_write → release bus", flush=True)
    kill_jsr()

    ser = serial.Serial(port, 1000000, timeout=0.1)
    ser.reset_input_buffer()
    time.sleep(0.05)

    try:
        # Step 1: torque off FIRST — kills any active oscillation immediately
        # and lets gravity park the arm at a static pose.
        if verbose:
            print("[reset] Torque_Enable=0  (vibration stops here, "
                  "arm settles under gravity)", flush=True)
        for sid in range(1, 7):
            write_byte(ser, sid, 40, 0)
            time.sleep(0.003)

        # Step 2: wait for arm to fully settle under gravity.
        # 700 ms is enough for SO-ARM101 worst-case (shoulder_lift fall from
        # extended → fully drooped) without being annoyingly long.
        time.sleep(0.7)

        # Step 3: optionally set PID gains while torque is off
        # (writing register 21/22/23 is safe regardless of torque state).
        if set_lerobot_defaults:
            if verbose:
                print(f"[reset] Setting lerobot defaults: "
                      f"P={LEROBOT_P} D={LEROBOT_D} I={LEROBOT_I}", flush=True)
            for sid in range(1, 7):
                p_ok = write_byte(ser, sid, 21, LEROBOT_P)
                d_ok = write_byte(ser, sid, 22, LEROBOT_D)
                i_ok = write_byte(ser, sid, 23, LEROBOT_I)
                if not (p_ok and d_ok and i_ok):
                    print(f"  WARN: servo {sid} PID write partial failure "
                          f"(p={p_ok} d={d_ok} i={i_ok})", file=sys.stderr)
                time.sleep(0.005)

        # Step 4: NOW re-read Present_Position (reflects the settled pose)
        # and write Goal_Position = Present so re-enabling torque produces
        # ZERO PID error → ZERO startup motion.
        if verbose:
            print("[reset] anti-jolt: Goal_Position ← settled Present_Position",
                  flush=True)
        for sid in range(1, 7):
            present = read_word(ser, sid, 56)
            if present is None:
                print(f"  WARN: servo {sid} read_word(present_position) failed",
                      file=sys.stderr)
                continue
            ok = write_word(ser, sid, 42, present)
            if not ok:
                print(f"  WARN: servo {sid} write_word(goal_position={present}) failed",
                      file=sys.stderr)
            time.sleep(0.005)

        # Step 5: re-enable torque (zero error → arm holds where it settled)
        if verbose:
            print("[reset] Torque_Enable=1  (re-engage holding settled pose)",
                  flush=True)
        for sid in range(1, 7):
            write_byte(ser, sid, 40, 1)
            time.sleep(0.003)

        # Step 6: verify the final state
        if verbose:
            print("[reset] verify final state:")
            print("  | ID | Joint           | P  | D  | I  | Torque | Goal | Present | Δ° |")
            for sid in range(1, 7):
                p = read_byte(ser, sid, 21)
                d = read_byte(ser, sid, 22)
                i = read_byte(ser, sid, 23)
                tq = read_byte(ser, sid, 40)
                goal = read_word(ser, sid, 42)
                pres = read_word(ser, sid, 56)
                if None in (p, d, i, tq, goal, pres):
                    print(f"  | {sid}  | {JOINT_NAMES[sid-1]:<15} | -- read fail |")
                    continue
                delta = (pres - goal) / 4096.0 * 360.0
                print(f"  | {sid}  | {JOINT_NAMES[sid-1]:<15} | "
                      f"{p:<2} | {d:<2} | {i:<2} | {tq:<6} | "
                      f"{goal:<4} | {pres:<7} | {delta:+.2f} |")
                time.sleep(0.005)
    finally:
        ser.close()

    if verbose:
        print("[reset] restart jsr_write → /joint_states resumes", flush=True)
    start_jsr()
    if verbose:
        print("[reset] done", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port', default=PORT)
    ap.add_argument('--set-lerobot-defaults', action='store_true',
                    help='Also write P=16, D=32, I=0 (lerobot defaults)')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()
    reset_arm(port=args.port,
              set_lerobot_defaults=args.set_lerobot_defaults,
              verbose=not args.quiet)


if __name__ == '__main__':
    main()
