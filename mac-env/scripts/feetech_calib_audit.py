#!/usr/bin/env python3
"""Read each servo's EEPROM Homing_Offset (register 31) and compare to
lerobot's saved calibration JSON. Diagnoses which joint(s) have a
calibration mismatch.

Also reports current Present_Position (register 56) and Goal_Position
(register 42) so we can see where the servo currently is in raw ticks
vs where it thinks it's being commanded to.

Doesn't write anything. Read-only audit.
"""
import json
import os
import struct
import sys
import time
import serial

PORT = '/dev/tty.usbmodem5A680088871'
LEROBOT_JSON = os.path.expanduser(
    '~/.cache/huggingface/lerobot/calibration/robots/so_follower/'
)
JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
               'wrist_flex', 'wrist_roll', 'gripper']


def chk(p):
    return (~sum(p)) & 0xFF


def read_byte(ser, sid, addr):
    payload = [sid, 4, 0x02, addr, 1]
    pkt = bytes([0xFF, 0xFF] + payload + [chk(payload)])
    ser.reset_input_buffer()
    ser.write(pkt)
    time.sleep(0.002)
    r = ser.read(7)
    if len(r) < 7 or r[0] != 0xFF or r[1] != 0xFF or r[2] != sid or r[4] != 0:
        return None
    return r[5]


def read_word(ser, sid, addr):
    payload = [sid, 4, 0x02, addr, 2]
    pkt = bytes([0xFF, 0xFF] + payload + [chk(payload)])
    ser.reset_input_buffer()
    ser.write(pkt)
    time.sleep(0.002)
    r = ser.read(8)
    if len(r) < 8 or r[0] != 0xFF or r[1] != 0xFF or r[2] != sid or r[4] != 0:
        return None
    return struct.unpack('<H', r[5:7])[0]


def read_signed_word(ser, sid, addr):
    """Read a signed Feetech register and decode using the 12-bit sign-
    magnitude convention used by STS3215 firmware for Homing_Offset
    (register 31). lerobot's motors_bus.py applies the same decode
    internally before storing in the calibration JSON, so EEPROM as
    decoded here matches JSON homing_offset 1-to-1.

    Encoding: bits 0-10 = magnitude (0-2047), bit 11 = sign (1=negative).
    Bits 12-15 are reserved/unused on STS3215 STORED values.

    Prior version of this function read raw uint16 and only flipped the
    16-bit two's-complement boundary (>=32768 → negative). That gave
    false MISMATCH alarms for any negative homing — the firmware never
    sets bits 12-15 so values like JSON=-137 came back as raw=2185
    (which the old decoder reported as +2185 instead of -137).
    """
    raw = read_word(ser, sid, addr)
    if raw is None:
        return None
    # Two's-complement int16 fallback (covers other registers that may
    # use the standard convention, e.g. Goal_Velocity).
    if raw >= 32768:
        return raw - 65536
    # 12-bit sign-magnitude (Feetech STS Homing_Offset convention).
    if raw & 0x800:  # bit 11 set → negative
        return -(raw & 0x7FF)
    return raw


def main():
    # Load lerobot calibration JSON
    json_files = [f for f in os.listdir(LEROBOT_JSON) if f.endswith('.json')]
    if not json_files:
        print(f"No JSON in {LEROBOT_JSON}")
        return 1
    json_path = os.path.join(LEROBOT_JSON, sorted(json_files)[-1])
    with open(json_path) as f:
        cal = json.load(f)
    print(f"lerobot JSON: {json_path}\n")

    ser = serial.Serial(PORT, 1000000, timeout=0.1)
    ser.reset_input_buffer()
    time.sleep(0.05)

    print("| ID | Joint           | EEPROM Homing  | JSON Homing | Match? | Goal_Pos | Present_Pos | Δ ticks | Δ° |")
    print("|----|-----------------|----------------|-------------|--------|----------|-------------|---------|------|")
    for sid in range(1, 7):
        joint = JOINT_NAMES[sid - 1]
        # Homing_Offset is 2 bytes at register 31
        eeprom_homing = read_signed_word(ser, sid, 31)
        json_homing = cal.get(joint, {}).get('homing_offset', 'N/A')
        goal = read_word(ser, sid, 42)
        present = read_word(ser, sid, 56)
        delta_ticks = present - goal if (present is not None and goal is not None) else 0
        delta_deg = delta_ticks / 4096.0 * 360.0

        match = '✓' if eeprom_homing == json_homing else 'MISMATCH'
        print(f"| {sid}  | {joint:<15} | "
              f"{eeprom_homing:>14} | "
              f"{json_homing:>11} | "
              f"{match:<6} | "
              f"{goal:>8} | {present:>11} | {delta_ticks:>+7} | {delta_deg:+.2f} |")
        time.sleep(0.005)

    ser.close()


if __name__ == '__main__':
    main()
