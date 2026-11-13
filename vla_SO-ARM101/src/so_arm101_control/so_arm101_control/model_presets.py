"""Model presets shared by control_gui and deploy_sim.sh.

Add a preset in one row: key: (HF id or local path, checkpoint, rename-map JSON).
Local paths may start with ~; point at the directory containing config.json.
"""
import os
import sys

MODEL_PRESETS = {
    "base-ft": ("/home/aaugus11/w_ckpt/smolvla-base-ft-rec222812-20k-s42-20261113", "", "{}"),
    "sim": ("anirudhrani/smolvla_sim_100ep_fft__10ksteps_h200", "", "{}"),
    "mix-75s25r": ("anirudhrani/smolvla_sim_100ep_75S_25R_fft__10ksteps_h200", "", "{}"),
    "mix-25s75r": ("anirudhrani/smolvla_sim_100ep_25S_75R_fft__10ksteps_h200", "", "{}"),
    "mix-50s50r": ("anirudhrani/smolvla_blue_sort_50sim_50real_10k", "", "{}"),
    "real-full": ("anirudhrani/smolvla_blue_sort_ven_50k", "050000", "{}"),
    "real-lora": ("anirudhrani/smolvla_blue_sort_v2_lora", "050000", '{"wrist": "camera1", "top": "camera2"}'),
}


def resolve_model(value):
    value = value or "sim"
    if value in MODEL_PRESETS:
        path, checkpoint, _ = MODEL_PRESETS[value]
        if checkpoint:
            path = f"{path}/checkpoints/{checkpoint}/pretrained_model"
    elif "/" in value:
        path = value
    else:
        raise ValueError(f"Unknown model preset {value!r}; choose {', '.join(MODEL_PRESETS)}, an HF id or local path")
    return os.path.expanduser(path)


if __name__ == "__main__":
    if sys.argv[1:] == ["--list"]:
        print("\n".join(MODEL_PRESETS))
    else:
        try:
            print(resolve_model(sys.argv[1] if len(sys.argv) > 1 else ""))
        except ValueError as exc:
            sys.exit(str(exc))
