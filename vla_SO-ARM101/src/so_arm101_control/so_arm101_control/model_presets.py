"""Model presets shared by control_gui and deploy_sim.sh.

Add a preset in one row: key: (HF id or local path, checkpoint, rename-map JSON).
Local paths may start with ~; point at the directory containing config.json.
"""
import os
import sys

MODEL_PRESETS = {
    "sim": ("anirudhrani/smolvla_sim_100ep_fft__10ksteps_h200", "", "{}"),
    "phase3-10k": ("/mnt/a4500-backup/exploring-vlas/eval/checkpoints/smolvla-phase3-normal-60k-s42-20261114/checkpoints/010000/pretrained_model", "", "{}"),
    "phase3-20k": ("/mnt/a4500-backup/exploring-vlas/eval/checkpoints/smolvla-phase3-normal-60k-s42-20261114/checkpoints/020000/pretrained_model", "", "{}"),
    "phase3-30k": ("/mnt/a4500-backup/exploring-vlas/eval/checkpoints/smolvla-phase3-normal-60k-s42-20261114/checkpoints/030000/pretrained_model", "", "{}"),
    "unfreeze30k+2500": ("/mnt/a4500-backup/exploring-vlas/eval/checkpoints/unfreeze30k/002500/pretrained_model", "", "{}"),
    "unfreeze30k+5000": ("/mnt/a4500-backup/exploring-vlas/eval/checkpoints/unfreeze30k/005000/pretrained_model", "", "{}"),
    "unfreeze30k+7500": ("/mnt/a4500-backup/exploring-vlas/eval/checkpoints/unfreeze30k/007500/pretrained_model", "", "{}"),
    "unfreeze30k+10000": ("/mnt/a4500-backup/exploring-vlas/eval/checkpoints/unfreeze30k/010000/pretrained_model", "", "{}"),
    "phase3-40k": ("/mnt/a4500-backup/exploring-vlas/eval/checkpoints/smolvla-phase3-normal-60k-s42-20261114/checkpoints/040000/pretrained_model", "", "{}"),
    "phase3-60k": ("/mnt/a4500-backup/exploring-vlas/eval/checkpoints/smolvla-phase3-normal-60k-s42-20261114/checkpoints/060000/pretrained_model", "", "{}"),
    # Crop checkpoint requires its bundled processor module on server PYTHONPATH; the native evaluator sets it.
    "next120-primary-10k": ("/mnt/a4500-backup/exploring-vlas/eval/checkpoints/next120-primary-init/010000/pretrained_model", "", "{}"),
    "next120-primary-20k": ("/mnt/a4500-backup/exploring-vlas/eval/checkpoints/next120-primary-init/020000/pretrained_model", "", "{}"),
    "next120-primary-30k": ("/mnt/a4500-backup/exploring-vlas/eval/checkpoints/next120-primary-init/030000/pretrained_model", "", "{}"),
    "next120-primary-40k": ("/mnt/a4500-backup/exploring-vlas/eval/checkpoints/next120-primary-init/040000/pretrained_model", "", "{}"),
    "robust-crop-5k": ("/mnt/a4500-backup/exploring-vlas/eval/checkpoints/recorder-crf18-crop/005000/pretrained_model", "", "{}"),
    "robust-crop-10k": ("/mnt/a4500-backup/exploring-vlas/eval/checkpoints/recorder-crf18-crop/010000/pretrained_model", "", "{}"),
    "robust-crop-15k": ("/mnt/a4500-backup/exploring-vlas/eval/checkpoints/recorder-crf18-crop/015000/pretrained_model", "", "{}"),
    "robust-crop-20k": ("/mnt/a4500-backup/exploring-vlas/eval/checkpoints/recorder-crf18-crop/020000/pretrained_model", "", "{}"),
    "robust-crop-25k": ("/mnt/a4500-backup/exploring-vlas/eval/checkpoints/recorder-crf18-crop/025000/pretrained_model", "", "{}"),
    "robust-crop-30k": ("/mnt/a4500-backup/exploring-vlas/eval/checkpoints/recorder-crf18-crop/030000/pretrained_model", "", "{}"),
    "memorise24-crop-end": ("/mnt/a4500-backup/exploring-vlas/eval/checkpoints/memorise24/workspace_crop/009102/pretrained_model", "", "{}"),
    "memorise24-original-end": ("/mnt/a4500-backup/exploring-vlas/eval/checkpoints/memorise24/original/009102/pretrained_model", "", "{}"),
    "base-ft": ("/home/aaugus11/w_ckpt/smolvla-base-ft-rec222812-20k-s42-20261113", "", "{}"),
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
