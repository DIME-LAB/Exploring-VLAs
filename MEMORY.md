# Project Memory — Exploring-VLAs

## Workflow preferences

- **Always restart the stack when a procedure requires it.** Don't ask "want me to restart?" — just run `bash mac-env/scripts/stack_stop.sh && bash mac-env/scripts/stack_start.sh --mode <mode> --rviz follower_port:=...` and wait for `/tmp/soarm_stack.log` to show ready signals. Reason: user runs `--dangerously-skip-permissions` and wants next-step actions taken without confirmation.
- **Real-mode follower port** is currently `/dev/tty.usbmodem5A680088871`. Pass it explicitly to `stack_start.sh`.
- **K_calib** lives at `~/.cache/so_arm101/gravity_comp/k_calib.json` and was refit 2026-04-27.
