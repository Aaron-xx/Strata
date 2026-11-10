#!/bin/bash
# Release the GPU clock lock that .serve-q4b.sh:10-11 puts in place on every start.
#
# That script runs nvidia-smi -pm 1 (persistence mode) and nvidia-smi -lgc 2100,2100 (locked
# graphics clock). Both are driver-level device state, not process state: when the serving
# process exits they stay applied, the driver reports Idle: Active but cannot drop below the
# locked floor, and the card sits at 1920 MHz / P5 / ~21 W instead of 300 MHz / P8 / ~6.4 W.
#
# On CUDA 13.2 / driver 595.84 -lgc takes the locked-clock path (nvidia-smi -q -d CLOCK reports
# Applications Clocks Setting: Requested functionality has been deprecated), so the lock is
# cleared with -rgc (--reset-gpu-clocks), not by re-issuing -lgc.
#
# Called from ExecStopPost of ~/.config/systemd/user/strata-serve.service, so the lock lifetime
# is exactly the service lifetime. Every command is best effort: the card may be busy, absent,
# or never locked, and a failed release must never turn a clean stop into a failed stop.
sudo -n nvidia-smi -rgc  >/dev/null 2>&1 || true
sudo -n nvidia-smi -pm 0 >/dev/null 2>&1 || true
exit 0
