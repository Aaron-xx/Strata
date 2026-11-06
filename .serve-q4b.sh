#!/bin/bash
cd /home/aaron/ai/Strata
for p in $(pgrep -f 'server[.]py --engine strata'); do kill -9 $p; done
pkill -9 -x strata 2>/dev/null
sleep 2
rm -f /home/aaron/ai/Strata/strata-q4ks.log /home/aaron/ai/Strata/.serve-q4b.log
# The card sits at its 1455 MHz base clock (laptop GTX 1660 Ti, 80 W cap). Locking the clock to the maximum
# lets the driver boost to ~1905 MHz (it stops at the power wall), measured +16.1% prefill and +6.4% decode
# against the unlocked clock in the same build (2026-11-05).
sudo -n nvidia-smi -pm 1 >/dev/null 2>&1 || true
sudo -n nvidia-smi -lgc 2100,2100 >/dev/null 2>&1 || true
export STRATA_Q36_PREFILL_TIMING=1
export STRATA_SERVE_PROMPT_DIFF=1
exec /home/aaron/miniconda3/envs/strata/bin/python serve/server.py --engine strata --config .serve-q4ks2.json --port 8000 --host 0.0.0.0 > /home/aaron/ai/Strata/.serve-q4b.log 2>&1
