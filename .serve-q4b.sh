#!/bin/bash
cd /home/aaron/ai/Strata
for p in $(pgrep -f 'server[.]py --engine strata'); do kill -9 $p; done
pkill -9 -x strata 2>/dev/null
sleep 2
rm -f /home/aaron/ai/Strata/strata-q4ks.log /home/aaron/ai/Strata/.serve-q4b.log
export STRATA_SERVE_PROMPT_DIFF=1
exec /home/aaron/miniconda3/envs/strata/bin/python serve/server.py --engine strata --config .serve-q4ks2.json --port 8000 --host 0.0.0.0 > /home/aaron/ai/Strata/.serve-q4b.log 2>&1
