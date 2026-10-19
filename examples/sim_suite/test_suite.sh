#!/bin/bash -l
# Smoke test: plan into a scratch directory and run a few skies of every task kind, timed.
set -e
ENV=/nvme/h/lchristodoulou/.conda/envs/camb
export LD_LIBRARY_PATH="$ENV/lib:$LD_LIBRARY_PATH" OMP_NUM_THREADS=1
export SUITE_OUT_DIR=/nvme/h/lchristodoulou/data_p318/simulations/suite_test
rm -rf "$SUITE_OUT_DIR"
cd /nvme/h/lchristodoulou/cmb/SkySimulation/examples/sim_suite
PY="$ENV/bin/python -u generate_suite.py"
$PY plan
first() { $ENV/bin/python -c "import pandas as pd; t=pd.read_csv('$SUITE_OUT_DIR/plan/tasks.csv'); print(t[t.kind=='$1'].task.iloc[0])"; }
$PY run --task 0 --limit 40                       # train (with hard negatives)
$PY run --task $(first lcdm | tail -1) --limit 40 # (train again is task 0; ref below)
$PY run --task 40 --limit 40                      # ref
$PY run --task $(first gates) --limit 20          # gates: 20 pairs = 40 skies
$PY run --task $(first anomalies) --limit 2       # 2 twins (15 skies each)
$PY run --task $(first w0wa) --limit 50           # 2 chunks of 25
$PY run --task $(first ede) --limit 2             # 2 EDE refits + 5 skies each
ls -la "$SUITE_OUT_DIR/shards"
