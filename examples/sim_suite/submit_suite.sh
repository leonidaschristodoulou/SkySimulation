#!/bin/bash
# Generate the simulation suite (cmb-latent/sims_suite.md) on the cpu partition:
#   1. plan (login node or a short job; deterministic, writes plan/ and config.json)
#   2. one array element per row of plan/tasks.csv (finished shards are skipped, so resubmitting is safe)
#   3. merge, then check, each only if everything before it succeeded
#
#   bash submit_suite.sh            # submit 2 + 3 (plan must exist)
#   bash submit_suite.sh --plan     # run the plan first, in a short job, then submit 2 + 3
set -e
cd "$(dirname "$0")"
ENV=/nvme/h/lchristodoulou/.conda/envs/camb
OUT=${SUITE_OUT_DIR:-/nvme/h/lchristodoulou/data_p318/simulations/suite_nside64}
SETUP="export LD_LIBRARY_PATH=$ENV/lib:\$LD_LIBRARY_PATH OMP_NUM_THREADS=1 SUITE_OUT_DIR=$OUT; cd $PWD"
PY="$ENV/bin/python -u generate_suite.py"
mkdir -p logs

dep=""
if [ "$1" == "--plan" ]; then
    jp=$(sbatch --parsable -p cpu -A p318 -c 4 --mem=16G -t 01:00:00 --job-name=suite_plan \
         -o logs/suite_plan_%j.out --wrap "$SETUP; $PY plan")
    dep="--dependency=afterok:$jp"
    N=139   # rows of plan/tasks.csv for the current constants; checked below once the plan exists
else
    N=$(($(wc -l < "$OUT/plan/tasks.csv") - 1))
fi

ja=$(sbatch --parsable $dep -p cpu -A p318 -c 40 --mem=96G -t 04:00:00 --array=0-$((N - 1)) \
     --job-name=suite_run -o logs/suite_run_%A_%a.out \
     --wrap "$SETUP; $PY run --task \$SLURM_ARRAY_TASK_ID --workers 40")
jm=$(sbatch --parsable --dependency=afterok:$ja -p cpu -A p318 -c 4 --mem=120G -t 04:00:00 --job-name=suite_merge \
     -o logs/suite_merge_%j.out --wrap "$SETUP; [ \$($ENV/bin/python -c \"import pandas as pd; print(len(pd.read_csv('$OUT/plan/tasks.csv')))\") == $N ] || { echo 'task count changed'; exit 1; }; $PY merge")
jc=$(sbatch --parsable --dependency=afterok:$jm -p cpu -A p318 -c 4 --mem=32G -t 02:00:00 --job-name=suite_check \
     -o logs/suite_check_%j.out --wrap "$SETUP; $PY check")
echo "plan ${jp:-(existing)}  run $ja (0-$((N - 1)))  merge $jm  check $jc  -> $OUT"
