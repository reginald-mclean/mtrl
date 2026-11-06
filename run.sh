
#!/usr/bin/env bash
set -u

# export MUJOCO_GL=egl
export MUJOCO_GL=osmesa
# export PYOPENGL_PLATFORM=egl
export PYOPENGL_PLATFORM=osmesa

export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd /home/reggie/Desktop/mtrl || exit 1

PYTHON=/home/reggie/Desktop/mtrl/.venv/bin/python
#SCRIPT=experiments/width_scaling/mt10_mtmhsac_v2_1024.py
SCRIPT=experiments/width_scaling/humanoidbench_mtmhsac_v2_0512.py
seeds=(12345 23456 34567)
# seeds=(12345 23456 34567 45678 56789 67890 78901 89012 90123 10123)
goals=(1)

for ng in "${goals[@]}"; do
    for seed in "${seeds[@]}"; do
        "$PYTHON" "$SCRIPT" \
            --track \
            --wandb-entity reggies-phd-research \
            --wandb-project goals_or_tasks \
            --seed "$seed" --width 2048 \
            || echo "FAILED: num_goals=$ng seed=$seed" >> failed_runs.log
    done
done
