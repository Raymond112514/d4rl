# D4RL Environment Setup

Reproduce the `d4rl` conda environment used for flow BC + residual RL on PointMaze.

## Prerequisites

- Linux x86_64
- NVIDIA GPU with CUDA 12.x driver (tested with `torch 2.6.0+cu124`)
- Miniconda or Anaconda

Optional system libs for MuJoCo:

```bash
sudo apt-get install -y libgl1 libglfw3 libglew2.2 libosmesa6
```

## 1. Create conda env

```bash
cd d4rl
conda env create -f environment.yml
conda activate d4rl
```

Alternative (pip-only inside an existing Python 3.11 env):

```bash
pip install -r requirements-lock.txt
```

## 2. Install DSRL Stable-Baselines3 fork

The residual RL code uses a modified SB3 fork (not PyPI):

```bash
git clone git@github.com:Raymond112514/dsrl.git
cd dsrl/stable-baselines3
git checkout 18623d180404fa14780f167928c27fd205f4a526
pip install -e .
```

Expected version: `stable_baselines3 2.6.0a1`

## 3. Clone / copy project code

```bash
# Expected layout:
# ~/d4rl/flow/           — flow matching BC
# ~/d4rl/residual_rl/    — residual SAC training
# ~/dsrl/stable-baselines3/
```

## 4. Download Minari dataset

PointMaze envs come from Minari (not the legacy `d4rl` pip package):

```bash
python -c "import minari; minari.download_dataset('D4RL/pointmaze/large-dense-v2')"
```

## 5. Verify install

```bash
conda activate d4rl
python -c "
import torch, minari, gymnasium, mujoco, stable_baselines3
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('minari', minari.__version__)
print('gymnasium', gymnasium.__version__)
print('mujoco', mujoco.__version__)
print('sb3', stable_baselines3.__version__)
"
```

## 6. Run training

```bash
cd d4rl/residual_rl
python train.py \
  --checkpoint ../flow/checkpoints/goal_conditioned_ac1/epoch_0030.pt \
  --fixed-eval --reset-cell 3 10 --goal-cell 7 1 \
  --n-envs 4 \
  --init-rollouts 20 \
  --wandb-project d4rl-residual-rl \
  --residual-scale 0.5
```

## Key package versions (source machine)

| Package | Version |
|---------|---------|
| Python | 3.11.15 |
| torch | 2.6.0+cu124 |
| numpy | 2.4.6 |
| gymnasium | 1.0.0 |
| gymnasium-robotics | 1.4.2 |
| minari | 0.5.3 |
| mujoco | 3.9.0 |
| stable-baselines3 | 2.6.0a1 (DSRL fork) |
| wandb | 0.27.2 |

## Notes

- **No legacy `d4rl` pip package** — datasets/envs are accessed via **Minari** (`D4RL/pointmaze/...`).
- **`gym==0.26.2`** is installed as a transitive dep; code uses **gymnasium**.
- For CPU-only machines, replace the torch line with `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu` and pass `--cpu` to training scripts.
