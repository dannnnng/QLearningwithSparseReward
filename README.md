# Maze RL Experiments

This repository is centered on one script:

- `QLearningUCBsparse-Maze.py`

It runs maze-navigation reinforcement-learning experiments, exports publication-style figures, and saves per-run data, paths, animations, and Q-table summaries.

## Current Algorithms

The current code compares four methods:

- `Proposed`
- `UCB-H [4]`
- `Epsilon-greedy`
- `HER [15]`

Implementation mapping:

- `Proposed` and `UCB-H [4]`: `QLearningUCBHoeffdingSparse`
- `Epsilon-greedy`: `QLearningEpsilonGreedy`
- `HER [15]`: `QLearningHER`

## Environment Notes

Important current behavior in the code:

- wall-hit actions are disabled from the Q-table before learning
- `wall_penalty` is effectively forced to `0.0`
- invalid actions are masked out for `Proposed`, `UCB-H [4]`, `Epsilon-greedy`, and `HER [15]`
- reward-curve y-axis is fixed to `[-500, 2000]`

With reward shaping enabled, the effective penalties are:

- `move_penalty = -0.2`
- `stay_penalty = -0.2`

Without reward shaping:

- `move_penalty = 0.0`
- `stay_penalty = 0.0`

## Default Run Settings

The current batch in `main()` uses:

- `maze_size = 20`
- `maze_seed = 1`
- `env_seed = 42`
- `episodes = 500`
- `horizon = 2000`
- `failure_prob p = 0.1`
- `eval_episodes = 500`
- `record_paths = True`

The current fixed maximum delta used by the code is:

- `delta_max = 88341`

In code this is exposed through `FIXED_DELTA_MAX`.

## Figure Mapping

Root-level output figures are currently:

- `Fig1`, `Fig3`, `Fig5`, `Fig7`: learned-path comparison figures
- `Fig2`, `Fig4`, `Fig6`, `Fig8`: reward-curve comparison figures

Scene-to-figure mapping:

| Group | Scenario label | Path figure | Curve figure |
| --- | --- | --- | --- |
| `a` | `maze1_20x20_a` | `Fig1.pdf` | `Fig2.pdf` |
| `b` | `maze1_20x20_b` | `Fig3.pdf` | `Fig4.pdf` |
| `c` | `maze1_20x20_c` | `Fig5.pdf` | `Fig6.pdf` |
| `d` | `maze1_20x20_d` | `Fig7.pdf` | `Fig8.pdf` |

## Current Experiment Groups

The four groups differ only in the `Proposed` configuration. `UCB-H [4]`, `Epsilon-greedy`, and `HER [15]` are run in every group for comparison.

`s` is implemented through `sparse_fraction * horizon`.

| Group | Proposed `s` | Proposed `delta` | Sparse reward only | Reward shaping |
| --- | --- | --- | --- | --- |
| `a` | `200` | `1` | `False` | `True` |
| `b` | `1` | `1` | `True` | `False` |
| `c` | `1` | `1` | `False` | `True` |
| `d` | `20` | `delta_max = 88341` | `False` | `True` |

Equivalent `Proposed` `sparse_fraction` values under the default `horizon = 2000`:

- group `a`: `200 / 2000 = 0.1`
- group `b`: `1 / 2000 = 0.0005`
- group `c`: `1 / 2000 = 0.0005`
- group `d`: `20 / 2000 = 0.01`

## UCB-H [4] Settings

The baseline `UCB-H [4]` currently uses:

- `sparse_fraction = 1.0`, i.e. `s = H`
- `delta = delta_max = 88341`
- no reward shaping
- no sparse-reward-only override

## CLI

Run the batch:

```bash
python QLearningUCBsparse-Maze1.py
```

Supported CLI arguments:

- `--seed`
- `--output_dir`
- `--wall_penalty`
- `--stay_penalty`
- `--move_penalty`
- `--episodes`
- `--horizon`

Note:

- the current code still exposes `--wall_penalty`, but wall-hit actions are disabled and the runtime wall penalty is forced to `0.0`

## Output Structure

Each run creates a timestamp-named output directory. Inside it:

- root: `Fig1.pdf` to `Fig8.pdf`
- per-scenario subdirectories such as `maze_seed_1_size_20_d/`
- each scenario subdirectory contains:
  - `plots/`
  - `visualizations/`
  - `animations/`
  - `data/`
  - `tables/`

Typical saved files:

- `data/experiment_config_*.json`
- `data/q_optimal_path_*.csv`
- `plots/reward_vs_step_*.pdf`
- `visualizations/q_greedy_path_*.pdf`
- `tables/q_table_detailed_*.txt`
- `tables/visit_counts_*.txt`
- `animations/episode_paths_*.gif`

## Dependencies

Python 3.10+ with:

```bash
pip install numpy matplotlib pillow
```

