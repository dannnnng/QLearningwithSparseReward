## Qlearning Maze Reinforcement Learning Experiments

This repository contains a single Python script, [QLearningUCBsparse-Maze.py](./QLearningUCBsparse-Maze.py), for running maze-navigation reinforcement learning experiments and exporting publication-style figures.

The script compares three methods on the same maze:

- `Proposed`
- `UCB-H`
- `ε-greedy`

It also exports:

- reward curves
- path visualizations
- path comparison figures
- GIF animations of episode trajectories
- Q-table and visit-count summaries

## Features

- Deterministic maze generation from a seed
- Configurable maze size, horizon, penalties, and number of episodes
- Side-by-side comparison of three algorithms
- Support for sparse-reward-only evaluation/training for selected methods
- Automatic export of PDF figures for reports or papers

## Algorithms

The script currently includes:

- `QLearningUCBHoeffdingSparse`
  - Used for both `Proposed` and `UCB-H`
  - Group-specific settings can be assigned to `Proposed`
- `QLearningEpsilonGreedy`

## Current Experiment Setup

The current experiment batch is defined inside `main()` and uses four groups:

- `maze1_20x20_a`
- `maze1_20x20_b`
- `maze1_20x20_c`
- `maze1_20x20_d`

Each group runs the following three methods:

- `Proposed`
- `UCB-H`
- `ε-greedy`

The current group-level `Proposed` settings are:

1. Group `a`
   - no reward shaping
   - `sparse_fraction = 0.01`
2. Group `b`
   - reward shaping enabled
   - `sparse_fraction = 1.0`
3. Group `c`
   - reward shaping enabled
   - `sparse_fraction = 0.0005`
4. Group `d`
   - reward shaping enabled
   - `sparse_fraction = 0.01`

For the current code:

- `UCB-H` is configured to train without reward shaping
- `ε-greedy` is configured to train without reward shaping

## Requirements

Install Python 3.10+ and the following packages:

```bash
pip install numpy matplotlib pillow
```

## Run

Run the script directly:

```bash
python QLearningUCBsparse-Maze1.py
```

Optional command-line arguments:

```bash
python QLearningUCBsparse-Maze1.py \
  --seed 42 \
  --episodes 200 \
  --horizon 2000 \
  --stay_penalty -0.2 \
  --move_penalty -0.2
```

Supported CLI arguments:

- `--seed`
- `--output_dir`
- `--wall_penalty`
- `--stay_penalty`
- `--move_penalty`
- `--episodes`
- `--horizon`

## Output Structure

Each run creates a timestamp-named output directory containing subfolders such as:

- `plots/`
- `visualizations/`
- `animations/`
- `data/`
- `tables/`

The script also exports summary PDF figures in the root of the output directory, currently named:

- `Fig1.pdf` to `Fig8.pdf`

These correspond to:

- path comparison figures
- reward comparison figures

for groups `a`, `b`, `c`, and `d`.

## Main File

Core implementation is in:

- [QLearningUCBsparse-Maze.py](./QLearningUCBsparse-Maze.py)

Important sections:

- `MazeEnv`: environment dynamics and rewards
- `QLearningUCBHoeffdingSparse`: proposed / UCB-H implementation
- `QLearningEpsilonGreedy`: epsilon-greedy baseline
- plotting and export helpers
- `main()`: experiment configuration and batch execution

## Notes

- Some historical experiment configuration code remains in the file but is no longer the active path; the effective experiment batch is the one built from `scenario_templates` and `algorithm_templates` inside `main()`.
- Figure naming and group settings are currently hardcoded for the present experiment design.

## License

Add a license file if you plan to publish the repository publicly.

- UCB uses custom random pools for reproducible tie-breaking and evaluation.
- Maze generation uses fixed `maze_seed`; different seeds produce different layouts.

## Notes
- The `STAY` action is included to study the effect of no-move behavior in sparse reward tasks and has a separate penalty.
- To run only a subset of algorithms, edit the `experiments` list in `main()`.
