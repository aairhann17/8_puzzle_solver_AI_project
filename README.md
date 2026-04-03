# 8-Puzzle Solver AI Project

An interactive AI project for solving the 8-puzzle with classical search and a modern ML extension.

It includes:

- Classical AI solvers: BFS, DFS, A* (Manhattan distance)
- A CLI workflow for algorithm comparison and reporting
- A Flask-based web app for live demos
- A supervised learning model that predicts the next move and compares itself with the A* expert

## What This Project Demonstrates

- State-space modeling for combinational puzzles
- Uninformed search (BFS, DFS)
- Informed search (A* with admissible heuristic)
- Performance analysis across algorithms
- Supervised imitation learning from expert labels
- Human-friendly visualization for presentations

## Features

- Solves custom start states for the 8-puzzle
- Detects unsolvable states using inversion parity
- Compares solver metrics:
  - expanded nodes
  - solution depth
  - runtime
  - maximum frontier size
- Optional full path output in CLI
- Optional matplotlib comparison plotting and image export
- Web interface with interactive board editing and solution playback
- ML next-move prediction with confidence values
- One-click `ML Step` control for controlled live demos
- ML-vs-A* expert comparison (match/mismatch)
- ML autoplay mode with speed control and run stats

## Project Structure

- `puzzle_solver.py`: core search algorithms, metrics, CLI, and plotting
- `web_app.py`: Flask app and API routes for solve/random/ML prediction
- `web/index.html`: web UI layout
- `web/styles.css`: UI styling and responsive layout
- `web/app.js`: frontend behavior, API calls, path playback, ML autoplay
- `ml_next_move.py`: dataset generation, training helpers, prediction logic
- `train_ml_model.py`: CLI script to train and save the ML model
- `requirements.txt`: Python dependencies
- `models/`: saved ML model artifacts (`.pkl` and metadata `.json`)

## Requirements

- Python 3.8+

Dependencies (installed via `requirements.txt`):

- Flask
- matplotlib
- scikit-learn

## Quick Start

From project root:

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## CLI Usage

Run with defaults:

```bash
python puzzle_solver.py
```

Run all algorithms on a custom state:

```bash
python puzzle_solver.py --start "1 2 3 4 0 6 7 5 8" --algorithm all
```

Run only A*:

```bash
python puzzle_solver.py --start "2 8 3 1 6 4 7 0 5" --algorithm astar
```

Use DFS with safer limits:

```bash
python puzzle_solver.py --algorithm dfs --dfs-depth-limit 40 --dfs-max-expansions 200000
```

Print full path details:

```bash
python puzzle_solver.py --show-path
```

Show performance plot:

```bash
python puzzle_solver.py --algorithm all --plot
```

Save performance plot:

```bash
python puzzle_solver.py --algorithm all --plot-save comparison.png
```

Show and save plot together:

```bash
python puzzle_solver.py --algorithm all --plot --plot-save comparison.png
```

## Machine Learning Workflow

The ML model learns to imitate the expert solver's first optimal move.

Training process:

1. Generate solvable states by scrambling the goal state.
2. Solve each state with A*.
3. Use A*'s first move as the label.
4. Train a RandomForest classifier.

Train with defaults:

```bash
python train_ml_model.py
```

Train with custom settings:

```bash
python train_ml_model.py --samples 1500 --estimators 250 --export-csv data/next_move_dataset.csv
```

Saved artifacts:

- `models/next_move_model.pkl`
- `models/next_move_model_info.json`

## Web App Usage

Start server:

```bash
python web_app.py
```

Open in browser:

```text
http://127.0.0.1:5000
```

Web demo highlights:

- Interactive click-to-move board editing
- Random solvable board generation
- One-click sample states (easy/medium/hard)
- BFS/DFS/A*/all execution and metrics display
- Step-by-step solution path viewer
- ML next-move prediction with confidence
- Use the `ML Step` button to apply one ML move at a time for step-by-step explanation
- Expert comparison against A* on the same state
- ML autoplay with stop control and speed selector

## Input Format

State format uses numbers `0-8` where `0` is the blank tile.

Grid example:

```text
1 2 3
4 0 6
7 5 8
```

CLI string example:

```text
"1 2 3 4 0 6 7 5 8"
```

## Metrics Reported

- `Solved`: whether a solution was found
- `Depth`: number of moves in the found solution
- `Expanded`: nodes expanded during search
- `Frontier`: peak frontier size
- `Time (s)`: total runtime in seconds

## Algorithm Notes

- BFS is complete and optimal for unit-cost moves.
- DFS is not optimal and is sensitive to depth/expansion limits.
- A* with Manhattan distance is admissible for 8-puzzle and usually expands fewer nodes than BFS/DFS.
