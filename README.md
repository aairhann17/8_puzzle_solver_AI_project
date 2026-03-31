# 8-Puzzle Solver AI Project

This project solves the 8-puzzle problem using three search algorithms:

- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- A* Search with Manhattan Distance heuristic

It now includes both:

- A CLI solver (`puzzle_solver.py`)
- A web frontend (`web_app.py`) for interactive demos
- A supervised ML extension that learns A*'s best next move

It is designed for AI coursework on state-space search and heuristic-guided search.

## AI Topics Covered

- State-space representation
- Uninformed search (BFS, DFS)
- Informed search (A*)
- Manhattan distance heuristic
- Performance comparison across algorithms
- Supervised learning with a Random Forest classifier
- Expert-label generation using A* solutions

## Features

- Solves a user-provided 8-puzzle start state
- Detects unsolvable states using inversion parity
- Compares:
	- Number of expanded nodes
	- Solution depth
	- Runtime
	- Maximum frontier size
- Optional full solution path printing (moves + intermediate states)
- Optional matplotlib graph for visual performance comparison
- Interactive frontend with click-to-move board editor and path viewer
- ML next-move predictor trained on solver-generated examples

## Project File

- `puzzle_solver.py`: complete implementation and CLI runner

## Requirements

- Python 3.8+

Optional (for graph feature):

- matplotlib
- Flask (for web frontend)
- scikit-learn (for ML extension)

Install matplotlib only if you want graphs:

```bash
pip install matplotlib
```

Or install all project dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

From the project root:

```bash
python puzzle_solver.py
```

### Run all algorithms on a custom start state

```bash
python puzzle_solver.py --start "1 2 3 4 0 6 7 5 8" --algorithm all
```

### Run only A*

```bash
python puzzle_solver.py --start "2 8 3 1 6 4 7 0 5" --algorithm astar
```

### DFS controls

DFS can explore deeply and may expand many nodes, so the script includes safe guards:

```bash
python puzzle_solver.py --algorithm dfs --dfs-depth-limit 40 --dfs-max-expansions 200000
```

### Print full path details

```bash
python puzzle_solver.py --show-path
```

### Show matplotlib comparison graph

```bash
python puzzle_solver.py --algorithm all --plot
```

### Save graph as image file

```bash
python puzzle_solver.py --algorithm all --plot-save comparison.png
```

### Show and save graph together

```bash
python puzzle_solver.py --algorithm all --plot --plot-save comparison.png
```

## Machine Learning Extension

The project now includes a supervised learning extension that predicts the best next move for a puzzle state.

### How the ML part works

- We generate solvable states by scrambling the goal board
- We solve those states with A*
- We use A*'s first optimal move as the training label
- We train a Random Forest classifier to imitate that decision

### Train the model

```bash
python train_ml_model.py
```

Optional example with custom settings:

```bash
python train_ml_model.py --samples 1500 --estimators 250 --export-csv data/next_move_dataset.csv
```

### What gets saved

- `models/next_move_model.pkl`: trained classifier bundle
- `models/next_move_model_info.json`: training metrics summary

### Web demo for ML

After training the model, start the web app and use `Predict Next Move (ML)`.

The frontend will show:

- ML predicted move
- Model confidence
- Training accuracy
- A* expert move for comparison
- Whether the model matches the expert on that state

## Web Frontend

Run the web app from project root:

```bash
python web_app.py
```

Then open:

```text
http://127.0.0.1:5000
```

### Web Features

- Click tiles next to the blank tile to build puzzle states
- Generate random solvable states
- One-click sample cases (easy, medium, hard)
- Run BFS / DFS / A* / all algorithms
- Ask the trained ML model for the next move on the current board
- View performance cards (expanded nodes, depth, runtime, frontier)
- View metric bars for quick comparison
- Play solution paths step-by-step for solved results
- Smooth tile transitions for board updates and path playback

## Input Format

Provide 9 numbers (0-8), where `0` is the blank tile.

Example:

```text
1 2 3
4 0 6
7 5 8
```

CLI input string:

```text
"1 2 3 4 0 6 7 5 8"
```

## Example Output Metrics

The comparison table reports:

- `Solved`: whether the algorithm found a solution
- `Depth`: number of moves in the found solution
- `Expanded`: nodes expanded during search
- `Frontier`: peak frontier size
- `Time (s)`: execution time in seconds

## Notes

- BFS guarantees shortest solution depth for unit-cost moves.
- DFS does not guarantee optimality and can be sensitive to depth limits.
- A* with Manhattan distance is admissible for the 8-puzzle and typically expands far fewer nodes than BFS/DFS.
