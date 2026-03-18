# 8-Puzzle Solver AI Project

This project solves the 8-puzzle problem using three search algorithms:

- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- A* Search with Manhattan Distance heuristic

It is designed for AI coursework on state-space search and heuristic-guided search.

## AI Topics Covered

- State-space representation
- Uninformed search (BFS, DFS)
- Informed search (A*)
- Manhattan distance heuristic
- Performance comparison across algorithms

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

## Project File

- `puzzle_solver.py`: complete implementation and CLI runner

## Requirements

- Python 3.8+

Optional (for graph feature):

- matplotlib

Install matplotlib only if you want graphs:

```bash
pip install matplotlib
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
