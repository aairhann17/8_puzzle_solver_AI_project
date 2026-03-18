from __future__ import annotations

# argparse builds a command-line interface for choosing algorithm and options.
import argparse
# heapq provides a priority queue for A* frontier ordering by lowest f-score.
import heapq
# itertools.count gives a stable tie-break counter for heap entries.
import itertools
# importlib loads optional modules at runtime (used for matplotlib plotting).
import importlib
# time.perf_counter gives high-resolution runtime measurements.
import time
# deque gives O(1) pops from left for BFS queue behavior.
from collections import deque
# dataclass reduces boilerplate for the SearchResult container.
from dataclasses import dataclass
# Type hints make intent clear for teammates and static analyzers.
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# A puzzle state is a 9-value tuple in row-major order. 0 is the blank tile.
State = Tuple[int, ...]

# This is the target solved arrangement of the puzzle.
GOAL_STATE: State = (1, 2, 3, 4, 5, 6, 7, 8, 0)

# Index movement offsets for the blank tile.
MOVE_DIRS = {
    "U": -3,
    "D": 3,
    "L": -1,
    "R": 1,
}


@dataclass
class SearchResult:
    # Name of the algorithm that produced this result.
    algorithm: str
    # True if a goal was found, False otherwise.
    found: bool
    # Move sequence from start to goal, e.g. ["U", "L", "D"].
    moves: List[str]
    # Board states from start to goal (inclusive).
    states: List[State]
    # Number of expanded states for performance comparison.
    expanded_nodes: int
    # Number of moves in solution path; None when unsolved.
    solution_depth: Optional[int]
    # Total wall-clock time spent in this search.
    elapsed_time: float
    # Largest size reached by frontier data structure.
    max_frontier_size: int
    # Extra message when run fails or stops early.
    reason: str = ""


def parse_state(raw: str) -> State:
    # Accept either commas or spaces between numbers.

    tokens = raw.replace(",", " ").split()

    # 8-puzzle must have exactly 9 positions.
    if len(tokens) != 9:
        raise ValueError("State must contain exactly 9 numbers.")

    # Convert each token to integer and freeze as tuple.
    values = tuple(int(t) for t in tokens)

    # Validate the set is exactly {0,1,2,3,4,5,6,7,8}.
    if set(values) != set(range(9)):
        raise ValueError("State must use all digits 0-8 exactly once.")
    return values


def inversion_count(state: State) -> int:
    # Build list without the blank because parity ignores 0.

    arr = [v for v in state if v != 0]

    # Running count of out-of-order tile pairs.
    inv = 0

    # Check every pair (i, j) with i < j.
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            # If earlier tile is larger, this pair is an inversion.
            if arr[i] > arr[j]:
                inv += 1
    return inv


def is_solvable(state: State) -> bool:
    # For odd-width board (3x3), even inversions means solvable.

    return inversion_count(state) % 2 == 0


def manhattan_distance(state: State) -> int:
    # Sum Manhattan distances of all numbered tiles to their goal positions.

    # Accumulator for total heuristic score.
    distance = 0

    # Visit each board index and tile value.
    for idx, value in enumerate(state):
        # Skip the blank tile in heuristic calculation.
        if value == 0:
            continue

        # In goal state, tile v should be at index v-1.
        goal_idx = value - 1

        # Convert current and goal indices to (row, col).
        r1, c1 = divmod(idx, 3)
        r2, c2 = divmod(goal_idx, 3)

        # Add vertical + horizontal distance for this tile.
        distance += abs(r1 - r2) + abs(c1 - c2)
    return distance


def get_neighbors(state: State) -> Iterable[Tuple[str, State]]:
    # Find blank index so we can move it around.

    zero = state.index(0)

    # Row/column of blank determines legal moves.
    row, col = divmod(zero, 3)

    # Use fixed move order for deterministic behavior.
    for move in ("U", "D", "L", "R"):
        # Skip moves that would leave the board.
        if move == "U" and row == 0:
            continue
        if move == "D" and row == 2:
            continue
        if move == "L" and col == 0:
            continue
        if move == "R" and col == 2:
            continue

        # Compute index to swap with blank for this move.
        swap_idx = zero + MOVE_DIRS[move]

        # Copy current state, apply swap, and emit as immutable tuple.
        next_state = list(state)
        next_state[zero], next_state[swap_idx] = next_state[swap_idx], next_state[zero]
        yield move, tuple(next_state)


def reconstruct_path(
    parents: Dict[State, Optional[State]],
    moves_to_state: Dict[State, Optional[str]],
    end_state: State,
) -> Tuple[List[str], List[State]]:
    # We backtrack from goal to start using parent pointers.

    # Will hold states from goal->start first, then reversed.
    path_states: List[State] = []

    # Will hold moves from goal->start first, then reversed.
    path_moves: List[str] = []

    # Start from the final state reached by search.
    cursor: Optional[State] = end_state

    # Walk through parent links until start (whose parent is None).
    while cursor is not None:
        path_states.append(cursor)

        # Start state's move is None; others store move from parent.
        move = moves_to_state[cursor]
        if move is not None:
            path_moves.append(move)

        # Move one step toward the start node.
        cursor = parents[cursor]

    # Convert lists from goal->start into start->goal order.
    path_states.reverse()
    path_moves.reverse()
    return path_moves, path_states


def bfs(start: State) -> SearchResult:
    # Record start time for runtime metric.

    start_time = time.perf_counter()

    # If already solved, return immediate result.
    if start == GOAL_STATE:
        elapsed = time.perf_counter() - start_time
        return SearchResult("BFS", True, [], [start], 0, 0, elapsed, 1)

    # FIFO queue ensures level-order expansion.
    frontier = deque([start])

    # Visited prevents repeated expansion of same state.
    visited = {start}

    # Store parent relation for path reconstruction.
    parents: Dict[State, Optional[State]] = {start: None}

    # Store move used to reach each state from its parent.
    moves_to_state: Dict[State, Optional[str]] = {start: None}

    # Count of popped/expanded nodes.
    expanded = 0

    # Track largest queue size seen during run.
    max_frontier = 1

    # Continue until queue is empty or goal is found.
    while frontier:
        # Update peak queue size metric.
        max_frontier = max(max_frontier, len(frontier))

        # Pop oldest element for BFS behavior.
        current = frontier.popleft()

        # This node is now considered expanded.
        expanded += 1

        # If goal reached, reconstruct and return full result.
        if current == GOAL_STATE:
            moves, states = reconstruct_path(parents, moves_to_state, current)
            elapsed = time.perf_counter() - start_time
            return SearchResult(
                "BFS",
                True,
                moves,
                states,
                expanded,
                len(moves),
                elapsed,
                max_frontier,
            )

        # Expand current node by generating legal neighbors.
        for move, nxt in get_neighbors(current):
            # Ignore already visited states.
            if nxt in visited:
                continue

            # Mark discovered so we do not enqueue duplicates.
            visited.add(nxt)

            # Save parent/move for later path reconstruction.
            parents[nxt] = current
            moves_to_state[nxt] = move

            # Add new state to the BFS queue.
            frontier.append(nxt)

    # This path is used only if no solution was found.
    elapsed = time.perf_counter() - start_time
    return SearchResult("BFS", False, [], [], expanded, None, elapsed, max_frontier, "No solution found.")


def dfs(start: State, depth_limit: int = 40, max_expansions: int = 200_000) -> SearchResult:
    # Record start time for runtime metric.

    start_time = time.perf_counter()

    # If already solved, return immediate result.
    if start == GOAL_STATE:
        elapsed = time.perf_counter() - start_time
        return SearchResult("DFS", True, [], [start], 0, 0, elapsed, 1)

    # Stack stores (state, depth) for depth-first traversal.
    stack: List[Tuple[State, int]] = [(start, 0)]

    # Global discovered set avoids cycles and repeated branches.
    discovered = {start}

    # Parent and move maps support final path reconstruction.
    parents: Dict[State, Optional[State]] = {start: None}
    moves_to_state: Dict[State, Optional[str]] = {start: None}

    # Count of expanded nodes and peak stack size.
    expanded = 0
    max_frontier = 1

    # Continue until stack is empty or stopping condition is reached.
    while stack:
        # Update peak stack size metric.
        max_frontier = max(max_frontier, len(stack))

        # Pop latest state for DFS behavior.
        current, depth = stack.pop()

        # This node is now expanded.
        expanded += 1

        # Return as soon as goal is found.
        if current == GOAL_STATE:
            moves, states = reconstruct_path(parents, moves_to_state, current)
            elapsed = time.perf_counter() - start_time
            return SearchResult(
                "DFS",
                True,
                moves,
                states,
                expanded,
                len(moves),
                elapsed,
                max_frontier,
            )

        # Do not go deeper when depth limit has been reached.
        if depth >= depth_limit:
            continue

        # Stop early if expansion cap is reached.
        if expanded >= max_expansions:
            elapsed = time.perf_counter() - start_time
            return SearchResult(
                "DFS",
                False,
                [],
                [],
                expanded,
                None,
                elapsed,
                max_frontier,
                "Expansion cap reached before finding a solution.",
            )

        # Get neighbors once so we can control push order.
        neighbors = list(get_neighbors(current))

        # Reverse push keeps pop order as U, D, L, R.
        for move, nxt in reversed(neighbors):
            # Skip states we already discovered.
            if nxt in discovered:
                continue

            # Mark as discovered and store how we reached it.
            discovered.add(nxt)
            parents[nxt] = current
            moves_to_state[nxt] = move

            # Push child with incremented depth.
            stack.append((nxt, depth + 1))

    # Returned when DFS ends without finding goal.
    elapsed = time.perf_counter() - start_time
    return SearchResult(
        "DFS",
        False,
        [],
        [],
        expanded,
        None,
        elapsed,
        max_frontier,
        "No solution found within depth limit.",
    )


def astar(start: State) -> SearchResult:
    # Record start time for runtime metric.

    start_time = time.perf_counter()

    # If already solved, return immediate result.
    if start == GOAL_STATE:
        elapsed = time.perf_counter() - start_time
        return SearchResult("A*", True, [], [start], 0, 0, elapsed, 1)

    # Tie-break counter prevents tuple comparison on State when f and g tie.
    counter = itertools.count()

    # Heap entries are (f_score, g_score, tie_breaker, state).
    open_heap: List[Tuple[int, int, int, State]] = []

    # Best known cost from start to each state.
    g_score: Dict[State, int] = {start: 0}

    # Parent/move maps for path reconstruction at goal.
    parents: Dict[State, Optional[State]] = {start: None}
    moves_to_state: Dict[State, Optional[str]] = {start: None}

    # Closed set contains states already expanded.
    closed = set()

    # Initial node has g=0 and f=h(start).
    heapq.heappush(open_heap, (manhattan_distance(start), 0, next(counter), start))

    # Count expanded nodes and peak frontier size.
    expanded = 0
    max_frontier = 1

    # Continue while there are candidate states to explore.
    while open_heap:
        # Update metric for largest heap size.
        max_frontier = max(max_frontier, len(open_heap))

        # Pop the state with smallest f-score.
        _, g_current, _, current = heapq.heappop(open_heap)

        # Ignore stale entries for states already processed.
        if current in closed:
            continue

        # Mark state as expanded.
        closed.add(current)
        expanded += 1

        # If goal reached, reconstruct and return result.
        if current == GOAL_STATE:
            moves, states = reconstruct_path(parents, moves_to_state, current)
            elapsed = time.perf_counter() - start_time
            return SearchResult(
                "A*",
                True,
                moves,
                states,
                expanded,
                len(moves),
                elapsed,
                max_frontier,
            )

        # Explore each neighbor of current state.
        for move, nxt in get_neighbors(current):
            # Skip neighbors already finalized in closed set.
            if nxt in closed:
                continue

            # Every move has cost 1 in this problem.
            tentative_g = g_current + 1

            # Compare against best known g-value for this neighbor.
            old_g = g_score.get(nxt)
            if old_g is None or tentative_g < old_g:
                # Save improved cost and path relation.
                g_score[nxt] = tentative_g
                parents[nxt] = current
                moves_to_state[nxt] = move

                # Compute f = g + h and push into priority queue.
                f_score = tentative_g + manhattan_distance(nxt)
                heapq.heappush(open_heap, (f_score, tentative_g, next(counter), nxt))

    # Returned when no path is found.
    elapsed = time.perf_counter() - start_time
    return SearchResult("A*", False, [], [], expanded, None, elapsed, max_frontier, "No solution found.")


def board_to_string(state: State) -> str:
    # Build a printable 3x3 board; blank tile is shown as underscore.

    # Accumulate row strings here.
    rows = []

    # Slice the flat tuple into 3 rows.
    for i in range(0, 9, 3):
        # Convert numbers to text and map 0 -> "_" for readability.
        row = ["_" if x == 0 else str(x) for x in state[i : i + 3]]

        # Join row values with spaces.
        rows.append(" ".join(row))

    # Join rows with newline characters.
    return "\n".join(rows)


def print_results(start: State, results: Sequence[SearchResult], show_path: bool) -> None:
    # Print start and goal boards first for quick visual context.

    print("Start state:")
    print(board_to_string(start))
    print("\nGoal state:")
    print(board_to_string(GOAL_STATE))

    # Print table header for algorithm comparison metrics.
    print("\nPerformance Comparison")
    print("=" * 78)
    print(
        f"{'Algorithm':<10} {'Solved':<8} {'Depth':<8} {'Expanded':<12} {'Frontier':<10} {'Time (s)':<10}"
    )
    print("-" * 78)

    # Print one row per algorithm result.
    for result in results:
        # Use dash when depth is unavailable (unsolved case).
        depth_text = "-" if result.solution_depth is None else str(result.solution_depth)
        print(
            f"{result.algorithm:<10} {str(result.found):<8} {depth_text:<8} "
            f"{result.expanded_nodes:<12} {result.max_frontier_size:<10} {result.elapsed_time:<10.6f}"
        )

    # Optionally print detailed move/state paths.
    if show_path:
        for result in results:
            print("\n" + "=" * 78)
            print(f"{result.algorithm} path details")
            print("-" * 78)

            # If algorithm failed, print reason and continue.
            if not result.found:
                print(f"No path available. {result.reason}".strip())
                continue

            # Print the move sequence; special label for solved start state.
            move_text = " ".join(result.moves) if result.moves else "(already solved)"
            print(f"Moves ({len(result.moves)}): {move_text}")
            print("State sequence:")

            # Print board for each step in the solution path.
            for idx, state in enumerate(result.states):
                print(f"\nStep {idx}")
                print(board_to_string(state))


def plot_comparison(results: Sequence[SearchResult], show_plot: bool, save_path: Optional[str]) -> None:
    # Load matplotlib only when needed so normal CLI usage has no extra dependency.
    plt = importlib.import_module("matplotlib.pyplot")

    # Build x-axis labels from algorithm names.
    labels = [result.algorithm for result in results]

    # Extract metrics that we want to compare visually.
    expanded = [result.expanded_nodes for result in results]
    elapsed = [result.elapsed_time for result in results]

    # Use 0 for unsolved depths and annotate those bars as N/A.
    depth_values = [result.solution_depth if result.solution_depth is not None else 0 for result in results]
    depth_missing = [result.solution_depth is None for result in results]

    # Create a 1x3 figure for expanded nodes, solution depth, and runtime.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # Chart 1: expanded nodes.
    axes[0].bar(labels, expanded, color="#3B82F6")
    axes[0].set_title("Expanded Nodes")
    axes[0].set_ylabel("Nodes")
    axes[0].grid(axis="y", alpha=0.25)

    # Chart 2: solution depth.
    bars = axes[1].bar(labels, depth_values, color="#10B981")
    axes[1].set_title("Solution Depth")
    axes[1].set_ylabel("Moves")
    axes[1].grid(axis="y", alpha=0.25)

    # Mark unsolved algorithms clearly on depth chart.
    for idx, missing in enumerate(depth_missing):
        if missing:
            bars[idx].set_hatch("//")
            axes[1].text(idx, 0.05, "N/A", ha="center", va="bottom", fontsize=9)

    # Chart 3: runtime in seconds.
    axes[2].bar(labels, elapsed, color="#F59E0B")
    axes[2].set_title("Runtime")
    axes[2].set_ylabel("Seconds")
    axes[2].grid(axis="y", alpha=0.25)

    # Add an overall figure title and tighten spacing.
    fig.suptitle("8-Puzzle Algorithm Comparison", fontsize=14)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))

    # Save image when requested.
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Saved comparison plot to: {save_path}")

    # Open an interactive window when requested.
    if show_plot:
        plt.show()

    # Close figure to release memory in longer runs.
    plt.close(fig)


def run_algorithms(
    start: State,
    algorithm: str,
    dfs_depth_limit: int,
    dfs_max_expansions: int,
) -> List[SearchResult]:
    # Normalize algorithm text for consistent comparisons.

    selected = algorithm.lower()

    # Collect all requested run outputs in this list.
    results: List[SearchResult] = []

    # Run BFS when requested or when running all algorithms.
    if selected in ("bfs", "all"):
        results.append(bfs(start))

    # Run DFS with user-configured limits when requested.
    if selected in ("dfs", "all"):
        results.append(dfs(start, depth_limit=dfs_depth_limit, max_expansions=dfs_max_expansions))

    # Run A* when requested (accept both astar and a* aliases).
    if selected in ("astar", "a*", "all"):
        results.append(astar(start))

    return results


def main() -> None:
    # Build command-line parser for solver configuration.

    parser = argparse.ArgumentParser(
        description="8-Puzzle solver using BFS, DFS, and A* with Manhattan distance."
    )

    # Start state argument: 9 numbers where 0 is blank tile.
    parser.add_argument(
        "--start",
        type=str,
        default="1 2 3 4 0 6 7 5 8",
        help="Start state as 9 numbers (0 is blank), e.g. '1 2 3 4 0 6 7 5 8'.",
    )

    # Choose which algorithm to run.
    parser.add_argument(
        "--algorithm",
        choices=["bfs", "dfs", "astar", "a*", "all"],
        default="all",
        help="Which algorithm to run.",
    )

    # DFS depth limit helps avoid unbounded deep exploration.
    parser.add_argument(
        "--dfs-depth-limit",
        type=int,
        default=40,
        help="Depth limit used by DFS to avoid very deep exploration.",
    )

    # DFS expansion cap prevents very long runs on hard branches.
    parser.add_argument(
        "--dfs-max-expansions",
        type=int,
        default=200000,
        help="Node expansion cap used by DFS.",
    )

    # Optional flag to print every state in each returned path.
    parser.add_argument(
        "--show-path",
        action="store_true",
        help="Print full move/state sequence for each solved run.",
    )

    # Optional flag to display a matplotlib comparison chart.
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display a comparison graph for expanded nodes, depth, and runtime.",
    )

    # Optional file path to save the comparison chart image.
    parser.add_argument(
        "--plot-save",
        type=str,
        default="",
        help="Save comparison graph to an image file (for example: comparison.png).",
    )

    # Parse all CLI arguments.
    args = parser.parse_args()

    # Parse and validate start state text.
    start = parse_state(args.start)

    # Exit early when state is mathematically unsolvable.
    if not is_solvable(start):
        print("This puzzle configuration is unsolvable (odd inversion count).")
        return

    # Run selected algorithm(s) and collect metrics.
    results = run_algorithms(
        start=start,
        algorithm=args.algorithm,
        dfs_depth_limit=args.dfs_depth_limit,
        dfs_max_expansions=args.dfs_max_expansions,
    )

    # Print summary table and optional path details.
    print_results(start, results, show_path=args.show_path)

    # If plotting was requested, try to render/save charts with matplotlib.
    if args.plot or args.plot_save:
        try:
            plot_comparison(results, show_plot=args.plot, save_path=args.plot_save or None)
        except ImportError:
            print("Matplotlib is not installed. Install it with: pip install matplotlib")


# Run main only when executed directly, not when imported as a module.
if __name__ == "__main__":
    main()
