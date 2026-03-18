from __future__ import annotations

import argparse
import heapq
import itertools
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

State = Tuple[int, ...]
GOAL_STATE: State = (1, 2, 3, 4, 5, 6, 7, 8, 0)
MOVE_DIRS = {
    "U": -3,
    "D": 3,
    "L": -1,
    "R": 1,
}


@dataclass
class SearchResult:
    algorithm: str
    found: bool
    moves: List[str]
    states: List[State]
    expanded_nodes: int
    solution_depth: Optional[int]
    elapsed_time: float
    max_frontier_size: int
    reason: str = ""


def parse_state(raw: str) -> State:
    tokens = raw.replace(",", " ").split()
    if len(tokens) != 9:
        raise ValueError("State must contain exactly 9 numbers.")

    values = tuple(int(t) for t in tokens)
    if set(values) != set(range(9)):
        raise ValueError("State must use all digits 0-8 exactly once.")
    return values


def inversion_count(state: State) -> int:
    arr = [v for v in state if v != 0]
    inv = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                inv += 1
    return inv


def is_solvable(state: State) -> bool:
    # For odd grid width (3x3), puzzle is solvable when inversions are even.
    return inversion_count(state) % 2 == 0


def manhattan_distance(state: State) -> int:
    distance = 0
    for idx, value in enumerate(state):
        if value == 0:
            continue
        goal_idx = value - 1
        r1, c1 = divmod(idx, 3)
        r2, c2 = divmod(goal_idx, 3)
        distance += abs(r1 - r2) + abs(c1 - c2)
    return distance


def get_neighbors(state: State) -> Iterable[Tuple[str, State]]:
    zero = state.index(0)
    row, col = divmod(zero, 3)

    for move in ("U", "D", "L", "R"):
        if move == "U" and row == 0:
            continue
        if move == "D" and row == 2:
            continue
        if move == "L" and col == 0:
            continue
        if move == "R" and col == 2:
            continue

        swap_idx = zero + MOVE_DIRS[move]
        next_state = list(state)
        next_state[zero], next_state[swap_idx] = next_state[swap_idx], next_state[zero]
        yield move, tuple(next_state)


def reconstruct_path(
    parents: Dict[State, Optional[State]],
    moves_to_state: Dict[State, Optional[str]],
    end_state: State,
) -> Tuple[List[str], List[State]]:
    path_states: List[State] = []
    path_moves: List[str] = []

    cursor: Optional[State] = end_state
    while cursor is not None:
        path_states.append(cursor)
        move = moves_to_state[cursor]
        if move is not None:
            path_moves.append(move)
        cursor = parents[cursor]

    path_states.reverse()
    path_moves.reverse()
    return path_moves, path_states


def bfs(start: State) -> SearchResult:
    start_time = time.perf_counter()
    if start == GOAL_STATE:
        elapsed = time.perf_counter() - start_time
        return SearchResult("BFS", True, [], [start], 0, 0, elapsed, 1)

    frontier = deque([start])
    visited = {start}
    parents: Dict[State, Optional[State]] = {start: None}
    moves_to_state: Dict[State, Optional[str]] = {start: None}

    expanded = 0
    max_frontier = 1

    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        current = frontier.popleft()
        expanded += 1

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

        for move, nxt in get_neighbors(current):
            if nxt in visited:
                continue
            visited.add(nxt)
            parents[nxt] = current
            moves_to_state[nxt] = move
            frontier.append(nxt)

    elapsed = time.perf_counter() - start_time
    return SearchResult("BFS", False, [], [], expanded, None, elapsed, max_frontier, "No solution found.")


def dfs(start: State, depth_limit: int = 40, max_expansions: int = 200_000) -> SearchResult:
    start_time = time.perf_counter()
    if start == GOAL_STATE:
        elapsed = time.perf_counter() - start_time
        return SearchResult("DFS", True, [], [start], 0, 0, elapsed, 1)

    stack: List[Tuple[State, int]] = [(start, 0)]
    discovered = {start}
    parents: Dict[State, Optional[State]] = {start: None}
    moves_to_state: Dict[State, Optional[str]] = {start: None}

    expanded = 0
    max_frontier = 1

    while stack:
        max_frontier = max(max_frontier, len(stack))
        current, depth = stack.pop()
        expanded += 1

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

        if depth >= depth_limit:
            continue

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

        neighbors = list(get_neighbors(current))
        # Reverse push keeps U/D/L/R expansion order when popping from stack.
        for move, nxt in reversed(neighbors):
            if nxt in discovered:
                continue
            discovered.add(nxt)
            parents[nxt] = current
            moves_to_state[nxt] = move
            stack.append((nxt, depth + 1))

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
    start_time = time.perf_counter()
    if start == GOAL_STATE:
        elapsed = time.perf_counter() - start_time
        return SearchResult("A*", True, [], [start], 0, 0, elapsed, 1)

    counter = itertools.count()
    open_heap: List[Tuple[int, int, int, State]] = []
    g_score: Dict[State, int] = {start: 0}
    parents: Dict[State, Optional[State]] = {start: None}
    moves_to_state: Dict[State, Optional[str]] = {start: None}
    closed = set()

    heapq.heappush(open_heap, (manhattan_distance(start), 0, next(counter), start))

    expanded = 0
    max_frontier = 1

    while open_heap:
        max_frontier = max(max_frontier, len(open_heap))
        _, g_current, _, current = heapq.heappop(open_heap)

        if current in closed:
            continue

        closed.add(current)
        expanded += 1

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

        for move, nxt in get_neighbors(current):
            if nxt in closed:
                continue

            tentative_g = g_current + 1
            old_g = g_score.get(nxt)
            if old_g is None or tentative_g < old_g:
                g_score[nxt] = tentative_g
                parents[nxt] = current
                moves_to_state[nxt] = move
                f_score = tentative_g + manhattan_distance(nxt)
                heapq.heappush(open_heap, (f_score, tentative_g, next(counter), nxt))

    elapsed = time.perf_counter() - start_time
    return SearchResult("A*", False, [], [], expanded, None, elapsed, max_frontier, "No solution found.")


def board_to_string(state: State) -> str:
    rows = []
    for i in range(0, 9, 3):
        row = ["_" if x == 0 else str(x) for x in state[i : i + 3]]
        rows.append(" ".join(row))
    return "\n".join(rows)


def print_results(start: State, results: Sequence[SearchResult], show_path: bool) -> None:
    print("Start state:")
    print(board_to_string(start))
    print("\nGoal state:")
    print(board_to_string(GOAL_STATE))

    print("\nPerformance Comparison")
    print("=" * 78)
    print(
        f"{'Algorithm':<10} {'Solved':<8} {'Depth':<8} {'Expanded':<12} {'Frontier':<10} {'Time (s)':<10}"
    )
    print("-" * 78)

    for result in results:
        depth_text = "-" if result.solution_depth is None else str(result.solution_depth)
        print(
            f"{result.algorithm:<10} {str(result.found):<8} {depth_text:<8} "
            f"{result.expanded_nodes:<12} {result.max_frontier_size:<10} {result.elapsed_time:<10.6f}"
        )

    if show_path:
        for result in results:
            print("\n" + "=" * 78)
            print(f"{result.algorithm} path details")
            print("-" * 78)
            if not result.found:
                print(f"No path available. {result.reason}".strip())
                continue

            print(f"Moves ({len(result.moves)}): {' '.join(result.moves) if result.moves else '(already solved)'}")
            print("State sequence:")
            for idx, state in enumerate(result.states):
                print(f"\nStep {idx}")
                print(board_to_string(state))


def run_algorithms(
    start: State,
    algorithm: str,
    dfs_depth_limit: int,
    dfs_max_expansions: int,
) -> List[SearchResult]:
    selected = algorithm.lower()
    results: List[SearchResult] = []

    if selected in ("bfs", "all"):
        results.append(bfs(start))
    if selected in ("dfs", "all"):
        results.append(dfs(start, depth_limit=dfs_depth_limit, max_expansions=dfs_max_expansions))
    if selected in ("astar", "a*", "all"):
        results.append(astar(start))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="8-Puzzle solver using BFS, DFS, and A* with Manhattan distance."
    )
    parser.add_argument(
        "--start",
        type=str,
        default="1 2 3 4 0 6 7 5 8",
        help="Start state as 9 numbers (0 is blank), e.g. '1 2 3 4 0 6 7 5 8'.",
    )
    parser.add_argument(
        "--algorithm",
        choices=["bfs", "dfs", "astar", "a*", "all"],
        default="all",
        help="Which algorithm to run.",
    )
    parser.add_argument(
        "--dfs-depth-limit",
        type=int,
        default=40,
        help="Depth limit used by DFS to avoid very deep exploration.",
    )
    parser.add_argument(
        "--dfs-max-expansions",
        type=int,
        default=200000,
        help="Node expansion cap used by DFS.",
    )
    parser.add_argument(
        "--show-path",
        action="store_true",
        help="Print full move/state sequence for each solved run.",
    )
    args = parser.parse_args()

    start = parse_state(args.start)
    if not is_solvable(start):
        print("This puzzle configuration is unsolvable (odd inversion count).")
        return

    results = run_algorithms(
        start=start,
        algorithm=args.algorithm,
        dfs_depth_limit=args.dfs_depth_limit,
        dfs_max_expansions=args.dfs_max_expansions,
    )
    print_results(start, results, show_path=args.show_path)


if __name__ == "__main__":
    main()
