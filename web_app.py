from __future__ import annotations

# Used to build random solvable board states for the UI.
import random
# Type hints keep API payload/response structures clear.
from typing import Any, Dict, List

# Flask powers the lightweight web server and JSON API endpoints.
from flask import Flask, jsonify, request, send_from_directory

# Reuse the core puzzle solver logic so CLI and web stay consistent.
from puzzle_solver import GOAL_STATE, SearchResult, State, get_neighbors, is_solvable, parse_state, run_algorithms

# Serve static frontend assets from ./web under /web URL prefix.
app = Flask(__name__, static_folder="web", static_url_path="/web")


def normalize_state(value: Any) -> State:
    # Convert incoming API state into a validated tuple used by solver functions.
    # Accepted forms:
    # 1) string: "1 2 3 4 0 6 7 5 8"
    # 2) list:   [1,2,3,4,0,6,7,5,8]

    # If state is already a string, reuse existing parser/validator.
    if isinstance(value, str):
        return parse_state(value)

    # If state is a list, validate and normalize manually.
    if isinstance(value, list):
        # 8-puzzle always has exactly 9 positions.
        if len(value) != 9:
            raise ValueError("State list must contain exactly 9 values.")

        # Cast all values to int and report friendly message on failure.
        try:
            state = tuple(int(v) for v in value)
        except (TypeError, ValueError) as exc:
            raise ValueError("State list must contain integer values.") from exc

        # Ensure the set is exactly digits 0..8 with no duplicates/missing values.
        if set(state) != set(range(9)):
            raise ValueError("State list must contain digits 0-8 exactly once.")
        return state

    # Reject unsupported payload types early.
    raise ValueError("State must be a list of 9 numbers or a space-separated string.")


def result_to_json(result: SearchResult, include_path: bool) -> Dict[str, Any]:
    # Map internal SearchResult fields to frontend-friendly JSON keys.
    payload: Dict[str, Any] = {
        "algorithm": result.algorithm,
        "found": result.found,
        "expandedNodes": result.expanded_nodes,
        "solutionDepth": result.solution_depth,
        "elapsedTime": result.elapsed_time,
        "maxFrontierSize": result.max_frontier_size,
        "reason": result.reason,
    }

    # Include move/state sequences only when requested (can be large).
    if include_path:
        payload["moves"] = result.moves
        payload["states"] = [list(state) for state in result.states]

    return payload


def random_reachable_state(steps: int) -> State:
    # Generate a solvable random state by walking from the goal state.
    # Any state reached by legal moves from goal is guaranteed solvable.
    state = GOAL_STATE

    # Track previous state to reduce immediate backtracking oscillations.
    previous: State | None = None

    # Apply a bounded number of random legal moves.
    for _ in range(max(1, steps)):
        # Exclude direct reversal to previous state for better random spread.
        options = [nxt for _, nxt in get_neighbors(state) if nxt != previous]

        # Choose one legal next state uniformly.
        next_state = random.choice(options)

        # Advance walk and remember where we came from.
        previous, state = state, next_state

    return state


@app.get("/")
def index() -> Any:
    # Serve the frontend entry page.
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/random")
def api_random() -> Any:
    # Read randomization depth from query params with safe defaults/bounds.
    steps = request.args.get("steps", default=35, type=int)
    steps = max(1, min(steps, 300))

    # Build and return a solvable random start state.
    state = random_reachable_state(steps)
    return jsonify({"ok": True, "state": list(state)})


@app.post("/api/solve")
def api_solve() -> Any:
    # Parse JSON body safely; fallback to empty dict if body is missing/invalid JSON.
    body = request.get_json(silent=True) or {}

    # Normalize incoming state to solver tuple format.
    try:
        state = normalize_state(body.get("start", list(GOAL_STATE)))
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    # Reject mathematically unsolvable states before running search.
    if not is_solvable(state):
        return jsonify({"ok": False, "solvable": False, "error": "Puzzle state is unsolvable."}), 400

    # Validate algorithm selector.
    algorithm = str(body.get("algorithm", "all")).lower()
    valid_algorithms = {"bfs", "dfs", "astar", "a*", "all"}
    if algorithm not in valid_algorithms:
        return jsonify({"ok": False, "error": "Invalid algorithm selection."}), 400

    # Parse optional DFS control values.
    try:
        dfs_depth_limit = int(body.get("dfsDepthLimit", 40))
        dfs_max_expansions = int(body.get("dfsMaxExpansions", 200000))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "DFS limits must be integers."}), 400

    # Caller can disable full path payload for smaller responses.
    include_path = bool(body.get("includePath", True))

    # Run selected algorithm(s) using shared core solver implementation.
    results = run_algorithms(
        start=state,
        algorithm=algorithm,
        dfs_depth_limit=dfs_depth_limit,
        dfs_max_expansions=dfs_max_expansions,
    )

    # Return start/goal plus all algorithm results in one response.
    return jsonify(
        {
            "ok": True,
            "start": list(state),
            "goal": list(GOAL_STATE),
            "results": [result_to_json(result, include_path=include_path) for result in results],
        }
    )


if __name__ == "__main__":
    # Local development server entry point.
    app.run(host="127.0.0.1", port=5000, debug=True)
