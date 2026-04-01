from __future__ import annotations

# argparse scripts and the web app both use this shared module.
import csv
import json
import pickle
import random
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Random forest gives a strong, low-risk supervised baseline for this task.
from sklearn.ensemble import RandomForestClassifier
# Accuracy is enough for first-pass model evaluation in this project.
from sklearn.metrics import accuracy_score
# Train/test split gives a simple honest evaluation on held-out states.
from sklearn.model_selection import train_test_split

# Reuse the existing puzzle logic so labels come from the current solver.
from puzzle_solver import GOAL_STATE, State, astar, get_neighbors, manhattan_distance

# Canonical move order used across solver and ML extension.
MOVE_LABELS: Tuple[str, ...] = ("U", "D", "L", "R")

# Default locations for saved model artifacts.
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "next_move_model.pkl"
MODEL_INFO_PATH = MODEL_DIR / "next_move_model_info.json"


# Convert a board state into numeric features for the model.
def state_to_features(state: State) -> List[int]:
    # Start with the raw board values in row-major order.
    features = list(state)

    # Add the current position index of each tile 0..8.
    positions = [state.index(tile) for tile in range(9)]
    features.extend(positions)

    # Add blank position as row/column features.
    blank_index = positions[0]
    blank_row, blank_col = divmod(blank_index, 3)
    features.extend([blank_row, blank_col])

    # Add two simple puzzle difficulty features.
    features.append(sum(1 for idx, value in enumerate(state) if value != 0 and value != GOAL_STATE[idx]))
    features.append(manhattan_distance(state))
    return features


# Build a solvable random state by making legal moves from the goal state.
def random_reachable_state(scramble_steps: int, rng: random.Random) -> State:
    # Start from solved board so every generated board is reachable and solvable.
    state = GOAL_STATE

    # Store previous state to reduce immediate backtracking.
    previous: Optional[State] = None

    # Apply random legal moves to scramble the board.
    for _ in range(max(1, scramble_steps)):
        options = [nxt for _, nxt in get_neighbors(state) if nxt != previous]
        next_state = rng.choice(options)
        previous, state = state, next_state

    return state


# Generate supervised examples where the label is A*'s first optimal move.
def generate_labeled_examples(
    sample_count: int,
    min_scramble: int = 4,
    max_scramble: int = 28,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    # Local RNG keeps generation reproducible for demos and comparisons.
    rng = random.Random(seed)

    # Collected training rows.
    examples: List[Dict[str, Any]] = []

    # Track seen states to avoid duplicate rows.
    seen: set[State] = set()

    # Keep generating until we reach desired sample count.
    while len(examples) < sample_count:
        scramble_steps = rng.randint(min_scramble, max_scramble)
        state = random_reachable_state(scramble_steps, rng)

        # Skip duplicates and already-solved boards to keep labels useful.
        if state in seen or state == GOAL_STATE:
            continue
        seen.add(state)

        # Use A* as the expert policy that produces training labels.
        result = astar(state)
        if not result.found or not result.moves:
            continue

        examples.append(
            {
                "state": state,
                "features": state_to_features(state),
                "label": result.moves[0],
                "solutionDepth": result.solution_depth,
                "scrambleSteps": scramble_steps,
            }
        )

    return examples


# Optional helper for exporting generated examples into a CSV file.
def export_examples_to_csv(examples: Sequence[Dict[str, Any]], output_path: Path | str) -> None:
    # Normalize path and ensure target directory exists.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Define CSV columns for both raw board and engineered features.
    fieldnames = [f"cell_{index}" for index in range(9)]
    fieldnames += [f"tile_{tile}_pos" for tile in range(9)]
    fieldnames += ["blank_row", "blank_col", "misplaced_tiles", "manhattan_distance", "label", "solution_depth", "scramble_steps"]

    # Write header and one row per generated example.
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for example in examples:
            features = example["features"]
            row = {fieldname: features[index] for index, fieldname in enumerate(fieldnames[:22])}
            row["label"] = example["label"]
            row["solution_depth"] = example["solutionDepth"]
            row["scramble_steps"] = example["scrambleSteps"]
            writer.writerow(row)


# Train a next-move classifier and return model plus summary metrics.
def train_next_move_model(
    examples: Sequence[Dict[str, Any]],
    estimators: int = 200,
    seed: int = 42,
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    # Build feature matrix and labels from example dicts.
    features = [example["features"] for example in examples]
    labels = [example["label"] for example in examples]

    # Split data so we can report unbiased held-out accuracy.
    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )

    # Configure and train the classifier.
    model = RandomForestClassifier(
        n_estimators=estimators,
        max_features="sqrt",
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(train_features, train_labels)

    # Evaluate on test split.
    predictions = model.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)

    # Store summary metrics that are useful for app display and presentation.
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy),
        "trainSize": len(train_features),
        "testSize": len(test_features),
        "sampleCount": len(examples),
        "labelDistribution": dict(sorted(Counter(labels).items())),
        "featureCount": len(features[0]) if features else 0,
    }
    return model, metrics


# Save both binary model bundle and human-readable metadata.
def save_model_bundle(model: RandomForestClassifier, metrics: Dict[str, Any]) -> None:
    # Ensure model output directory exists.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Bundle model + metadata so one file can be loaded by the web app.
    bundle = {
        "model": model,
        "metrics": metrics,
        "moveLabels": list(MOVE_LABELS),
    }

    # Save binary model artifact.
    with MODEL_PATH.open("wb") as handle:
        pickle.dump(bundle, handle)

    # Save human-readable metrics file.
    with MODEL_INFO_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    # Clear loader cache so web app immediately sees latest model.
    load_model_bundle.cache_clear()


# Cached model loader so repeated web requests stay fast.
@lru_cache(maxsize=1)
def load_model_bundle(model_path: Optional[Path] = None) -> Dict[str, Any]:
    # Use default model location unless caller overrides it.
    path = model_path or MODEL_PATH
    with path.open("rb") as handle:
        return pickle.load(handle)


# Convenience helper for checking whether a trained model exists yet.
def model_is_available(model_path: Optional[Path] = None) -> bool:
    # Quick existence check used by API before prediction requests.
    return (model_path or MODEL_PATH).exists()


# Predict the best next move for a given puzzle state.
def predict_next_move(state: State, model_path: Optional[Path] = None) -> Dict[str, Any]:
    # Load trained model bundle.
    bundle = load_model_bundle(model_path)
    model: RandomForestClassifier = bundle["model"]

    # Predict probability distribution across move labels.
    probabilities = model.predict_proba([state_to_features(state)])[0]
    classes = list(model.classes_)

    # Choose highest-probability move as final prediction.
    best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    best_move = classes[best_index]
    confidence = float(probabilities[best_index])

    # Also return ranked probability table for UI transparency.
    sorted_scores = sorted(
        (
            {"move": move, "probability": float(probability)}
            for move, probability in zip(classes, probabilities)
        ),
        key=lambda item: item["probability"],
        reverse=True,
    )

    # Return both prediction and model metadata used in demo panel.
    return {
        "predictedMove": best_move,
        "confidence": confidence,
        "probabilities": sorted_scores,
        "metrics": bundle["metrics"],
    }
