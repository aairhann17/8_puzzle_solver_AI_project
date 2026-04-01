from __future__ import annotations

# argparse lets teammates train the ML model from the command line.
import argparse

# Shared ML helpers keep training logic in one place.
from ml_next_move import export_examples_to_csv, generate_labeled_examples, save_model_bundle, train_next_move_model


def main() -> None:
    # Configure CLI arguments so training can be tuned without code edits.
    parser = argparse.ArgumentParser(description="Train the 8-puzzle next-move ML model.")
    parser.add_argument("--samples", type=int, default=1200, help="Number of labeled puzzle states to generate.")
    parser.add_argument("--min-scramble", type=int, default=4, help="Minimum random scramble length.")
    parser.add_argument("--max-scramble", type=int, default=28, help="Maximum random scramble length.")
    parser.add_argument("--estimators", type=int, default=220, help="Number of trees in the random forest.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible generation and training.")
    parser.add_argument(
        "--export-csv",
        type=str,
        default="",
        help="Optional path to export the generated training examples as CSV.",
    )

    # Parse user-provided training configuration.
    args = parser.parse_args()

    # Generate supervised dataset labeled by A* expert moves.
    print(f"Generating {args.samples} labeled puzzle states...")
    examples = generate_labeled_examples(
        sample_count=args.samples,
        min_scramble=args.min_scramble,
        max_scramble=args.max_scramble,
        seed=args.seed,
    )

    # Optionally write generated examples for inspection or reporting.
    if args.export_csv:
        export_examples_to_csv(examples, output_path=args.export_csv)
        print(f"Saved generated dataset to: {args.export_csv}")

    # Train classifier and compute evaluation metrics.
    print("Training random forest model...")
    model, metrics = train_next_move_model(
        examples,
        estimators=args.estimators,
        seed=args.seed,
    )

    # Save artifacts used by the Flask API and web interface.
    save_model_bundle(model, metrics)

    # Print concise training summary for terminal users.
    print("Training complete.")
    print(f"Samples: {metrics['sampleCount']}")
    print(f"Features: {metrics['featureCount']}")
    print(f"Train/Test: {metrics['trainSize']}/{metrics['testSize']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Label distribution: {metrics['labelDistribution']}")


if __name__ == "__main__":
    main()
