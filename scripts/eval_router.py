"""Evaluate the learned router on a held-out split.

Usage:
    python scripts/eval_router.py --data data/routing_dataset.jsonl --model models/router.pkl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def load_jsonl(path: Path) -> tuple[list[str], list[str]]:
    queries: list[str] = []
    labels: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            queries.append(ex["query"])
            labels.append(ex["action_type"])
    return queries, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the learned router")
    parser.add_argument(
        "--data",
        type=str,
        default="data/routing_dataset.jsonl",
        help="Path to JSONL data",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/router.pkl",
        help="Path to trained model",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to hold out for testing",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    model_path = Path(args.model)

    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        raise SystemExit(1)
    if not model_path.exists():
        print(f"Error: {model_path} not found. Run train_router.py first.")
        raise SystemExit(1)

    print(f"Loading data from {data_path}...")
    queries, labels = load_jsonl(data_path)

    # Same split seed as potential future training scripts
    _, test_queries, _, test_labels = train_test_split(
        queries, labels, test_size=args.test_size, random_state=42, stratify=labels
    )
    print(f"Evaluating on {len(test_queries)} held-out examples")

    from src.router.learned_router import LearnedRouter

    print(f"Loading model from {model_path}...")
    router = LearnedRouter.from_pretrained(model_path)

    predictions = [router.classify(q).action_type.value for q in test_queries]

    print("\n" + classification_report(test_labels, predictions))

    # Summary accuracy
    correct = sum(1 for p, t in zip(predictions, test_labels) if p == t)
    accuracy = correct / len(test_labels)
    target = 0.90
    status = "PASS" if accuracy >= target else "BELOW TARGET"
    print(f"Overall accuracy: {accuracy:.1%} (target: {target:.0%}) [{status}]")


if __name__ == "__main__":
    main()
