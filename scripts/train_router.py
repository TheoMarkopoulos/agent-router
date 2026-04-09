"""Train the learned router on JSONL data and save to models/router.pkl.

Usage:
    python scripts/train_router.py --data data/routing_dataset.jsonl --output models/router.pkl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> tuple[list[str], list[str]]:
    """Load queries and labels from a JSONL file."""
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
    parser = argparse.ArgumentParser(description="Train the learned router classifier")
    parser.add_argument(
        "--data",
        type=str,
        default="data/routing_dataset.jsonl",
        help="Path to JSONL training data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/router.pkl",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model name",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: {data_path} not found. Run generate_training_data.py first.")
        raise SystemExit(1)

    print(f"Loading data from {data_path}...")
    queries, labels = load_jsonl(data_path)
    print(f"Loaded {len(queries)} examples")

    from collections import Counter

    dist = Counter(labels)
    print("Label distribution:")
    for label, count in sorted(dist.items()):
        print(f"  {label}: {count}")

    from src.router.learned_router import LearnedRouter

    print(f"\nTraining with embedding model: {args.embedding_model}")
    router = LearnedRouter.train(
        queries,
        labels,
        embedding_model=args.embedding_model,
    )

    output_path = Path(args.output)
    router.save(output_path)
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()
