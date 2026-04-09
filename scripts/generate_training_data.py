"""Generate query→action_type training pairs using a frontier model via LiteLLM.

Usage:
    python scripts/generate_training_data.py --n 5000 --output data/routing_dataset.jsonl
    python scripts/generate_training_data.py --sample  # 50 examples for quick testing
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from dotenv import load_dotenv

import litellm

load_dotenv()

ACTION_TYPES = [
    "direct_response",
    "single_agent",
    "multi_agent",
    "escalation",
    "clarification",
]

SYSTEM_PROMPT = """\
You are a dataset generator for a query-routing classifier. The classifier maps user queries
to one of 5 action types:

- direct_response: Simple factual questions, greetings, conversions, definitions, short answers.
- single_agent: Standard tasks for one model — code generation, summarisation, writing, analysis.
- multi_agent: Complex tasks needing multi-step reasoning, comparisons, system design, workflows.
- escalation: High-stakes queries about security, production incidents, critical failures, major refactors.
- clarification: Vague, ambiguous, or too-short queries where the user's intent is unclear.

Generate exactly {batch_size} training examples as a JSON array. Each element must be:
{{"query": "<realistic user query>", "action_type": "<one of the 5 types>"}}

Requirements:
- Distribute evenly across all 5 action types ({per_type} each).
- Include {hard_negatives} HARD NEGATIVES — queries that look like one type but are actually another.
  Mark these with an extra field: "hard_negative": true
- Vary length (3 words to 3 sentences), tone (casual, formal, technical), and domain.
- Do NOT repeat queries. Every query must be unique.
- Output ONLY the JSON array, no markdown fences or commentary.
"""


def _generate_batch(
    batch_size: int,
    hard_negative_count: int,
    model: str = "anthropic/claude-sonnet-4-20250514",
) -> list[dict[str, str]]:
    """Call the frontier model to produce one batch of training examples."""
    per_type = batch_size // len(ACTION_TYPES)
    prompt = SYSTEM_PROMPT.format(
        batch_size=batch_size,
        per_type=per_type,
        hard_negatives=hard_negative_count,
    )
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=4096,
    )
    text = response.choices[0].message.content.strip()
    # Strip markdown fences if the model wraps output
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()
    examples: list[dict[str, str]] = json.loads(text)
    # Validate
    valid = []
    for ex in examples:
        if ex.get("action_type") in ACTION_TYPES and ex.get("query"):
            valid.append({"query": ex["query"], "action_type": ex["action_type"]})
    return valid


def generate_dataset(n: int, output_path: Path, model: str) -> None:
    """Generate n examples in batches and write JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    batch_size = 50  # sweet spot for reliable JSON output
    hard_neg_per_batch = max(5, batch_size // 5)
    collected: list[dict[str, str]] = []
    seen_queries: set[str] = set()

    batches_needed = (n + batch_size - 1) // batch_size
    print(f"Generating {n} examples in {batches_needed} batches of {batch_size}...")

    for i in range(batches_needed):
        if len(collected) >= n:
            break
        print(f"  Batch {i + 1}/{batches_needed}...", end=" ", flush=True)
        try:
            batch = _generate_batch(batch_size, hard_neg_per_batch, model=model)
            deduped = [ex for ex in batch if ex["query"] not in seen_queries]
            seen_queries.update(ex["query"] for ex in deduped)
            collected.extend(deduped)
            print(f"got {len(deduped)} unique examples (total: {len(collected)})")
        except Exception as e:
            print(f"error: {e}")
            continue

    # Trim to exact count and shuffle
    collected = collected[:n]
    random.shuffle(collected)

    with open(output_path, "w") as f:
        for ex in collected:
            f.write(json.dumps(ex) + "\n")

    # Print distribution
    from collections import Counter

    dist = Counter(ex["action_type"] for ex in collected)
    print(f"\nWrote {len(collected)} examples to {output_path}")
    print("Distribution:")
    for at in ACTION_TYPES:
        print(f"  {at}: {dist.get(at, 0)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate routing training data")
    parser.add_argument("--n", type=int, default=5000, help="Number of examples to generate")
    parser.add_argument(
        "--output",
        type=str,
        default="data/routing_dataset.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Generate only 50 examples for quick testing",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-sonnet-4-20250514",
        help="Frontier model to use for generation",
    )
    args = parser.parse_args()

    n = 50 if args.sample else args.n
    output_path = Path(args.output)

    generate_dataset(n, output_path, model=args.model)


if __name__ == "__main__":
    main()
