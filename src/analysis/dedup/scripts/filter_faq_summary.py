"""Filter FAQ summary to include only questions above a conversation threshold.

Reads:
    src/analysis/outputs/latest/faq_summary.json (or --input)

Writes:
    scripts/outputs/faq_summary_filtered.json
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Filter FAQ summary by min conversations"
    )
    parser.add_argument(
        "--min-conversations",
        type=int,
        default=2,
        help="Minimum unique conversations threshold (default: 2)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="src/analysis/outputs/latest/faq_summary.json",
        help="Input FAQ summary path",
    )
    args = parser.parse_args()

    # Load
    input_path = Path(args.input)
    print(f"Loading {input_path}...")
    with open(input_path) as f:
        data = json.load(f)

    # Filter
    output = {"clusters": []}
    min_convs = args.min_conversations

    for cluster in data.get("clusters", []):
        filtered_questions = [
            q
            for q in cluster.get("questions", [])
            if q.get("unique_conversations", 0) >= min_convs
        ]

        if filtered_questions:  # Only include clusters with questions
            output["clusters"].append(
                {
                    "cluster_id": cluster.get("cluster_id"),
                    "cluster_name": cluster.get("cluster_name"),
                    "questions": filtered_questions,
                }
            )

    # Stats
    total = sum(len(c["questions"]) for c in output["clusters"])
    original_total = data.get("summary", {}).get("total_unique_questions", 0)

    output["summary"] = {
        "total_clusters": len(output["clusters"]),
        "total_unique_questions": total,
        "min_conversations_threshold": min_convs,
        "original_total": original_total,
        "filtered_out": original_total - total,
    }

    # Save to same directory as input
    output_path = input_path.parent / "faq_summary_filtered.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(
        f"Filtered: {original_total} → {total} questions (removed {original_total - total})"
    )
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
