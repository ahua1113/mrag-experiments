#!/usr/bin/env python3
"""Compare JSON result files emitted by benchmark_vidore.py."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+")
    args = parser.parse_args()

    paths: list[Path] = []
    for pattern in args.results:
        matches = [Path(value) for value in glob.glob(pattern)]
        paths.extend(matches or [Path(pattern)])

    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "dataset": payload["dataset"].removeprefix("vidore/"),
                "model": payload["model_type"],
                "queries": payload["num_queries"],
                "docs": payload["num_documents"],
                "nDCG@5": payload["metrics"]["ndcg_at_5"],
                "Recall@1": payload["metrics"]["recall_at_1"],
                "Recall@5": payload["metrics"]["recall_at_5"],
                "doc_s/item": payload["timings_seconds"]["document_seconds_per_item"],
                "query_s/item": payload["timings_seconds"]["query_seconds_per_item"],
            }
        )

    headers = list(rows[0])
    widths = {
        key: max(len(key), *(len(f"{row[key]:.4f}" if isinstance(row[key], float) else str(row[key])) for row in rows))
        for key in headers
    }
    print(" | ".join(key.ljust(widths[key]) for key in headers))
    print("-+-".join("-" * widths[key] for key in headers))
    for row in sorted(rows, key=lambda item: (item["dataset"], item["model"])):
        values = []
        for key in headers:
            value = row[key]
            rendered = f"{value:.4f}" if isinstance(value, float) else str(value)
            values.append(rendered.ljust(widths[key]))
        print(" | ".join(values))


if __name__ == "__main__":
    main()
