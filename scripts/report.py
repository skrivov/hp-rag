from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate comparison reports for HP-RAG vs vector RAG."
    )
    parser.add_argument("results", nargs="+", help="Paths to evaluation result artifacts")
    parser.add_argument("--output", help="Optional path for rendered report (Markdown)")
    return parser.parse_args()


def load_metrics(result_paths: List[str]) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for path in result_paths:
        payload = json.loads(Path(path).read_text())
        suite = payload.get("suite_name", Path(path).stem)
        metrics[suite] = payload.get("metrics", {})
    return metrics


def render_markdown(metrics: Dict[str, Dict[str, float]]) -> str:
    lines = ["# Evaluation Summary"]
    for suite, suite_metrics in metrics.items():
        lines.append(f"\n## {suite}")
        for name, value in sorted(suite_metrics.items()):
            lines.append(f"- **{name}**: {value:.3f}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    metrics = load_metrics(args.results)
    report = render_markdown(metrics)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
    else:
        print(report)


if __name__ == "__main__":
    main()
