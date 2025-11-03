from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class ContractContext:
    sections: Sequence[tuple[str, str]]
    index: Dict[str, List[str]]
    full_text: str

    def lookup(self, key: str) -> List[str]:
        return self.index.get(key.lower().strip(), [])

    def find_by_substring(self, text: str) -> str | None:
        needle = text.lower()
        for title, content in self.sections:
            haystack = f"{title}\n{content}".lower()
            if needle in haystack:
                return content
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten contract ground-truth JSON into question/answer rows for evaluations.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/contracts/contracts_queries_ground_truth.json"),
        help="Path to the ground-truth JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/contracts/questions.jsonl"),
        help="Destination path for the flattened JSONL questions file.",
    )
    return parser.parse_args()


def load_ground_truth(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of contracts in {path}, got {type(data)}")
    return data


def normalize_key(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def parse_markdown_sections(path: Path) -> ContractContext:
    text = path.read_text(encoding="utf-8")
    sections: list[tuple[str, str]] = []
    current_title: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal current_title, buffer
        if current_title is not None:
            content = "\n".join(buffer).strip()
            sections.append((current_title, content))
        current_title = None
        buffer = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        heading_match = re.match(r"^(#{2,})\s+(.*)", line)
        if heading_match:
            flush()
            current_title = heading_match.group(2).strip()
        else:
            if current_title is not None:
                buffer.append(line)
    flush()

    index: Dict[str, List[str]] = {}

    def register(key: str, content: str) -> None:
        key_norm = key.lower().strip()
        if not key_norm or not content:
            return
        index.setdefault(key_norm, [])
        if content not in index[key_norm]:
            index[key_norm].append(content)

    for title, content in sections:
        if not content:
            continue
        register(normalize_key(title), content)

        section_match = re.match(r"^(\d+)[\.\s]", title)
        if section_match:
            register(f"section {section_match.group(1)}", content)

        schedule_match = re.match(r"^schedule\s+([a-z])", title, re.IGNORECASE)
        if schedule_match:
            register(f"schedule {schedule_match.group(1)}", content)

    return ContractContext(sections=sections, index=index, full_text=text)


@lru_cache(maxsize=16)
def load_contract_context(markdown_path: Path) -> ContractContext:
    if not markdown_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {markdown_path}")
    return parse_markdown_sections(markdown_path)


def section_numbers_from_citation(citation: str) -> List[str]:
    cleaned = citation.lower()
    results: list[str] = []
    for match in re.finditer(r"section[s]?\s+(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?", cleaned):
        start = match.group(1)
        end = match.group(2)
        start_main = start.split(".")[0]
        results.append(start_main)
        if end:
            end_main = end.split(".")[0]
            try:
                start_num = int(start_main)
                end_num = int(end_main)
            except ValueError:
                continue
            for value in range(start_num + 1, end_num + 1):
                results.append(str(value))
    return results


def schedule_letters_from_citation(citation: str) -> List[str]:
    letters = []
    for match in re.finditer(r"schedule\s+([a-z])", citation.lower()):
        letters.append(match.group(1))
    return letters


def gather_reference_contexts(
    citations: Sequence[str],
    context: ContractContext,
) -> List[str]:
    collected: list[str] = []
    seen: set[str] = set()

    for citation in citations:
        citation = citation.strip()
        if not citation:
            continue

        matches: list[str] = []

        # Section references
        section_numbers = section_numbers_from_citation(citation)
        for number in section_numbers:
            matches.extend(context.lookup(f"section {number}"))

        # Schedule references
        for letter in schedule_letters_from_citation(citation):
            matches.extend(context.lookup(f"schedule {letter}"))

        # Direct normalized lookup
        matches.extend(context.lookup(normalize_key(citation)))

        # Fallback: locate section containing the citation text
        if not matches:
            fallback = context.find_by_substring(citation)
            if fallback:
                matches.append(fallback)

        for match in matches:
            snippet = match.strip()
            if snippet and snippet not in seen:
                collected.append(snippet)
                seen.add(snippet)

    return collected


def format_expected_answer(payload: Dict[str, Any]) -> str:
    if not payload:
        return ""

    summary = str(payload.get("summary", "")).strip()
    extras = {key: value for key, value in payload.items() if key != "summary" and value}
    if not extras:
        return summary

    extras_json = json.dumps(extras, ensure_ascii=False, indent=2)
    if summary:
        return f"{summary}\n\nDetails:\n{extras_json}"
    return extras_json


def flatten_contract(
    contract: Dict[str, Any],
) -> Iterable[Dict[str, Any]]:
    queries = contract.get("queries") or []
    if not isinstance(queries, list):
        return []

    markdown_path_value = contract.get("file_name_markdown")
    markdown_path = Path(markdown_path_value) if markdown_path_value else None
    if markdown_path and not markdown_path.is_absolute():
        markdown_path = (Path.cwd() / markdown_path).resolve()
    context = load_contract_context(markdown_path) if markdown_path else None

    pdf_path = contract.get("file_name_pdf")
    slug_source = markdown_path_value or pdf_path or "contract"
    slug = Path(slug_source).stem.replace(" ", "_")
    metadata = contract.get("metadata") or {}

    rows: List[Dict[str, Any]] = []
    for index, entry in enumerate(queries, start=1):
        question = str(entry.get("question", "")).strip()
        if not question:
            continue

        answer_payload = entry.get("answer") or {}
        citations = answer_payload.get("policy_citations") if isinstance(answer_payload, dict) else []
        citation_list = list(citations) if isinstance(citations, Sequence) else []
        references = gather_reference_contexts(citation_list, context) if context else []

        rows.append(
            {
                "id": f"{slug}-{index}",
                "question": question,
                "expected_answer": format_expected_answer(answer_payload),
                "metadata": metadata,
                "contract_markdown": markdown_path_value,
                "contract_pdf": pdf_path,
                "reference_contexts": references,
            }
        )
    return rows


def write_questions(rows: Iterable[Dict[str, Any]], path: Path) -> int:
    questions = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in questions:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    return len(questions)


def main() -> None:
    args = parse_args()
    contracts = load_ground_truth(args.input)

    flattened: List[Dict[str, Any]] = []
    for contract in contracts:
        flattened.extend(flatten_contract(contract))

    total = write_questions(flattened, args.output)
    print(f"Wrote {total} questions to {args.output}")


if __name__ == "__main__":
    main()
