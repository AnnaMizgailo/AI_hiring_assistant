from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from parsing import parse_resume_to_profile


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.lower().split())
    return " ".join(str(value).lower().split())


def scalar_match(expected: Any, predicted: Any) -> float:
    if expected in (None, "", []):
        return 1.0 if predicted in (None, "", []) else 0.0
    return 1.0 if norm_text(expected) == norm_text(predicted) else 0.0


def set_f1(expected: Iterable[str], predicted: Iterable[str]) -> float:
    expected_set = {norm_text(item) for item in expected if norm_text(item)}
    predicted_set = {norm_text(item) for item in predicted if norm_text(item)}
    if not expected_set and not predicted_set:
        return 1.0
    if not expected_set or not predicted_set:
        return 0.0
    intersection = len(expected_set & predicted_set)
    precision = intersection / len(predicted_set)
    recall = intersection / len(expected_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def experience_item_key(item: Dict[str, Any]) -> Tuple[str, str]:
    return norm_text(item.get("company")), norm_text(item.get("position"))


def experience_score(expected: List[Dict[str, Any]], predicted: List[Dict[str, Any]]) -> float:
    if not expected and not predicted:
        return 1.0
    if not expected or not predicted:
        return 0.0

    expected_map = {experience_item_key(item): item for item in expected}
    predicted_map = {experience_item_key(item): item for item in predicted}

    matched_scores: List[float] = []
    for key, exp_item in expected_map.items():
        pred_item = predicted_map.get(key)
        if not pred_item:
            matched_scores.append(0.0)
            continue
        item_score = 0.4
        item_score += 0.2 * scalar_match(exp_item.get("date_start"), pred_item.get("date_start"))
        item_score += 0.2 * scalar_match(exp_item.get("date_end"), pred_item.get("date_end"))
        item_score += 0.2 * (1.0 if norm_text(exp_item.get("description", "")) in norm_text(pred_item.get("description", "")) else 0.0)
        matched_scores.append(item_score)
    return sum(matched_scores) / len(expected_map)


def evaluate_record(record: Dict[str, Any]) -> Dict[str, float]:
    raw_text = record["raw_text"]
    expected = record["expected"]

    predicted, _, _ = parse_resume_to_profile(raw_text)

    scores = {
        "full_name": scalar_match(expected.get("full_name"), predicted.get("full_name")),
        "desired_position": scalar_match(expected.get("desired_position"), predicted.get("desired_position")),
        "location": scalar_match(expected.get("location"), predicted.get("location")),
        "english_level": scalar_match(expected.get("english_level"), predicted.get("english_level")),
        "summary": 1.0 if norm_text(expected.get("summary", "")) in norm_text(predicted.get("summary", "")) else 0.0,
        "skills_f1": set_f1(expected.get("skills", []), predicted.get("skills", [])),
        "technologies_f1": set_f1(expected.get("technologies", []), predicted.get("technologies", [])),
        "frameworks_f1": set_f1(expected.get("frameworks", []), predicted.get("frameworks", [])),
        "tools_f1": set_f1(expected.get("tools", []), predicted.get("tools", [])),
        "languages_f1": set_f1(expected.get("languages", []), predicted.get("languages", [])),
        "experience_score": experience_score(expected.get("experience", []), predicted.get("experience", [])),
        "education_hit": 1.0 if bool(predicted.get("education")) == bool(expected.get("education")) else 0.0,
        "total_experience_years": 1.0 if abs(float(expected.get("total_experience_years", 0) or 0) - float(predicted.get("total_experience_years", 0) or 0)) <= 0.6 else 0.0,
    }
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate resume parser against a JSONL dataset")
    parser.add_argument("dataset", type=Path, help="Path to JSONL with fields: id, raw_text, expected")
    args = parser.parse_args()

    rows = []
    with args.dataset.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        raise SystemExit("Dataset is empty")

    aggregate: Dict[str, List[float]] = {}
    for row in rows:
        metrics = evaluate_record(row)
        for key, value in metrics.items():
            aggregate.setdefault(key, []).append(value)

    report = {key: round(statistics.mean(values), 4) for key, values in aggregate.items()}
    report["records"] = len(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
