"""Evaluate the BR name classifier on the Mission 2 Lusophone eval set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

from scorer import classify_name, classify_record


BASE_DIR = Path(__file__).resolve().parent
EVAL_DIR = BASE_DIR / "data" / "eval"
DEFAULT_REPORT_ROOT = BASE_DIR / "output" / "eval_reports"
FINAL_RUNS_PATH = DEFAULT_REPORT_ROOT / "final_runs.json"
THRESHOLDS = (15, 30, 50, 70)
NEGATIVE_COUNTRIES = {"ES-MX-generic", "HT", "US"}
COUNTRY_CLASSES = ("BR", "PT", "CV", "AO", "MZ", "PALOP_OTHER")
COUNTRY_ALIASES = {
    "BRAZIL": "BR",
    "BRAZILIAN": "BR",
    "BRASIL": "BR",
    "PORTUGAL": "PT",
    "PORTUGUESE": "PT",
    "CAPE_VERDE": "CV",
    "CAPE VERDE": "CV",
    "CABO_VERDE": "CV",
    "CABO VERDE": "CV",
    "ANGOLA": "AO",
    "MOZAMBIQUE": "MZ",
    "MOCAMBIQUE": "MZ",
    "GUINEA_BISSAU": "PALOP_OTHER",
    "GUINEA BISSAU": "PALOP_OTHER",
    "SAO_TOME": "PALOP_OTHER",
    "SAO TOME": "PALOP_OTHER",
    "TIMOR_LESTE": "PALOP_OTHER",
    "TIMOR LESTE": "PALOP_OTHER",
}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def load_eval_rows() -> list[dict[str, str]]:
    with (EVAL_DIR / "eval_names.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    ht_path = EVAL_DIR / "eval_names_ht.csv"
    if ht_path.exists():
        with ht_path.open(newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def load_split_indices(split: str) -> list[int]:
    base_rows = []
    with (EVAL_DIR / "eval_names.csv").open(newline="") as f:
        base_rows = list(csv.DictReader(f))
    split_data = json.loads((EVAL_DIR / "eval_split.json").read_text())
    if split == "all":
        indices = sorted(set(split_data["tuning"]) | set(split_data["final"]))
    else:
        indices = list(split_data[split])

    ht_split_path = EVAL_DIR / "eval_split_ht.json"
    if ht_split_path.exists():
        ht_split = json.loads(ht_split_path.read_text())
        offset = len(base_rows)
        if split == "all":
            ht_indices = sorted(set(ht_split["tuning"]) | set(ht_split["final"]))
        else:
            ht_indices = list(ht_split[split])
        indices.extend(offset + index for index in ht_indices)
    return indices


def model_hash() -> str:
    hasher = hashlib.sha256()
    paths = [
        BASE_DIR / "scorer.py",
        BASE_DIR / "constants.py",
        BASE_DIR / "output" / "frequency_tables.json",
        BASE_DIR / "output" / "meta_model.json",
        BASE_DIR / "output" / "ngram_model.json",
        BASE_DIR / "output" / "pcthispanic_lookup.json",
        BASE_DIR / "output" / "country_tables.json",
        BASE_DIR / "models" / "ngram_pipeline.pkl",
        BASE_DIR / "models" / "meta_classifier.pkl",
    ]
    for path in paths:
        if not path.exists():
            continue
        hasher.update(str(path.relative_to(BASE_DIR)).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def load_final_runs() -> dict[str, str]:
    if not FINAL_RUNS_PATH.exists():
        return {}
    return json.loads(FINAL_RUNS_PATH.read_text())


def save_final_runs(final_runs: dict[str, str]) -> None:
    FINAL_RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINAL_RUNS_PATH.write_text(json.dumps(final_runs, indent=2, sort_keys=True) + "\n")


def guard_final_split(split: str, hash_value: str, force: bool) -> str | None:
    if split not in {"final", "all"}:
        return None
    final_runs = load_final_runs()
    previous = final_runs.get(hash_value)
    if previous and not force:
        raise SystemExit(
            "Refusing to evaluate the final split again for this model hash "
            f"({hash_value[:12]}). First final run: {previous}. Use --force to override."
        )
    if previous and force:
        return (
            "WARNING: --force is overriding the final split guard for model hash "
            f"{hash_value[:12]}; previous final run was {previous}."
        )
    return None


def mark_final_run(split: str, hash_value: str, timestamp: str) -> None:
    if split not in {"final", "all"}:
        return
    final_runs = load_final_runs()
    final_runs[hash_value] = timestamp
    save_final_runs(final_runs)


def mapped_country(country: str) -> str:
    if country in {"GW", "ST", "TL"}:
        return "PALOP_OTHER"
    return country


def normalize_prob_key(key: str) -> str | None:
    raw = str(key).strip()
    upper = raw.upper().replace("-", "_")
    if upper in {"GW", "ST", "TL"}:
        return "PALOP_OTHER"
    if upper in COUNTRY_CLASSES:
        return upper
    return COUNTRY_ALIASES.get(upper)


def extract_country_probs(result: Any) -> dict[str, float]:
    probs = None
    if isinstance(result, dict):
        probs = result.get("country_probs")
    else:
        probs = getattr(result, "country_probs", None)
    if not probs:
        return {}
    if not isinstance(probs, dict):
        return {}

    normalized: dict[str, float] = defaultdict(float)
    for key, value in probs.items():
        mapped = normalize_prob_key(str(key))
        if not mapped:
            continue
        try:
            normalized[mapped] += float(value)
        except (TypeError, ValueError):
            continue
    total = sum(v for v in normalized.values() if v > 0)
    if total <= 0:
        return {}
    return {country: normalized.get(country, 0.0) / total for country in COUNTRY_CLASSES}


def classify_eval_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    evaluated = []
    for row in rows:
        result = classify_name(row["name_normalized"])
        evaluated.append(
            {
                "row": row,
                "score": int(result.score),
                "country_probs": extract_country_probs(result),
            }
        )
    return evaluated


def rate(count: int, total: int) -> float | None:
    if total == 0:
        return None
    return count / total


def stage1_metrics(evaluated: list[dict[str, Any]]) -> dict[str, Any]:
    by_country: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in evaluated:
        by_country[item["row"]["country"]].append(item)

    country_rates = {}
    for country in sorted(by_country):
        items = by_country[country]
        country_rates[country] = {
            "n": len(items),
            "thresholds": {
                str(threshold): {
                    "count": sum(1 for item in items if item["score"] >= threshold),
                    "rate": rate(sum(1 for item in items if item["score"] >= threshold), len(items)),
                }
                for threshold in THRESHOLDS
            },
        }

    negatives = [item for item in evaluated if item["row"]["country"] in NEGATIVE_COUNTRIES]
    negative_rates = {
        str(threshold): {
            "count": sum(1 for item in negatives if item["score"] >= threshold),
            "total": len(negatives),
            "rate": rate(sum(1 for item in negatives if item["score"] >= threshold), len(negatives)),
        }
        for threshold in THRESHOLDS
    }

    return {
        "thresholds": list(THRESHOLDS),
        "country_detection": country_rates,
        "negative_fp": {
            "countries": sorted(NEGATIVE_COUNTRIES),
            "combined": negative_rates,
        },
    }


def entropy(probs: dict[str, float]) -> float:
    return -sum(p * math.log2(p) for p in probs.values() if p > 0)


def normalized_entropy(probs: dict[str, float]) -> float:
    if not probs:
        return 0.0
    return entropy(probs) / math.log2(len(probs))


def stage2_metrics(
    evaluated: list[dict[str, Any]],
    operating_metric: str = "margin",
    operating_threshold: float = 0.45,
) -> dict[str, Any]:
    rows_with_probs = [item for item in evaluated if item["country_probs"]]
    if not rows_with_probs:
        return {
            "available": False,
            "message": "STAGE2: N/A (model does not attribute countries yet)",
        }

    lusophone_rows = [
        item for item in rows_with_probs
        if item["row"]["country"] not in NEGATIVE_COUNTRIES
    ]
    labels = list(COUNTRY_CLASSES)
    confusion = {actual: {pred: 0 for pred in labels} for actual in labels}
    class_totals = Counter()
    class_correct = Counter()
    entropies = []
    normalized_entropies = []
    margins = []
    attributed_predicted = Counter()
    attributed_correct = Counter()
    attributed_actual_correct = Counter()
    calibration_bins = [
        {"bin": i, "min": i / 10, "max": (i + 1) / 10, "n": 0, "correct": 0, "confidence_sum": 0.0}
        for i in range(10)
    ]

    for item in lusophone_rows:
        actual = mapped_country(item["row"]["country"])
        probs = item["country_probs"]
        ranked = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))
        pred, top_prob = ranked[0]
        second_prob = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_prob - second_prob
        normalized_entropy_value = normalized_entropy(probs)
        correct = pred == actual

        confusion[actual][pred] += 1
        class_totals[actual] += 1
        if correct:
            class_correct[actual] += 1
        is_attributed = (
            margin >= operating_threshold
            if operating_metric == "margin"
            else normalized_entropy_value <= operating_threshold
        )
        if is_attributed:
            attributed_predicted[pred] += 1
            if correct:
                attributed_correct[pred] += 1
                attributed_actual_correct[actual] += 1
        entropies.append(entropy(probs))
        normalized_entropies.append(normalized_entropy_value)
        margins.append(margin)

        bin_index = min(9, max(0, int(top_prob * 10)))
        calibration_bins[bin_index]["n"] += 1
        calibration_bins[bin_index]["correct"] += 1 if correct else 0
        calibration_bins[bin_index]["confidence_sum"] += top_prob

    calibration = []
    for item in calibration_bins:
        n = item["n"]
        avg_conf = item["confidence_sum"] / n if n else None
        accuracy = item["correct"] / n if n else None
        calibration.append(
            {
                "bin": item["bin"],
                "range": [item["min"], item["max"]],
                "n": n,
                "accuracy": accuracy,
                "avg_confidence": avg_conf,
                "abs_gap": abs(avg_conf - accuracy) if n else None,
            }
        )

    return {
        "available": True,
        "evaluated_rows": len(lusophone_rows),
        "operating_point": {
            "metric": operating_metric,
            "threshold": operating_threshold,
            "rule": (
                f"top1_top2_margin >= {operating_threshold}"
                if operating_metric == "margin"
                else f"normalized_entropy <= {operating_threshold}"
            ),
        },
        "attributed_precision_coverage_by_class": {
            label: {
                "attributed": attributed_predicted[label],
                "correct": attributed_correct[label],
                "actual_total": class_totals[label],
                "precision": rate(attributed_correct[label], attributed_predicted[label]),
                "coverage": rate(attributed_actual_correct[label], class_totals[label]),
            }
            for label in labels
        },
        "top1_accuracy_by_class": {
            label: {
                "correct": class_correct[label],
                "total": class_totals[label],
                "accuracy": rate(class_correct[label], class_totals[label]),
            }
            for label in labels
        },
        "confusion_matrix": confusion,
        "mean_entropy": sum(entropies) / len(entropies) if entropies else None,
        "mean_normalized_entropy": sum(normalized_entropies) / len(normalized_entropies) if normalized_entropies else None,
        "mean_top1_top2_margin": sum(margins) / len(margins) if margins else None,
        "calibration": calibration,
    }


def run_benchmark_suite() -> dict[str, Any]:
    cases = json.loads((BASE_DIR / "benchmark_cases.json").read_text())
    failures = []
    group_totals: Counter[str] = Counter()
    group_passed: Counter[str] = Counter()

    with redirect_stdout(StringIO()):
        for case in cases:
            name1 = case.get("name1", case.get("name", ""))
            name2 = case.get("name2", "")
            result = classify_name(name1) if not name2 else classify_record(name1, name2)
            group = case.get("group", "default")
            group_totals[group] += 1
            ok = True
            if "min" in case:
                ok = ok and result.score >= case["min"]
            if "max" in case:
                ok = ok and result.score <= case["max"]
            if ok:
                group_passed[group] += 1
            else:
                failures.append(
                    {
                        "name": name1 if not name2 else f"{name1} | {name2}",
                        "score": result.score,
                        "confidence": result.confidence,
                        "group": group,
                    }
                )

    total = len(cases)
    passed = total - len(failures)
    return {
        "passed": passed,
        "total": total,
        "failed": len(failures),
        "pass_rate": rate(passed, total),
        "by_group": {
            group: {
                "passed": group_passed[group],
                "total": group_totals[group],
                "pass_rate": rate(group_passed[group], group_totals[group]),
            }
            for group in sorted(group_totals)
        },
        "failures": failures,
    }


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def build_report_md(report: dict[str, Any]) -> str:
    stage1 = report["stage1"]
    rows = []
    for country, metrics in stage1["country_detection"].items():
        rows.append(
            [
                country,
                metrics["n"],
                fmt_pct(metrics["thresholds"]["15"]["rate"]),
                fmt_pct(metrics["thresholds"]["30"]["rate"]),
                fmt_pct(metrics["thresholds"]["50"]["rate"]),
                fmt_pct(metrics["thresholds"]["70"]["rate"]),
            ]
        )

    fp_rows = []
    for threshold in THRESHOLDS:
        data = stage1["negative_fp"]["combined"][str(threshold)]
        fp_rows.append([f">={threshold}", data["count"], data["total"], fmt_pct(data["rate"])])

    benchmark = report["benchmark"]
    parts = [
        f"# BR Name Classifier Eval Report",
        "",
        f"- Split: `{report['split']}`",
        f"- Eval rows: `{report['row_count']}`",
        f"- Model hash: `{report['model_hash']}`",
        f"- Generated at: `{report['generated_at']}`",
        f"- Benchmark: `{benchmark['passed']}/{benchmark['total']}` passed",
        "",
        "## Stage 1 Gate",
        "",
        markdown_table(["Country", "n", ">=15", ">=30", ">=50", ">=70"], rows),
        "",
        "## Negative False Positives",
        "",
        markdown_table(["Threshold", "FP count", "Negative total", "FP rate"], fp_rows),
        "",
        "## Stage 2 Country Attribution",
        "",
    ]

    stage2 = report["stage2"]
    if not stage2["available"]:
        parts.append(stage2["message"])
    else:
        op = stage2["operating_point"]
        attributed_rows = [
            [
                label,
                values["attributed"],
                values["correct"],
                values["actual_total"],
                fmt_pct(values["precision"]),
                fmt_pct(values["coverage"]),
            ]
            for label, values in stage2["attributed_precision_coverage_by_class"].items()
        ]
        acc_rows = [
            [
                label,
                values["correct"],
                values["total"],
                fmt_pct(values["accuracy"]),
            ]
            for label, values in stage2["top1_accuracy_by_class"].items()
        ]
        parts.extend(
            [
                f"- Evaluated rows: `{stage2['evaluated_rows']}`",
                f"- Operating point: `{op['rule']}`",
                f"- Mean entropy: `{stage2['mean_entropy']:.4f}`",
                f"- Mean normalized entropy: `{stage2['mean_normalized_entropy']:.4f}`",
                f"- Mean top1-top2 margin: `{stage2['mean_top1_top2_margin']:.4f}`",
                "",
                "### Attributed Precision / Coverage",
                "",
                markdown_table(
                    ["Class", "Attributed", "Correct", "Actual total", "Precision", "Coverage"],
                    attributed_rows,
                ),
                "",
                "### Top-1 Accuracy (All Rows, Diagnostic Only)",
                "",
                markdown_table(["Class", "Correct", "Total", "Accuracy"], acc_rows),
                "",
                "### Confusion Matrix",
                "",
                markdown_table(
                    ["Actual"] + list(COUNTRY_CLASSES),
                    [
                        [actual] + [stage2["confusion_matrix"][actual][pred] for pred in COUNTRY_CLASSES]
                        for actual in COUNTRY_CLASSES
                    ],
                ),
                "",
                "### Calibration",
                "",
                markdown_table(
                    ["Bin", "Range", "n", "Accuracy", "Avg confidence", "Abs gap"],
                    [
                        [
                            item["bin"],
                            f"{item['range'][0]:.1f}-{item['range'][1]:.1f}",
                            item["n"],
                            fmt_pct(item["accuracy"]),
                            fmt_pct(item["avg_confidence"]),
                            fmt_pct(item["abs_gap"]),
                        ]
                        for item in stage2["calibration"]
                    ],
                ),
            ]
        )

    if benchmark["failures"]:
        parts.extend(
            [
                "",
                "## Benchmark Failures",
                "",
                markdown_table(
                    ["Group", "Score", "Confidence", "Name"],
                    [
                        [failure["group"], failure["score"], failure["confidence"], failure["name"]]
                        for failure in benchmark["failures"]
                    ],
                ),
            ]
        )

    return "\n".join(parts) + "\n"


def build_report(
    split: str,
    selected_rows: list[dict[str, str]],
    hash_value: str,
    timestamp: str,
    operating_metric: str,
    operating_threshold: float,
) -> dict[str, Any]:
    evaluated = classify_eval_rows(selected_rows)
    country_counts = Counter(item["row"]["country"] for item in evaluated)
    return {
        "schema_version": 1,
        "split": split,
        "generated_at": timestamp,
        "model_hash": hash_value,
        "row_count": len(evaluated),
        "country_counts": dict(sorted(country_counts.items())),
        "stage1": stage1_metrics(evaluated),
        "stage2": stage2_metrics(evaluated, operating_metric, operating_threshold),
        "benchmark": run_benchmark_suite(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("tuning", "final", "all"), required=True)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="Override the final split guard.")
    parser.add_argument(
        "--stage2-operating-metric",
        choices=("margin", "entropy"),
        default="margin",
        help="Metric used to decide whether a Stage-2 country prediction is attributed.",
    )
    parser.add_argument(
        "--stage2-operating-threshold",
        type=float,
        default=0.45,
        help="Threshold for the Stage-2 operating metric.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = utc_timestamp()
    report_dir = args.report_dir or (DEFAULT_REPORT_ROOT / timestamp)
    report_dir = report_dir.resolve()
    hash_value = model_hash()
    warning = guard_final_split(args.split, hash_value, args.force)
    if warning:
        print(warning, file=sys.stderr)

    if report_dir.exists() and any(report_dir.iterdir()):
        raise SystemExit(f"Report directory already exists and is not empty: {report_dir}")
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = load_eval_rows()
    selected = [rows[i] for i in load_split_indices(args.split)]
    report = build_report(
        args.split,
        selected,
        hash_value,
        timestamp,
        args.stage2_operating_metric,
        args.stage2_operating_threshold,
    )

    (report_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (report_dir / "report.md").write_text(build_report_md(report))
    mark_final_run(args.split, hash_value, timestamp)

    print(f"Wrote {report_dir / 'report.json'}")
    print(f"Wrote {report_dir / 'report.md'}")
    print(f"Benchmark: {report['benchmark']['passed']}/{report['benchmark']['total']} passed")
    print(report["stage2"]["message"] if not report["stage2"]["available"] else "STAGE2: available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
