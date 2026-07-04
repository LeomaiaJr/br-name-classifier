"""Benchmark the scoring API against curated edge cases."""

import json
from pathlib import Path

from scorer import classify_name, classify_record


def main() -> int:
    cases = json.loads((Path(__file__).parent / "benchmark_cases.json").read_text())
    failures = []
    group_totals: dict[str, int] = {}
    group_passed: dict[str, int] = {}

    print("=== BR Name Classifier Benchmark ===\n")
    print(f"{'Status':6s} {'Score':>5s}  {'Range':12s} {'Group':15s} Name")
    print("-" * 86)

    for case in cases:
        name1 = case.get("name1", case.get("name", ""))
        name2 = case.get("name2", "")
        result = classify_name(name1) if not name2 else classify_record(name1, name2)
        group = case.get("group", "default")
        group_totals[group] = group_totals.get(group, 0) + 1

        ok = True
        expected = []
        if "min" in case:
            ok = ok and result.score >= case["min"]
            expected.append(f">={case['min']}")
        if "max" in case:
            ok = ok and result.score <= case["max"]
            expected.append(f"<={case['max']}")

        if ok:
            group_passed[group] = group_passed.get(group, 0) + 1
        else:
            failures.append((case, result))

        display_name = name1 if not name2 else f"{name1} | {name2}"
        print(f"{'PASS' if ok else 'FAIL':6s} {result.score:5d}  {','.join(expected):12s} {group:15s} {display_name}")

    print("-" * 86)
    print("\nBy group:")
    for group in sorted(group_totals):
        print(f"  {group:15s} {group_passed.get(group, 0):2d}/{group_totals[group]:2d}")

    if failures:
        print("\nFailures:")
        for case, result in failures:
            display_name = case.get("name1", case.get("name", ""))
            if case.get("name2"):
                display_name = f"{display_name} | {case['name2']}"
            print(f"  {display_name}: score={result.score}, confidence={result.confidence}, reasons={result.reasons}")
        return 1

    print("\nAll benchmark cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
