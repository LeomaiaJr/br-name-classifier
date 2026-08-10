import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from precision import classify_record  # noqa: E402


class ClassifierPrecisionTest(unittest.TestCase):
    def test_production_regressions(self) -> None:
        cases = json.loads(
            (ROOT / "precision_cases.json").read_text(encoding="utf-8")
        )
        for case in cases:
            name1 = case.get("name1", case.get("name", ""))
            name2 = case.get("name2", "")
            with self.subTest(group=case["group"], name1=name1, name2=name2):
                result = classify_record(
                    name1,
                    name2,
                    comma_mode=case.get("comma_mode", "auto"),
                )
                if "min" in case:
                    self.assertGreaterEqual(result.score, case["min"])
                if "max" in case:
                    self.assertLessEqual(result.score, case["max"])

    def test_census_conflict_artifact_is_explicit_and_count_guarded(self) -> None:
        payload = json.loads(
            (
                ROOT / "output/surname_conflicts.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            payload["thresholds"],
            {"min_count": 100, "pctapi": 70.0, "pctblack": 85.0},
        )
        self.assertGreaterEqual(payload["data"]["DO"]["count"], 100)
        self.assertGreaterEqual(payload["data"]["DO"]["pctapi"], 70)
        self.assertGreaterEqual(payload["data"]["BELONY"]["pctblack"], 85)


if __name__ == "__main__":
    unittest.main()
