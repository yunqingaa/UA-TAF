import random
import pickle
import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "encoder"))
from path_utils import build_bigcn_folds, select_rows_for_split


class BiGCNFoldTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"event_id": str(label * 100 + i), "label": label}
                     for label in range(4) for i in range(13)]

    def test_original_floor_slices_and_remainders(self):
        classes = [[str(label * 100 + i) for i in range(13)] for label in range(4)]
        rng = random.Random(2026)
        for ids in classes:
            rng.shuffle(ids)
        folds = build_bigcn_folds(self.rows)
        all_ids = {r["event_id"] for r in self.rows}
        tested = set()
        for fold, (test, train) in enumerate(folds):
            expected = {eid for ids in classes for eid in ids[fold*2:(fold+1)*2]}
            self.assertEqual(set(test), expected)
            self.assertEqual(set(train), all_ids - expected)
            self.assertFalse(tested & set(test))
            tested.update(test)
        self.assertEqual(len(all_ids - tested), 12)
        self.assertEqual(folds, build_bigcn_folds(self.rows, 2026))
        self.assertNotEqual(folds, build_bigcn_folds(self.rows, 2027))

    def test_partition_is_disjoint_and_iteration_controls_test(self):
        with patch("path_utils.PROJECT_ROOT", Path("missing-test-project")), patch("path_utils.apply_npz_labels", side_effect=lambda rows, *a: rows):
            def select(split, iteration=0):
                return {r["event_id"] for r in select_rows_for_split(
                    self.rows, split, "PHEME", 0, graph_dir="missing-test-graphs",
                    iteration=iteration)}
            train, dev, test = [select(s) for s in ("train", "dev", "test")]
            self.assertFalse(train & dev or train & test or dev & test)
            self.assertEqual(train | dev | test, {r["event_id"] for r in self.rows})
            self.assertEqual(test, select("test"))
            self.assertNotEqual(test, select("test", 1))

    def test_prepared_roster_excludes_unprocessed_csv_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            meta = root / "data" / "PHEME4L"
            meta.mkdir(parents=True)
            with open(meta / "folds.pkl", "wb") as handle:
                pickle.dump(build_bigcn_folds(self.rows, 99), handle)
            extra = {"event_id": "unprocessed", "label": 0}
            with patch("path_utils.PROJECT_ROOT", root), patch(
                "path_utils.apply_npz_labels", side_effect=lambda rows, *a: rows
            ) as labels:
                selected = select_rows_for_split(self.rows + [extra], "test", "PHEME", 0)
                self.assertEqual(labels.call_args.args[0], self.rows)
                self.assertEqual({r["event_id"] for r in selected},
                                 set(build_bigcn_folds(self.rows, 2026)[0][0]))


if __name__ == "__main__":
    unittest.main()
