"""
test_teacher.py
---------------
Tests for teacher indexing, feedback handling, and soft-label generation.
"""

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock


def _torch_stubs():
    """Fake torch/torchvision modules for tests that import student.common (which imports torch).
    Applied only inside those tests: a global fake breaks scikit-learn/scipy, which check torch."""
    names = ["torch", "torch.nn", "torchvision", "torchvision.models", "torchvision.transforms"]
    return {n: MagicMock() for n in names if n not in sys.modules}

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd
from PIL import Image

from teacher.teacher import (
    index_feedback,
    out_of_fold_logits,
    run_update,
    sigmoid,
    soft_label_table,
    write_soft_labels,
)


class TestTeacherAuditFeedback(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(sys.modules, _torch_stubs())
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_audit_rows_never_reach_training(self):
        """Proves feedback rows with trigger == 'audit' are dropped and never reach training."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            crops_dir = tmp_path / "crops"
            crops_dir.mkdir()

            # Create dummy image files for crops
            for name in ["crop_low_conf.jpg", "crop_audit.jpg"]:
                im = Image.new("RGB", (32, 32), color=(200, 100, 50))
                im.save(crops_dir / name)

            # Write feedback.csv with both low_confidence and audit triggers
            feedback_csv = tmp_path / "feedback.csv"
            fieldnames = [
                "load_id",
                "item_index",
                "crop_file",
                "box",
                "human_label",
                "trigger",
                "student_version",
                "from_drop",
            ]
            rows = [
                {
                    "load_id": "load_001",
                    "item_index": "0",
                    "crop_file": "crops/crop_low_conf.jpg",
                    "box": "0 0 32 32",
                    "human_label": "dry",
                    "trigger": "low_confidence",
                    "student_version": "v1",
                    "from_drop": "1",
                },
                {
                    "load_id": "load_002",
                    "item_index": "0",
                    "crop_file": "crops/crop_audit.jpg",
                    "box": "0 0 32 32",
                    "human_label": "wet",
                    "trigger": "audit",
                    "student_version": "v1",
                    "from_drop": "1",
                },
            ]
            with open(feedback_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            # Index feedback
            df_feedback = index_feedback(tmp_path)

            # Assertions:
            # 1. Returned rows must only contain non-audit feedback
            self.assertEqual(len(df_feedback), 1)
            self.assertNotIn("audit", df_feedback["trigger"].str.lower().values)
            self.assertEqual(df_feedback.iloc[0]["trigger"], "low_confidence")
            self.assertEqual(df_feedback.iloc[0]["filename"], "crop_low_conf.jpg")

            # 2. When partitioned into training split, audit rows are strictly absent
            train_rows = df_feedback[df_feedback.split.isin(["train", "feedback"])]
            self.assertFalse((train_rows["trigger"].str.lower() == "audit").any())

    def test_unmatched_crop_raises_value_error(self):
        """A crop from load_feedback with no matching row in feedback.csv must raise ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            feedback_csv = tmp_path / "feedback.csv"
            with open(feedback_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["crop_file", "human_label", "trigger", "load_id"])
                writer.writeheader()

            orphan_crop = tmp_path / "crops" / "orphan.jpg"
            with mock.patch("student.common.load_feedback", return_value=[(orphan_crop, 0)]):
                with self.assertRaises(ValueError) as ctx:
                    index_feedback(tmp_path)
                self.assertIn("orphan.jpg", str(ctx.exception))


def _rows(n, dataset="dataset_D_v2_balanced"):
    return pd.DataFrame({"dataset": dataset, "split": "train", "class_folder": "dry waste",
                         "filename": [f"img_{i}.jpg" for i in range(n)]})


class TestSoftLabelFormat(unittest.TestCase):
    def test_columns_and_order(self):
        """Columns must be exactly: crop_id, teacher_logit, teacher_wet_prob, fold, teacher_version."""
        df = pd.DataFrame({
            "crop_id": ["crop_1", "crop_2"],
            "filename": ["crop_1.jpg", "crop_2.jpg"],
        })
        logits = np.array([1.5, -2.0])
        folds = np.array([0, 1])
        tab = soft_label_table(df, logits, version="teacher_v2", folds=folds)

        expected_cols = ["crop_id", "teacher_logit", "teacher_wet_prob", "fold", "teacher_version"]
        self.assertEqual(tab.columns.tolist(), expected_cols)
        self.assertEqual(tab["crop_id"].tolist(), ["crop_1", "crop_2"])
        self.assertEqual(tab["fold"].tolist(), [0, 1])
        self.assertEqual(tab["teacher_version"].tolist(), ["teacher_v2", "teacher_v2"])

    def test_teacher_wet_prob_equals_sigmoid_logit(self):
        """teacher_wet_prob must equal sigmoid(teacher_logit)."""
        df = _rows(5)
        logits = np.array([-3.5, -1.0, 0.0, 1.2, 4.0])
        tab = soft_label_table(df, logits, version="teacher_v1", folds=[0, 1, 2, 3, 4])

        np.testing.assert_allclose(
            tab["teacher_wet_prob"].values,
            sigmoid(tab["teacher_logit"].values),
            rtol=1e-7,
            atol=1e-7,
        )

    def test_two_output_head_logits(self):
        """If the head has 2 outputs: teacher_logit = wet minus dry."""
        df = _rows(2)
        # 2D logits: [dry, wet]
        logits_2d = np.array([[0.5, 2.5], [3.0, 1.0]])  # wet - dry => [2.0, -2.0]
        tab = soft_label_table(df, logits_2d, version="teacher_v1", folds=[0, 1])

        np.testing.assert_allclose(tab["teacher_logit"].values, [2.0, -2.0])
        np.testing.assert_allclose(tab["teacher_wet_prob"].values, sigmoid([2.0, -2.0]))

    def test_out_of_fold_logits_folds(self):
        """Verify out_of_fold_logits returns folds matching GroupKFold."""
        np.random.seed(42)
        X = np.random.randn(20, 4)
        y = np.array([0, 1] * 10)
        groups = np.array([f"g_{i // 2}" for i in range(20)])
        views = np.zeros(20, dtype=int)
        C = 1.0

        z, folds = out_of_fold_logits(X, y, groups, views, C, n_folds=3, return_folds=True)
        self.assertEqual(len(folds), 20)
        self.assertEqual(len(z), 20)
        self.assertTrue(set(folds).issubset({0, 1, 2}))

        df = _rows(20)
        tab = soft_label_table(df, z, version="teacher_v2", folds=folds)
        self.assertEqual(tab["fold"].tolist(), folds.tolist())

    def test_write_soft_labels_output_path(self):
        """Writes to runs/teacher_vN/soft_labels.csv."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "runs" / "teacher_v3" / "soft_labels.csv"
            df = _rows(1)
            tab = soft_label_table(df, [0.5], version="teacher_v3", folds=[0])
            written = write_soft_labels(tab, out_file)
            self.assertTrue(written.is_file())
            loaded = pd.read_csv(written)
            self.assertEqual(list(loaded.columns), ["crop_id", "teacher_logit", "teacher_wet_prob", "fold", "teacher_version"])

    def test_crop_ids_unique_when_c_and_d_share_photos(self):
        """C and D hold the same photos (same file names): ids must still be unique."""
        df = pd.concat([_rows(3, "dataset_C_v2"), _rows(3, "dataset_D_v2_balanced")], ignore_index=True)
        tab = soft_label_table(df, np.zeros(6), "teacher_v1", np.zeros(6, int))
        self.assertFalse(tab["crop_id"].duplicated().any())
        self.assertEqual(tab["crop_id"].iloc[0], "dataset_C_v2/train/dry waste/img_0")

    def test_real_crop_id_column_wins(self):
        df = _rows(2).assign(crop_id=["a1", "b2"])
        tab = soft_label_table(df, [0.1, 0.2], "teacher_v1", [0, 1])
        self.assertEqual(tab["crop_id"].tolist(), ["a1", "b2"])

    def test_bad_inputs_raise(self):
        df = _rows(2)
        with self.assertRaises(ValueError):
            soft_label_table(df, [0.1, 0.2], "teacher_v1", [0])          # folds length
        with self.assertRaises(ValueError):
            soft_label_table(df, [0.1, 0.2], "teacher_v1", [0, -1])      # missing fold
        with self.assertRaises(ValueError):
            soft_label_table(df, [0.1, 0.2], "v1", [0, 1])               # version name
        with self.assertRaises(TypeError):
            soft_label_table(df, [0.1, 0.2], "teacher_v1")               # folds required


class _FakeExtractor:
    """Deterministic 8-d 'features' from the label plus noise: no DINOv2 needed."""
    def features(self, df, view=0, seed=0, batch_size=64):
        rng = np.random.default_rng(1000 * view + len(df))
        return rng.normal(size=(len(df), 8)) + df.y.values[:, None] * 2.0


class TestRunUpdate(unittest.TestCase):
    def test_writes_one_soft_label_file_per_train_image(self):
        """Full cycle on two tiny datasets that share file names (like C and D)."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for ds in ("C", "D"):
                for split, n in (("train", 6), ("val", 3), ("test", 3)):
                    for cls in ("dry waste", "wet waste"):
                        folder = tmp / ds / split / cls
                        folder.mkdir(parents=True)
                        for i in range(n):
                            Image.new("RGB", (8, 8)).save(folder / f"img_{i}.jpg")
            datasets = {"dataset_C_v2": (tmp / "C", None, False),
                        "dataset_D_v2_balanced": (tmp / "D", None, False)}
            run_update(datasets, tmp / "runs", "teacher_v7", n_aug=1, extractor=_FakeExtractor())

            out = tmp / "runs" / "teacher_v7" / "soft_labels.csv"
            self.assertTrue(out.is_file())
            self.assertEqual(sorted(p.name for p in out.parent.glob("soft_labels*")), ["soft_labels.csv"])
            tab = pd.read_csv(out)
            self.assertEqual(list(tab.columns), ["crop_id", "teacher_logit", "teacher_wet_prob",
                                                 "fold", "teacher_version"])
            self.assertEqual(len(tab), 2 * 2 * 6)                       # train rows only, view 0 only
            self.assertFalse(tab["crop_id"].duplicated().any())
            self.assertFalse(tab["crop_id"].str.contains("/val/|/test/").any())
            self.assertTrue((tab["fold"] >= 0).all())
            self.assertTrue((tab["teacher_version"] == "teacher_v7").all())


if __name__ == "__main__":
    unittest.main()
