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
from unittest.mock import MagicMock

# Stub torch/torchvision if not installed so student.common can be imported for load_feedback
for mod in ["torch", "torch.nn", "torchvision", "torchvision.models", "torchvision.transforms"]:
    sys.modules.setdefault(mod, MagicMock())

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd
from PIL import Image

from teacher.teacher import index_feedback


class TestTeacherAuditFeedback(unittest.TestCase):
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
            with unittest.mock.patch("student.common.load_feedback", return_value=[(orphan_crop, 0)]):
                with self.assertRaises(ValueError) as ctx:
                    index_feedback(tmp_path)
                self.assertIn("orphan.jpg", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
