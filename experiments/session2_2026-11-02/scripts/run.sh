#!/usr/bin/env bash
# Session 2 (2026-11-02): score the fixed models on the new items. No training, no tuning.
#
#   bash experiments/session2_2026-11-02/scripts/run.sh <session folder> <models folder> [--dry_only]
#
# Run from the repo root. <session> holds dry/, wet/, background/; <models> holds the five model files
# (README step 2). Writes <session>/items.csv, <session>/crops/ (+ crops/check/ sheets to look at),
# <session>/preds/ and <session>/scores_tables.md.
set -euo pipefail
SESSION=${1:?session folder}; MODELS=${2:?models folder}; DRY_ONLY=${3:-}
S1=experiments/camera_day_2026-10-30/scripts          # the session 1 scripts, unchanged
S2=experiments/session2_2026-11-02/scripts

python $S2/items.py "$SESSION" $DRY_ONLY
rm -rf "$SESSION/crops" "$SESSION/preds"; mkdir -p "$SESSION/preds"
# background removal with exactly the 30 Oct settings (threshold 16, best plate photo, edge strips dropped)
python $S1/crop2.py . "$SESSION" "$SESSION/background" 16 "$SESSION/crops"

score() { python $S1/predict.py . "$SESSION/crops" test "$MODELS/$1" "$2" "$SESSION/preds/$3" 2>&1 | grep -v "xFormers\|warnings.warn\|Using cache" || true; }
score organik_gray_v2.pt     student_organik_gray_v2  test_student_base.csv
score student_u1.pt          student_u1               test_student_u1.csv
score student_u1_B.pt        student_u1_B_no_teacher  test_student_u1_B.csv
score teacher_v1_gray.joblib teacher_v1_gray          test_teacher_v1.csv
score teacher_v2.joblib      teacher_v2               test_teacher_v2.csv

python $S1/report.py . "$SESSION/preds" "$SESSION/scores_tables.md" > /dev/null
python $S2/paired_items.py "$SESSION/preds" >> "$SESSION/scores_tables.md"
echo "done: $SESSION/scores_tables.md  (check the crops in $SESSION/crops/check/ before using the numbers)"
