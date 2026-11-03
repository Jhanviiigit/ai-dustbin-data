
## Per photo

| Model | n | Errors | Accuracy [95% CI] | Balanced | Dry P / R / F1 | Wet P / R / F1 | AUC | ECE | Confusion (rows true dry, wet) |
|---|---|---|---|---|---|---|---|---|---|
| student_organik_gray_v2 | 148 | 20 | 86.5% [80.0%, 91.1%] | 86.6% | 0.89 / 0.85 / 0.87 | 0.83 / 0.88 / 0.86 | 0.952 | 0.085 | [[68, 12], [8, 60]] |
| student_u1 | 148 | 12 | 91.9% [86.4%, 95.3%] | 92.2% | 0.96 / 0.89 / 0.92 | 0.88 / 0.96 / 0.92 | 0.972 | 0.082 | [[71, 9], [3, 65]] |
| student_u1_B_no_teacher | 148 | 13 | 91.2% [85.6%, 94.8%] | 91.4% | 0.95 / 0.89 / 0.92 | 0.88 / 0.94 / 0.91 | 0.969 | 0.040 | [[71, 9], [4, 64]] |
| teacher_v1_gray | 148 | 12 | 91.9% [86.4%, 95.3%] | 91.7% | 0.91 / 0.94 / 0.93 | 0.92 / 0.90 / 0.91 | 0.956 | 0.057 | [[75, 5], [7, 61]] |
| teacher_v2 | 148 | 10 | 93.2% [88.0%, 96.3%] | 93.0% | 0.92 / 0.96 / 0.94 | 0.95 / 0.90 / 0.92 | 0.962 | 0.052 | [[77, 3], [7, 61]] |

## Per item

| Model | n | Errors | Accuracy [95% CI] | Balanced | Dry P / R / F1 | Wet P / R / F1 | AUC | ECE | Confusion (rows true dry, wet) |
|---|---|---|---|---|---|---|---|---|---|
| student_organik_gray_v2 | 50 | 4 | 92.0% [81.2%, 96.8%] | 92.3% | 0.96 / 0.89 / 0.92 | 0.88 / 0.96 / 0.92 | 0.963 | 0.120 | [[24, 3], [1, 22]] |
| student_u1 | 50 | 3 | 94.0% [83.8%, 97.9%] | 94.4% | 1.00 / 0.89 / 0.94 | 0.88 / 1.00 / 0.94 | 0.981 | 0.080 | [[24, 3], [0, 23]] |
| student_u1_B_no_teacher | 50 | 4 | 92.0% [81.2%, 96.8%] | 92.3% | 0.96 / 0.89 / 0.92 | 0.88 / 0.96 / 0.92 | 0.979 | 0.070 | [[24, 3], [1, 22]] |
| teacher_v1_gray | 50 | 3 | 94.0% [83.8%, 97.9%] | 94.1% | 0.96 / 0.93 / 0.94 | 0.92 / 0.96 / 0.94 | 0.961 | 0.062 | [[25, 2], [1, 22]] |
| teacher_v2 | 50 | 2 | 96.0% [86.5%, 98.9%] | 96.0% | 0.96 / 0.96 / 0.96 | 0.96 / 0.96 / 0.96 | 0.963 | 0.078 | [[26, 1], [1, 22]] |

## Review policy (threshold 0.8, per photo)

| Model | Sent to a person | Errors caught | Errors left (auto-tilted wrongly) | Accuracy of what the bin does alone |
|---|---|---|---|---|
| student_organik_gray_v2 | 16/148 = 11% | 8/20 | 12 | 120/132 = 90.9% |
| student_u1 | 10/148 = 7% | 3/12 | 9 | 129/138 = 93.5% |
| student_u1_B_no_teacher | 15/148 = 10% | 5/13 | 8 | 125/133 = 94.0% |
| teacher_v1_gray | 15/148 = 10% | 7/12 | 5 | 128/133 = 96.2% |
| teacher_v2 | 14/148 = 9% | 6/10 | 4 | 130/134 = 97.0% |

## Paired comparisons (same photos, exact McNemar)

| A | B | n | only A wrong | only B wrong | both wrong | p |
|---|---|---|---|---|---|---|
| student_organik_gray_v2 | student_u1 | 148 | 11 | 3 | 9 | 0.057 |
| student_organik_gray_v2 | student_u1_B_no_teacher | 148 | 11 | 4 | 9 | 0.118 |
| student_organik_gray_v2 | teacher_v1_gray | 148 | 12 | 4 | 8 | 0.077 |
| student_organik_gray_v2 | teacher_v2 | 148 | 14 | 4 | 6 | 0.031 |
| student_u1 | student_u1_B_no_teacher | 148 | 2 | 3 | 10 | 1.000 |
| student_u1 | teacher_v1_gray | 148 | 7 | 7 | 5 | 1.000 |
| student_u1 | teacher_v2 | 148 | 7 | 5 | 5 | 0.774 |
| student_u1_B_no_teacher | teacher_v1_gray | 148 | 7 | 6 | 6 | 1.000 |
| student_u1_B_no_teacher | teacher_v2 | 148 | 8 | 5 | 5 | 0.581 |
| teacher_v1_gray | teacher_v2 | 148 | 2 | 0 | 10 | 0.500 |

## Paired comparisons per item (exact McNemar)

| A | B | items | only A wrong | only B wrong | both wrong | p |
|---|---|---|---|---|---|---|
| student_organik_gray_v2 | student_u1 | 50 | 3 | 2 | 1 | 1.000 |
| student_organik_gray_v2 | student_u1_B_no_teacher | 50 | 3 | 3 | 1 | 1.000 |
| student_organik_gray_v2 | teacher_v1_gray | 50 | 2 | 1 | 2 | 1.000 |
| student_organik_gray_v2 | teacher_v2 | 50 | 3 | 1 | 1 | 0.625 |
| student_u1 | student_u1_B_no_teacher | 50 | 0 | 1 | 3 | 1.000 |
| student_u1 | teacher_v1_gray | 50 | 2 | 2 | 1 | 1.000 |
| student_u1 | teacher_v2 | 50 | 2 | 1 | 1 | 1.000 |
| student_u1_B_no_teacher | teacher_v1_gray | 50 | 3 | 2 | 1 | 1.000 |
| student_u1_B_no_teacher | teacher_v2 | 50 | 3 | 1 | 1 | 0.625 |
| teacher_v1_gray | teacher_v2 | 50 | 1 | 0 | 2 | 1.000 |
