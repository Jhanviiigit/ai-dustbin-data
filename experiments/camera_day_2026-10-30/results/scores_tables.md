
## Per photo

| Model | n | Errors | Accuracy [95% CI] | Balanced | Dry P / R / F1 | Wet P / R / F1 | AUC | ECE | Confusion (rows true dry, wet) |
|---|---|---|---|---|---|---|---|---|---|
| student_organik_gray_v2 | 113 | 21 | 81.4% [73.3%, 87.5%] | 81.2% | 0.80 / 0.86 / 0.83 | 0.84 / 0.76 / 0.80 | 0.921 | 0.100 | [[51, 8], [13, 41]] |
| student_u1 | 113 | 10 | 91.2% [84.5%, 95.1%] | 91.1% | 0.92 / 0.92 / 0.92 | 0.91 / 0.91 / 0.91 | 0.971 | 0.071 | [[54, 5], [5, 49]] |
| student_u1_B_no_teacher | 113 | 9 | 92.0% [85.6%, 95.8%] | 91.9% | 0.90 / 0.95 / 0.93 | 0.94 / 0.89 / 0.91 | 0.966 | 0.061 | [[56, 3], [6, 48]] |
| teacher_v1_gray | 113 | 7 | 93.8% [87.8%, 97.0%] | 94.1% | 1.00 / 0.88 / 0.94 | 0.89 / 1.00 / 0.94 | 0.991 | 0.050 | [[52, 7], [0, 54]] |
| teacher_v2 | 113 | 7 | 93.8% [87.8%, 97.0%] | 94.1% | 1.00 / 0.88 / 0.94 | 0.89 / 1.00 / 0.94 | 0.994 | 0.044 | [[52, 7], [0, 54]] |

## Per item

| Model | n | Errors | Accuracy [95% CI] | Balanced | Dry P / R / F1 | Wet P / R / F1 | AUC | ECE | Confusion (rows true dry, wet) |
|---|---|---|---|---|---|---|---|---|---|
| student_organik_gray_v2 | 58 | 9 | 84.5% [73.1%, 91.6%] | 84.3% | 0.82 / 0.90 / 0.86 | 0.88 / 0.79 / 0.83 | 0.923 | 0.069 | [[27, 3], [6, 22]] |
| student_u1 | 58 | 4 | 93.1% [83.6%, 97.3%] | 93.1% | 0.93 / 0.93 / 0.93 | 0.93 / 0.93 / 0.93 | 0.974 | 0.058 | [[28, 2], [2, 26]] |
| student_u1_B_no_teacher | 58 | 3 | 94.8% [85.9%, 98.2%] | 94.8% | 0.94 / 0.97 / 0.95 | 0.96 / 0.93 / 0.95 | 0.969 | 0.093 | [[29, 1], [2, 26]] |
| teacher_v1_gray | 58 | 2 | 96.6% [88.3%, 99.0%] | 96.7% | 1.00 / 0.93 / 0.97 | 0.93 / 1.00 / 0.97 | 1.000 | 0.065 | [[28, 2], [0, 28]] |
| teacher_v2 | 58 | 2 | 96.6% [88.3%, 99.0%] | 96.7% | 1.00 / 0.93 / 0.97 | 0.93 / 1.00 / 0.97 | 1.000 | 0.044 | [[28, 2], [0, 28]] |

## Review policy (threshold 0.8, per photo)

| Model | Sent to a person | Errors caught | Errors left (auto-tilted wrongly) | Accuracy of what the bin does alone |
|---|---|---|---|---|
| student_organik_gray_v2 | 18/113 = 16% | 11/21 | 10 | 85/95 = 89.5% |
| student_u1 | 10/113 = 9% | 5/10 | 5 | 98/103 = 95.1% |
| student_u1_B_no_teacher | 22/113 = 19% | 6/9 | 3 | 88/91 = 96.7% |
| teacher_v1_gray | 9/113 = 8% | 4/7 | 3 | 101/104 = 97.1% |
| teacher_v2 | 8/113 = 7% | 5/7 | 2 | 103/105 = 98.1% |

## Paired comparisons (same photos, exact McNemar)

| A | B | n | only A wrong | only B wrong | both wrong | p |
|---|---|---|---|---|---|---|
| student_organik_gray_v2 | student_u1 | 113 | 13 | 2 | 8 | 0.007 |
| student_organik_gray_v2 | student_u1_B_no_teacher | 113 | 14 | 2 | 7 | 0.004 |
| student_organik_gray_v2 | teacher_v1_gray | 113 | 19 | 5 | 2 | 0.007 |
| student_organik_gray_v2 | teacher_v2 | 113 | 19 | 5 | 2 | 0.007 |
| student_u1 | student_u1_B_no_teacher | 113 | 5 | 4 | 5 | 1.000 |
| student_u1 | teacher_v1_gray | 113 | 10 | 7 | 0 | 0.629 |
| student_u1 | teacher_v2 | 113 | 9 | 6 | 1 | 0.607 |
| student_u1_B_no_teacher | teacher_v1_gray | 113 | 9 | 7 | 0 | 0.804 |
| student_u1_B_no_teacher | teacher_v2 | 113 | 9 | 7 | 0 | 0.804 |
| teacher_v1_gray | teacher_v2 | 113 | 1 | 1 | 6 | 1.000 |
