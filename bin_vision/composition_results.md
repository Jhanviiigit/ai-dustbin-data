Test-split items, 80 scenes per scenario, threshold 0.6. Stand-in classifier instead of the student model (see test_composition.py).

**perfect stand-in**

| Load | Correct | Stopped (safe) | Unsafe tilt | Load type right (all dry / all wet / mixed) |
|---|---|---|---|---|
| dry | 91% | 9% | 0% | 91% |
| wet | 95% | 5% | 0% | 95% |
| dry+dry apart | 100% | 0% | 0% | 100% |
| wet+wet apart | 100% | 0% | 0% | 100% |
| dry+wet apart | 91% | 1% | 8% | 91% |
| dry+dry touching, one by one | 99% | 1% | 0% | 99% |
| dry+wet touching, one by one | 95% | 1% | 4% | 95% |
| dry+dry touching, together | 72% | 28% | 0% | 72% |
| dry+wet touching, together | 4% | 16% | 80% | 4% |
| three, mixed, apart | 95% | 0% | 5% | 95% |

**stand-in, 5% wrong + 5% unsure**

| Load | Correct | Stopped (safe) | Unsafe tilt | Load type right (all dry / all wet / mixed) |
|---|---|---|---|---|
| dry | 81% | 12% | 6% | 85% |
| wet | 86% | 10% | 4% | 91% |
| dry+dry apart | 84% | 16% | 0% | 96% |
| wet+wet apart | 80% | 20% | 0% | 91% |
| dry+wet apart | 71% | 11% | 18% | 81% |
| dry+dry touching, one by one | 69% | 30% | 1% | 80% |
| dry+wet touching, one by one | 81% | 9% | 10% | 89% |
| dry+dry touching, together | 66% | 31% | 2% | 69% |
| dry+wet touching, together | 2% | 21% | 76% | 2% |
| three, mixed, apart | 86% | 4% | 10% | 89% |
