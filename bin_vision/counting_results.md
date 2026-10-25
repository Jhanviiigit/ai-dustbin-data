Test scenes: 150 per scenario, test-split items. Shape rule tuned on val-split scenes: min_solidity=0.0, max_area_frac=0.09, min_color_sep=inf, min_notch2=0.5.

| Method | False alarm (single) | Recall: two apart | Recall: two touching, one by one | Recall: two touching, together | Recall: three apart |
|---|---|---|---|---|---|
| count only | 2% | 83% | 9% | 7% | 95% |
| count + shape | 4% | 89% | 35% | 33% | 99% |
| count + drops | 2% | 95% | 93% | 7% | 100% |
| count + shape + drops | 4% | 96% | 94% | 33% | 100% |

Shape rules that fired on test scenes (single-blob cases): single: {'blob larger than one item': 2, 'notches on both sides of a join': 1}; two apart: {'blob larger than one item': 8, 'notches on both sides of a join': 1}; two touching, one by one: {'blob larger than one item': 37, 'notches on both sides of a join': 2}; two touching, together: {'blob larger than one item': 33, 'notches on both sides of a join': 6}; three apart: {'blob larger than one item': 5, 'notches on both sides of a join': 1}
