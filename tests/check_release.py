"""CPU packaging checks; no DB image inference or training."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from paperlib.eval import evaluate_rows, hungarian_pairs, match_tp


def main():
    # Exact radius included; just-outside excluded; duplicate predictions do not
    # increase TP; the empty-set rules are those of the existing evaluator.
    assert match_tp(np.array([[10., 0.]]), np.array([[0., 0.]]), 10) == 1
    assert match_tp(np.array([[10.01, 0.]]), np.array([[0., 0.]]), 10) == 0
    assert match_tp(np.array([[0., 0.], [1., 0.]]), np.array([[0., 0.]]), 10) == 1
    assert hungarian_pairs(np.empty((0, 2)), np.array([[0., 0.]])).shape == (0, 3)
    row = dict(key='synthetic', gt=1, pred=2, pred_xy=[[0, 0], [1, 0]], gt_xy=[[0, 0]])
    result = evaluate_rows([row])
    assert result['mae'] == 1 and result['tp'] == 1 and result['fp'] == 1
    assert abs(result['f1'] - 2/3) < 1e-12
    print('PASS: synthetic matching and counting-metric edge cases; no dataset or weights required.')


if __name__ == '__main__':
    main()
