"""python run_eval.py [--seeds 1 2 3 4 5] [--n 100]   -> results/tables.md, results/summary_mean_std.json, plots"""
import argparse

from p7.eval import main

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--givens', type=int, default=30)
    ap.add_argument('--no-overconf', action='store_true')
    a = ap.parse_args()
    main(a.seeds, a.n, a.givens, not a.no_overconf)
