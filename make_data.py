"""Download, verify and preprocess the data reproducibly.

    python make_data.py --seed 1

1. downloads the four MNIST gz files (tries yann.lecun.com, then two mirrors) and checks their MD5;
2. scales pixels to 0-1, draws the seeded 50,000/10,000 train/validation split, drops digit 0,
   and writes data/processed/mnist_1to9_seed{seed}.npz;
3. generates the seeded Sudoku puzzles (unique solution) and writes data/processed/puzzles_seed{seed}_g{givens}.json.
The trained network and the hard-digit bank (needs the network) are produced by `python run_eval.py`
(or `python -c "from p7.eval import prepare_models; prepare_models(1)"`).
"""
import argparse
import json

import numpy as np

from p7.csp import make_puzzle
from p7.data import DATA, download_mnist, prepare

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--n', type=int, default=100, help='number of puzzles')
    ap.add_argument('--givens', type=int, default=30)
    a = ap.parse_args()
    print('MD5:', download_mnist())
    D = prepare(a.seed)
    out = DATA / 'processed'
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / f'mnist_1to9_seed{a.seed}.npz',
                        **{f'X_{k}': v[0] for k, v in D.items()}, **{f'y_{k}': v[1] for k, v in D.items()})
    print({k: v[0].shape for k, v in D.items()})
    import random
    rng = random.Random(a.seed)
    pz = [make_puzzle(rng, a.givens) for _ in range(a.n)]
    json.dump([{'puzzle': p, 'solution': s} for p, s in pz], open(out / f'puzzles_seed{a.seed}_g{a.givens}.json', 'w'))
    print(f'{len(pz)} puzzles written; mean givens = {np.mean([sum(1 for v in p if v) for p, _ in pz]):.1f}')
