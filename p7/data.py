"""Data layer: MNIST (digits 1-9), digit bank with the documented injection rule, mosaics.

REAL data      : MNIST digit images and labels.
GENERATED data : Sudoku puzzles/solutions (csp.make_puzzle), 9x9 mosaics, augmented 'hard' digit images,
                 blank-cell images.  See docs/DATA_DICTIONARY.md.
"""
import gzip
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
from scipy.ndimage import affine_transform

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
RES = ROOT / 'results'

FILES = {'train-images-idx3-ubyte.gz': 'f68b3c2dcbeaaa9fbdd348bbdeb94873',
         'train-labels-idx1-ubyte.gz': 'd53e105ee54ea40749a09fcbcd1e9432',
         't10k-images-idx3-ubyte.gz': '9fb629c4189551a2d022fa330f9573f3',
         't10k-labels-idx1-ubyte.gz': 'ec29112dd5afa0611ce80d1b7f02629c'}
MIRRORS = ['http://yann.lecun.com/exdb/mnist/',
           'https://ossci-datasets.s3.amazonaws.com/mnist/',
           'https://raw.githubusercontent.com/golbin/TensorFlow-MNIST/master/mnist/data/']

HARD_PER_DIGIT = 60
MAX_P, MIN_P_TRUE, MAX_RANK = 0.80, 0.05, 3
BLANK_THR = 0.015            # brightness rule: mean pixel < BLANK_THR  =>  blank cell


def run_dir(seed, variant='std'):
    d = RES / f'seed{seed}' / variant
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------ download + load
def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def download_mnist(d=DATA):
    d.mkdir(parents=True, exist_ok=True)
    for name, want in FILES.items():
        p = d / name
        if p.exists() and md5(p) == want:
            continue
        for base in MIRRORS:
            try:
                urllib.request.urlretrieve(base + name, p)
                if md5(p) == want:
                    print(f'downloaded {name} from {base}')
                    break
                print(f'checksum mismatch from {base}')
            except Exception as e:                       # noqa: BLE001
                print(f'{base} failed: {e}')
        else:
            raise RuntimeError(f'could not download {name}; put it in {d}')
    return {n: md5(d / n) for n in FILES}


def _imgs(path):
    b = gzip.open(path).read()
    return np.frombuffer(b, np.uint8, offset=16).reshape(int.from_bytes(b[4:8], 'big'), 28, 28)


def _labs(path):
    return np.frombuffer(gzip.open(path).read(), np.uint8, offset=8)


def load_mnist(d=DATA):
    return (_imgs(d / 'train-images-idx3-ubyte.gz'), _labs(d / 'train-labels-idx1-ubyte.gz'),
            _imgs(d / 't10k-images-idx3-ubyte.gz'), _labs(d / 't10k-labels-idx1-ubyte.gz'))


def prepare(seed=0):
    """Pixels scaled to 0-1; 50,000 train / 10,000 validation split of the 60,000 training images
    (seeded permutation); test set untouched; digit 0 dropped afterwards; labels = digit-1 (0..8)."""
    Xtr, ytr, Xte, yte = load_mnist()
    perm = np.random.RandomState(seed).permutation(len(Xtr))
    tr, va = perm[:50000], perm[50000:]

    def f(X, y):
        m = y > 0
        return X[m].astype('float32') / 255.0, (y[m] - 1).astype('int64')

    return {'train': f(Xtr[tr], ytr[tr]), 'val': f(Xtr[va], ytr[va]), 'test': f(Xte, yte)}


# ------------------------------------------------------------------ hard-digit rule
def augment(img, rng):
    ang = np.deg2rad(rng.uniform(15, 45) * rng.choice([-1, 1]))
    sh, sc = rng.uniform(-0.3, 0.3), rng.uniform(0.85, 1.1)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    M = R @ (np.array([[1, sh], [0, 1]]) * sc)
    ctr = np.array([13.5, 13.5])
    out = affine_transform(img, M, offset=ctr - M @ ctr, order=1)
    if rng.random() < 0.6:
        r, c = rng.integers(0, 20, 2)
        out[r:r + 8, c:c + 8] = 0
    if rng.random() < 0.5:
        out = out + rng.normal(0, 0.1, out.shape)
    return np.clip(out, 0, 1).astype('float32')


def accept_hard(p, d):
    """Rule: argmax != d, rank of d <= 3, max(p) <= 0.80, p[d] >= 0.05  (p calibrated, d 0-based)."""
    rank = np.argmax(np.argsort(-p, axis=1) == d, axis=1) + 1
    return (p.argmax(1) != d) & (rank <= MAX_RANK) & (p.max(1) <= MAX_P) & (p[:, d] >= MIN_P_TRUE)


def build_bank(seed=0, variant='std'):
    from . import nn
    rd = run_dir(seed, variant)
    rng = np.random.default_rng(seed)
    Xte, yte = prepare(seed)['test']
    z = np.load(rd / 'cnn_logits.npz')
    T = float(z['T'])
    P_cal, P_raw = nn.softmax(z['test'], T), nn.softmax(z['test'])
    assert (Xte.reshape(len(Xte), -1).mean(1) > BLANK_THR).all(), 'blank threshold too high'
    model = nn.load_model(rd)
    H = {k: [] for k in ('img', 'lab', 'cal', 'raw', 'src')}
    for d in range(9):
        idx = np.where(yte == d)[0]
        nat = idx[accept_hard(P_cal[idx], d)]
        for i in nat:
            for k, v in zip(H, (Xte[i], d, P_cal[i], P_raw[i], 0)):
                H[k].append(v)
        got, tries = len(nat), 0
        while got < HARD_PER_DIGIT and tries < 40000:
            batch = np.stack([augment(Xte[i], rng) for i in rng.choice(idx, 1000)])
            lg = nn.predict_logits(model, batch)
            pc = nn.softmax(lg, T)
            for k in np.where(accept_hard(pc, d))[0][:HARD_PER_DIGIT - got]:
                for key, v in zip(H, (batch[k], d, pc[k], nn.softmax(lg[k:k + 1])[0], 1)):
                    H[key].append(v)
                got += 1
            tries += 1000
    np.savez_compressed(rd / 'bank.npz', T=T, Xte=Xte, yte=yte, P_cal=P_cal, P_raw=P_raw,
                        H_img=np.stack(H['img']), H_lab=np.array(H['lab']), H_cal=np.stack(H['cal']),
                        H_raw=np.stack(H['raw']), H_src=np.array(H['src']), seed=seed)
    return rd / 'bank.npz'


class Bank:
    def __init__(self, seed=0, variant='std'):
        z = np.load(run_dir(seed, variant) / 'bank.npz')
        self.T = float(z['T'])
        self.Xte, self.yte, self.P_cal, self.P_raw = z['Xte'], z['yte'], z['P_cal'], z['P_raw']
        self.H_img, self.H_lab, self.H_cal, self.H_raw, self.H_src = (
            z['H_img'], z['H_lab'], z['H_cal'], z['H_raw'], z['H_src'])
        ok = self.P_cal.argmax(1) == self.yte
        self.easy = {d: np.where((self.yte == d - 1) & ok)[0] for d in range(1, 10)}
        self.anyimg = {d: np.where(self.yte == d - 1)[0] for d in range(1, 10)}
        self.hard = {d: np.where(self.H_lab == d - 1)[0] for d in range(1, 10)}


# ------------------------------------------------------------------ mosaics
class Mosaic:
    def __init__(self, images, probs_cal, probs_raw, puzzle, solution, injected):
        self.images, self.probs_cal, self.probs_raw = images, probs_cal, probs_raw
        self.puzzle, self.solution, self.injected = list(puzzle), list(solution), list(injected)

    def image(self):
        return self.images.reshape(9, 9, 28, 28).transpose(0, 2, 1, 3).reshape(252, 252)


def build_mosaic(puzzle, solution, bank, rng, k=0, natural=False):
    """rng: random.Random.  k hard digits injected among the givens.
    natural=True: background digits are random test images of the right digit (natural NN errors allowed);
    otherwise they are images the calibrated CNN reads correctly, so misread level == k exactly."""
    givens = [i for i, v in enumerate(puzzle) if v]
    inj = set(rng.sample(givens, k)) if k else set()
    imgs = np.zeros((81, 28, 28), 'float32')
    pc, pr = np.zeros((81, 9)), np.zeros((81, 9))
    for i, v in enumerate(puzzle):
        if not v:
            imgs[i] = np.abs(np.random.RandomState(rng.randrange(2 ** 31)).normal(0, 0.002, (28, 28)))
        elif i in inj:
            j = rng.choice(list(bank.hard[v]))
            imgs[i], pc[i], pr[i] = bank.H_img[j], bank.H_cal[j], bank.H_raw[j]
        else:
            j = rng.choice(list((bank.anyimg if natural else bank.easy)[v]))
            imgs[i], pc[i], pr[i] = bank.Xte[j], bank.P_cal[j], bank.P_raw[j]
    return Mosaic(imgs, pc, pr, puzzle, solution, sorted(inj))


def read_mosaic(m, source='cal', predict_fn=None):
    """Brightness rule finds blank cells; the NN gives probabilities for the others.
    Returns (read_grid, P) with P (81,9) (zero rows for blank cells)."""
    blank = m.images.reshape(81, -1).mean(1) < BLANK_THR
    if predict_fn is not None:
        P = np.zeros((81, 9))
        if (~blank).any():
            P[~blank] = predict_fn(m.images[~blank])
    else:
        P = (m.probs_cal if source == 'cal' else m.probs_raw).copy()
        P[blank] = 0
    return [0 if blank[i] else int(P[i].argmax()) + 1 for i in range(81)], P
