"""Figures for the report: architecture, confusion matrix, reliability diagrams (std and over-confident CNN).
Usage: python make_figures.py [--seed 1]   (needs results/seed{S}/*/cnn_logits.npz)"""
import argparse

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from p7 import nn
from p7.data import RES, prepare, run_dir


def architecture():
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.axis('off')
    boxes = [('Handwritten\nmosaic 9x9', .02, '#eeeeee'), ('CNN\n(+ brightness rule)\nprobabilities', .20, '#cfe2f3'),
             ('AC-3\nconflict?', .38, '#fff2cc'), ('Search\nUCS / Greedy / A*\ncheapest corrections', .56, '#d9ead3'),
             ('CSP\nBT + FC + AC-3\nsolvable?', .75, '#fff2cc'), ('Solution\nor "unreadable"', .92, '#eeeeee')]
    for t, x, c in boxes:
        ax.text(x, .55, t, ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', fc=c, ec='k'))
    for x0, x1 in zip([.07, .26, .43, .63, .81], [.14, .32, .50, .68, .87]):
        ax.annotate('', xy=(x1, .55), xytext=(x0, .55), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(.56, .18), xytext=(.75, .18), arrowprops=dict(arrowstyle='->', color='C3'))
    ax.text(.655, .08, 'corrected puzzle tested; first solvable wins', ha='center', fontsize=8, color='C3')
    ax.text(.20, .95, 'p(digit) = cost of changing it', ha='center', fontsize=8, color='C0')
    ax.set_xlim(-.03, 1.0)
    ax.set_ylim(0, 1)
    fig.savefig(RES / 'architecture.png', dpi=150, bbox_inches='tight')


def classifier(seed):
    yte = prepare(seed)['test'][1]
    z = np.load(run_dir(seed, 'std') / 'cnn_logits.npz')
    cm = confusion_matrix(yte, nn.softmax(z['test'], float(z['T'])).argmax(1))
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].imshow(cm, cmap='Blues')
    ax[0].set_xticks(range(9), range(1, 10))
    ax[0].set_yticks(range(9), range(1, 10))
    ax[0].set_xlabel('predicted')
    ax[0].set_ylabel('true')
    ax[0].set_title('CNN confusion matrix (test)')
    for i in range(9):
        for j in range(9):
            if cm[i, j]:
                ax[0].text(j, i, cm[i, j], ha='center', va='center', fontsize=6,
                           color='white' if cm[i, j] > cm.max() / 2 else 'black')
    for a, variant, ttl in ((ax[1], 'std', 'std CNN'), (ax[2], 'overconf', 'over-confident CNN')):
        z = np.load(run_dir(seed, variant) / 'cnn_logits.npz')
        T = float(z['T'])
        for p, lab, mk in ((nn.softmax(z['test']), f'raw (ECE {nn.ece(nn.softmax(z["test"]), yte):.4f})', 'o-'),
                           (nn.softmax(z['test'], T), f'T={T:.2f} (ECE {nn.ece(nn.softmax(z["test"], T), yte):.4f})', 's-')):
            conf, acc = p.max(1), (p.argmax(1) == yte)
            edges = np.linspace(.3, 1, 11)
            xs, ys = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (conf > lo) & (conf <= hi)
                if m.sum() >= 5:
                    xs.append(conf[m].mean())
                    ys.append(acc[m].mean())
            a.plot(xs, ys, mk, label=lab)
        a.plot([.3, 1], [.3, 1], 'k--', lw=1)
        a.set_title(f'Reliability, {ttl}')
        a.set_xlabel('confidence')
        a.set_ylabel('accuracy')
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / 'classifier.png', dpi=140)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    architecture()
    classifier(a.seed)
