"""Brightness rule + calibration tests (no trained model needed)."""
import numpy as np

from p7 import nn
from p7.data import BLANK_THR, Mosaic, load_mnist, read_mosaic


def test_normal_brightness_rule_finds_blank_cells():
    X, y, _, _ = load_mnist()
    imgs = np.zeros((81, 28, 28), 'float32')
    digits = X[:81].astype('float32') / 255
    given = np.arange(81) % 2 == 0
    imgs[given] = digits[given]
    m = Mosaic(imgs, np.zeros((81, 9)), np.zeros((81, 9)), [0] * 81, [0] * 81, [])
    g, _ = read_mosaic(m, predict_fn=lambda X_: np.tile(np.eye(9)[0], (len(X_), 1)))
    assert [v == 0 for v in g] == list(~given)


def test_normal_every_mnist_digit_image_is_brighter_than_the_blank_threshold():
    X, y, _, _ = load_mnist()
    assert (X[y > 0].reshape(-1, 784).mean(1) / 255 > BLANK_THR).all()


def test_normal_temperature_scaling_fixes_overconfident_logits():
    rng = np.random.RandomState(0)
    n = 4000
    y = rng.randint(0, 9, n)
    logits = rng.normal(0, 1, (n, 9)) * 3
    logits[np.arange(n), y] += 4                      # noisy but informative
    logits *= 4                                       # artificially over-confident
    T = nn.fit_temperature(logits, y)
    assert T > 2
    assert nn.ece(nn.softmax(logits, T), y) < nn.ece(nn.softmax(logits), y)


def test_edge_ece_of_perfect_one_hot_predictions_is_zero():
    y = np.arange(9)
    assert nn.ece(np.eye(9), y) == 0.0
