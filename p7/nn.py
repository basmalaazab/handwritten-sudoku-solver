"""Neural network: small CNN (Keras) on MNIST digits 1-9 + logistic-regression baseline + calibration.

variant 'std'      : Conv16-Conv32-Dense64, dropout 0.3, 5 epochs (the main model)
variant 'overconf' : no dropout, Dense128, 15 epochs  (deliberately over-confident, used to test temperature scaling)
"""
import json

import numpy as np

from .data import prepare, run_dir


# ------------------------------------------------------------------ calibration
def softmax(z, T=1.0):
    z = z / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())


def ece(p, y, bins=15):
    conf, acc = p.max(1), (p.argmax(1) == y).astype(float)
    edges, out = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(out)


def fit_temperature(logits, y):
    from scipy.optimize import minimize_scalar
    return float(minimize_scalar(lambda T: nll(softmax(logits, T), y), bounds=(0.05, 10), method='bounded').x)


# ------------------------------------------------------------------ model
def build_cnn(variant='std'):
    from tensorflow import keras
    L = keras.layers
    layers = [keras.Input((28, 28, 1)), L.Conv2D(16, 3, activation='relu'), L.MaxPooling2D(),
              L.Conv2D(32, 3, activation='relu'), L.MaxPooling2D(), L.Flatten()]
    if variant == 'std':
        layers += [L.Dropout(0.3), L.Dense(64, activation='relu')]
    else:
        layers += [L.Dense(128, activation='relu')]
    layers += [L.Dense(9)]                               # logits
    m = keras.Sequential(layers)
    m.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    return m


def load_model(rd):
    from tensorflow import keras
    return keras.models.load_model(rd / 'cnn.keras')


def predict_logits(model, X, bs=512):
    return model.predict(X[..., None] if X.ndim == 3 else X, batch_size=bs, verbose=0)


def train(seed=0, variant='std', baseline=True, verbose=0):
    """Train + calibrate + (optionally) fit the logistic-regression baseline. Writes into results/seed*/variant/."""
    import tensorflow as tf
    from sklearn.linear_model import LogisticRegression
    rd = run_dir(seed, variant)
    tf.keras.utils.set_random_seed(seed)
    D = prepare(seed)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = D['train'], D['val'], D['test']
    model = build_cnn(variant)
    model.fit(Xtr[..., None], ytr, validation_data=(Xva[..., None], yva),
              epochs=5 if variant == 'std' else 15, batch_size=128, verbose=verbose)
    model.save(rd / 'cnn.keras')
    lv, lt = predict_logits(model, Xva), predict_logits(model, Xte)
    T = fit_temperature(lv, yva)
    np.savez(rd / 'cnn_logits.npz', val=lv, test=lt, T=T)
    rep = {'sizes': [len(Xtr), len(Xva), len(Xte)], 'temperature': T}
    for name, p in (('cnn_raw', softmax(lt)), ('cnn_temp_scaled', softmax(lt, T))):
        rep[name] = {'test_acc': float((p.argmax(1) == yte).mean()), 'ECE': ece(p, yte), 'NLL': nll(p, yte)}
    if baseline:
        lr = LogisticRegression(max_iter=150)
        lr.fit(Xtr.reshape(len(Xtr), -1), ytr)
        plr = lr.predict_proba(Xte.reshape(len(Xte), -1))
        rep['logreg'] = {'test_acc': float((plr.argmax(1) == yte).mean()), 'ECE': ece(plr, yte), 'NLL': nll(plr, yte)}
    from sklearn.metrics import confusion_matrix
    rep['confusion_matrix_cnn_test'] = confusion_matrix(yte, softmax(lt, T).argmax(1)).tolist()
    json.dump(rep, open(rd / 'classifier_report.json', 'w'), indent=2)
    return rep
