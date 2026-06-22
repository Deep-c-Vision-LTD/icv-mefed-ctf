#!/usr/bin/env python3
"""
Robustness check: repeat the baseline + coarse-to-fine pipeline over several
random 80:20 5-fold partitions and report mean +/- standard deviation.

Usage:  python multiseed.py
"""
import os, time, warnings
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
FEAT = os.path.join(HERE, 'features')
def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

X = np.load(os.path.join(FEAT, 'aligned_combined_X.npy'))
y = np.load(os.path.join(FEAT, 'aligned_combined_y.npy'))
Xs = StandardScaler().fit_transform(X)

GC  = {0: list(range(1, 15)), 1: list(range(22, 29)) + list(range(43, 50)),
       2: list(range(15, 22)) + list(range(29, 43)) + [50]}
fine_to_coarse = lambda y: np.where(y <= 49, np.ceil(y/7).astype(int), 8)
def fine_to_group(y):
    g = np.full(len(y), 2, int); g[np.isin(y, GC[0])] = 0; g[np.isin(y, GC[1])] = 1; return g
mk = lambda: OneVsOneClassifier(LinearSVC(C=1.0, max_iter=10000), n_jobs=-1)

SEEDS = [0, 1, 7, 42, 123]
base_accs, ctf_accs = [], []
log(f'Running {len(SEEDS)} seeds ...')
for s in SEEDS:
    skf = StratifiedKFold(5, shuffle=True, random_state=s)
    bh = ch = tot = 0
    for tr, te in skf.split(Xs, y):
        Xtr, Xte, ytr, yte = Xs[tr], Xs[te], y[tr], y[te]
        bh += (mk().fit(Xtr, ytr).predict(Xte) == yte).sum()
        fsv, cmap = {}, {}
        for g, cl in GC.items():
            ca = np.array(sorted(cl)); cmap[g] = ca; im = {c: k for k, c in enumerate(ca)}
            m = np.isin(ytr, ca); fsv[g] = mk().fit(Xtr[m], np.array([im[c] for c in ytr[m]]))
        tg = fine_to_group(yte); pred = np.zeros(len(yte), int)
        for g in range(3):
            mm = tg == g
            if mm.sum(): pred[mm] = cmap[g][np.clip(fsv[g].predict(Xte[mm]), 0, len(cmap[g])-1)]
        ch += (pred == yte).sum(); tot += len(yte)
    base_accs.append(bh/tot*100); ctf_accs.append(ch/tot*100)
    log(f'  seed {s:>3}: baseline={base_accs[-1]:.2f}%  coarse-to-fine={ctf_accs[-1]:.2f}%')

b, c = np.array(base_accs), np.array(ctf_accs)
log('\n' + '='*52)
log(f'Baseline       : {b.mean():.2f}% +/- {b.std():.2f}%   [{b.min():.2f}-{b.max():.2f}]')
log(f'Coarse-to-fine : {c.mean():.2f}% +/- {c.std():.2f}%   [{c.min():.2f}-{c.max():.2f}]')
log(f'Improvement    : +{c.mean()-b.mean():.2f} points')
log('='*52)
