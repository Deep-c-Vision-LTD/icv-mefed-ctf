#!/usr/bin/env python3
"""
Flexible evaluator on the saved feature representation.
Reports baseline / coarse-to-fine (hard cascade) / coarse-to-fine (group-known) accuracy.

Examples:
  python svm_eval.py                    # sample-mixed and subject-independent, LinearSVC C=1
  python svm_eval.py --split mixed      # sample-mixed only (paper protocol)
  python svm_eval.py --split subject    # subject-independent (cross-subject generalisation)
  python svm_eval.py --C 10
  python svm_eval.py --kernel rbf
  python svm_eval.py --softmax          # softmax the 1000-d GoogLeNet block first
"""
import os, argparse, warnings
import numpy as np
from scipy.special import softmax as _softmax
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__)); FEAT = os.path.join(HERE, 'features')
ap = argparse.ArgumentParser()
ap.add_argument('--split', choices=['mixed', 'subject', 'both'], default='both')
ap.add_argument('--C', type=float, default=1.0)
ap.add_argument('--kernel', choices=['linear', 'rbf'], default='linear')
ap.add_argument('--softmax', action='store_true')
ap.add_argument('--seed', type=int, default=42)
args = ap.parse_args()

X = np.load(os.path.join(FEAT, 'aligned_combined_X.npy'))
y = np.load(os.path.join(FEAT, 'aligned_combined_y.npy'))
subj = np.load(os.path.join(FEAT, 'aligned_combined_subj.npy'))
if args.softmax:
    X = np.hstack([_softmax(X[:, :1000], axis=1), X[:, 1000:]])
Xs = StandardScaler().fit_transform(X)
print(f'Loaded {Xs.shape} | kernel={args.kernel} C={args.C} softmax={args.softmax}')

GC = {0: list(range(1, 15)), 1: list(range(22, 29)) + list(range(43, 50)),
      2: list(range(15, 22)) + list(range(29, 43)) + [50]}
C2G = {1:0, 2:0, 3:2, 4:1, 5:2, 6:2, 7:1, 8:2}
fine_to_coarse = lambda y: np.where(y <= 49, np.ceil(y/7).astype(int), 8)
def fine_to_group(y):
    g = np.full(len(y), 2, int); g[np.isin(y, GC[0])] = 0; g[np.isin(y, GC[1])] = 1; return g
def mk():
    base = SVC(kernel='rbf', C=args.C, gamma='scale') if args.kernel == 'rbf' else LinearSVC(C=args.C, max_iter=10000)
    return OneVsOneClassifier(base, n_jobs=-1)

def evaluate(folds, name):
    base = hard = orac = tot = 0
    for tr, te in folds:
        Xtr, Xte, ytr, yte = Xs[tr], Xs[te], y[tr], y[te]
        base += (mk().fit(Xtr, ytr).predict(Xte) == yte).sum()
        pl1 = mk().fit(Xtr, fine_to_coarse(ytr)).predict(Xte)
        fsv, cm = {}, {}
        for g, cl in GC.items():
            ca = np.array(sorted(cl)); cm[g] = ca; im = {c: k for k, c in enumerate(ca)}
            m = np.isin(ytr, ca); fsv[g] = mk().fit(Xtr[m], np.array([im[c] for c in ytr[m]]))
        pg = np.array([C2G.get(int(c), 2) for c in pl1]); hp = np.zeros(len(yte), int)
        for g in range(3):
            mm = pg == g
            if mm.sum(): hp[mm] = cm[g][np.clip(fsv[g].predict(Xte[mm]), 0, len(cm[g])-1)]
        hard += (hp == yte).sum()
        tg = fine_to_group(yte); op = np.zeros(len(yte), int)
        for g in range(3):
            mm = tg == g
            if mm.sum(): op[mm] = cm[g][np.clip(fsv[g].predict(Xte[mm]), 0, len(cm[g])-1)]
        orac += (op == yte).sum(); tot += len(yte)
    print(f'\n  [{name}]  baseline={base/tot*100:.2f}%  '
          f'CTF(hard-cascade)={hard/tot*100:.2f}%  CTF(group-known)={orac/tot*100:.2f}%')

if args.split in ('mixed', 'both'):
    skf = StratifiedKFold(5, shuffle=True, random_state=args.seed)
    evaluate(list(skf.split(Xs, y)), 'sample-mixed (paper protocol)')
if args.split in ('subject', 'both'):
    su = np.array(sorted(set(subj))); rng = np.random.RandomState(args.seed); rng.shuffle(su)
    sf = np.array_split(su, 5)
    folds = [(np.where(~np.isin(subj, sf[f]))[0], np.where(np.isin(subj, sf[f]))[0]) for f in range(5)]
    evaluate(folds, 'subject-independent')
