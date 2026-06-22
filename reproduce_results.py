#!/usr/bin/env python3
"""
Reproduce the main results of:
  "Improved Coarse to Fine Compound Facial Expression Classification
   Using Geometric and Appearance Based Features"

Loads the pre-computed 1035-d feature representation (1000-d GoogLeNet loss3-classifier
texture features + 35-d OpenFace action units) and reproduces, from a single consistent
run, all reported numbers and all six confusion matrices:

  baseline 50-class            (Fig. S1)
  Level-1 8-class              (Fig. 8)
  third-level group matrices   (Figs. 10-12)
  coarse-to-fine 50-class      (Fig. 13)

Protocol: sample-mixed 80:20 5-fold cross-validation; the third-level (coarse-to-fine)
result follows the paper's method, where each group classifier is evaluated on its own
fine-grained classes. Figures are written to ./figures/.

Usage:  python reproduce_results.py
"""
import os, time, warnings
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
FEAT = os.path.join(HERE, 'features')
FIG  = os.path.join(HERE, 'figures'); os.makedirs(FIG, exist_ok=True)

def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

X = np.load(os.path.join(FEAT, 'aligned_combined_X.npy'))
y = np.load(os.path.join(FEAT, 'aligned_combined_y.npy'))
Xs = StandardScaler().fit_transform(X)
log(f'Loaded {Xs.shape} (1000-d GoogLeNet + 35-d AU), {len(np.unique(y))} classes')

# Second-level groups (basic emotions 1..8 -> coarse-to-fine groups)
GC  = {0: list(range(1, 15)),                                   # G1 Angry + Contempt
       1: list(range(22, 29)) + list(range(43, 50)),           # G2 Fear + Surprise
       2: list(range(15, 22)) + list(range(29, 43)) + [50]}    # G3 Disgust+Happy+Sad+Neutral
C2G = {1:0, 2:0, 3:2, 4:1, 5:2, 6:2, 7:1, 8:2}
fine_to_coarse = lambda y: np.where(y <= 49, np.ceil(y/7).astype(int), 8)
def fine_to_group(y):
    g = np.full(len(y), 2, int); g[np.isin(y, GC[0])] = 0; g[np.isin(y, GC[1])] = 1; return g
mk = lambda: OneVsOneClassifier(LinearSVC(C=1.0, max_iter=10000), n_jobs=-1)

cm_base = np.zeros((50, 50), int); cm_l1 = np.zeros((8, 8), int)
cm_grp  = [np.zeros((len(GC[g]),)*2, int) for g in range(3)]; cm_ctf = np.zeros((50, 50), int)

skf = StratifiedKFold(5, shuffle=True, random_state=42)
for fold, (tr, te) in enumerate(skf.split(Xs, y), 1):
    log(f'fold {fold}/5 ...')
    Xtr, Xte, ytr, yte = Xs[tr], Xs[te], y[tr], y[te]
    pb = mk().fit(Xtr, ytr).predict(Xte)
    for t, p in zip(yte, pb): cm_base[t-1, p-1] += 1
    pl1 = mk().fit(Xtr, fine_to_coarse(ytr)).predict(Xte)
    for t, p in zip(fine_to_coarse(yte), pl1): cm_l1[t-1, p-1] += 1
    fsv, cmap = {}, {}
    for g, cl in GC.items():
        ca = np.array(sorted(cl)); cmap[g] = ca; im = {c: k for k, c in enumerate(ca)}
        m = np.isin(ytr, ca); fsv[g] = mk().fit(Xtr[m], np.array([im[c] for c in ytr[m]]))
    tg = fine_to_group(yte)
    for g in range(3):
        mm = tg == g; ca = cmap[g]; im = {c: k for k, c in enumerate(ca)}
        if mm.sum():
            pr = fsv[g].predict(Xte[mm])
            for tl, pl in zip(yte[mm], pr):
                cm_grp[g][im[int(tl)], int(pl)] += 1
                cm_ctf[int(tl)-1, int(ca[int(pl)])-1] += 1

def metrics(cm):
    tot = cm.sum(); acc = np.trace(cm)/tot*100; n = cm.shape[0]; P=[];R=[];F=[];S=[]
    for i in range(n):
        tp=cm[i,i]; cs=cm[:,i].sum(); rs=cm[i,:].sum()
        p=tp/cs if cs else 0; r=tp/rs if rs else 0; f=2*p*r/(p+r) if (p+r) else 0
        P.append(p);R.append(r);F.append(f);S.append(rs)
    S=np.array(S); w=S/S.sum()
    return acc, np.sum(w*P)*100, np.sum(w*R)*100, np.sum(w*F)*100

aB,pB,rB,fB = metrics(cm_base); aL,*_ = metrics(cm_l1); aC,pC,rC,fC = metrics(cm_ctf)
gacc = [np.trace(cm_grp[g])/cm_grp[g].sum()*100 for g in range(3)]

log('\n' + '='*60)
log('REPRODUCED RESULTS (sample-mixed 80:20 5-fold)')
log(f'  Baseline 50-class (Fig S1):   {aB:.2f}%  (P={pB:.2f} R={rB:.2f} F1={fB:.2f})')
log(f'  Level-1 8-class   (Fig 8):    {aL:.2f}%')
log(f'  Group 1/2/3       (Figs 10-12): {gacc[0]:.2f}% / {gacc[1]:.2f}% / {gacc[2]:.2f}%')
log(f'  Coarse-to-fine    (Fig 13):   {aC:.2f}%  (P={pC:.2f} R={rC:.2f} F1={fC:.2f})')
log(f'  Improvement from clustering:  +{aC-aB:.2f} points')
log('='*60)

def plot(mat, labels, title, out, figsize, fs, show_zero=False):
    n=len(labels); rs=mat.sum(axis=1,keepdims=True); norm=np.where(rs>0,mat/rs,0); bg=np.ones((n,n,3))
    for i in range(n):
        for j in range(n):
            v=norm[i,j]; bg[i,j]=[1-v*0.75,1-v*0.65,1.] if i==j else [1.,1-v*0.7,1-v*0.7]
    fig,ax=plt.subplots(figsize=figsize,dpi=300); ax.imshow(bg,aspect='auto',interpolation='nearest')
    for i in range(n):
        for j in range(n):
            if mat[i,j]>0 or show_zero:
                ax.text(j,i,str(mat[i,j]),ha='center',va='center',fontsize=fs,
                        color='black' if norm[i,j]<0.6 else 'white')
    for k in range(n+1): ax.axhline(k-.5,color='gray',lw=0.3); ax.axvline(k-.5,color='gray',lw=0.3)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels,rotation=90,fontsize=fs)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels,fontsize=fs)
    ax.set_xlabel('Predicted',fontsize=fs+1); ax.set_ylabel('True',fontsize=fs+1); ax.set_title(title,fontsize=fs+2)
    fig.tight_layout(); fig.savefig(out,dpi=300,bbox_inches='tight',facecolor='white'); plt.close(fig)
    log(f'  saved {os.path.relpath(out, HERE)}')

EMO=['Angry','Contempt','Disgust','Fear','Happy','Sadness','Surprise','Neutral']
l50=[str(i) for i in range(1,51)]
log('\nplotting figures ...')
plot(cm_base,l50,f'50-class baseline - {aB:.2f}%',os.path.join(FIG,'image6.png'),(18,16),5)
plot(cm_l1,EMO,f'Level-1 8-class - {aL:.2f}%',os.path.join(FIG,'image7.png'),(8,7),9,show_zero=True)
for g,fn in enumerate(['image9.png','image10.png','image11.png']):
    plot(cm_grp[g],[str(c) for c in sorted(GC[g])],f'Group {g+1} - {gacc[g]:.2f}%',
         os.path.join(FIG,fn),(12,10) if g<2 else (14,12),7)
plot(cm_ctf,l50,f'Coarse-to-fine - {aC:.2f}%',os.path.join(FIG,'image12.png'),(18,16),5)
log('Done.')
