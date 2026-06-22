#!/usr/bin/env python3
"""
Build the 1035-d feature representation from the raw iCV-MEFED data.
This script is provided for full transparency; reviewers who only wish to
reproduce the reported numbers can skip it and use the pre-computed files in
./features/ directly with reproduce_results.py.

Requirements (not included in this repository due to size / licensing):
  - The iCV-MEFED dataset archive, which ships the OpenFace-aligned face crops at
        iCV MEFED/emotion/openface/{training,validation}/frame_det_*.bmp
    in the same order as the filename lists below.
  - The OpenFace action-unit CSVs and the ordered filename lists:
        features/training.csv, features/validation.csv
        features/train_filenames.txt, features/val_filenames.txt

Pipeline:
  aligned face (112x112) -> resize 224 -> GoogLeNet (ImageNet, frozen)
  -> 1000-d loss3-classifier vector ; concatenated with 35-d OpenFace AU = 1035-d.

Set DATASET_DIR below to the extracted dataset path, then run:  python extract_features.py
"""
import os, re, math, glob, warnings
import numpy as np, pandas as pd
import torch
from torchvision import models, transforms
from PIL import Image
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
FEAT = os.path.join(HERE, 'features')
DATASET_DIR = os.environ.get('ICV_MEFED_DIR', os.path.expanduser('~/iCV MEFED/emotion/openface'))

def fname_to_label(fn): return math.ceil(int(re.search(r'_(\d+)\.jpg', fn).group(1)) / 5)
def read_lines(p):
    with open(p) as f: return [l.strip() for l in f if l.strip()]

# GoogLeNet, ImageNet-pretrained, used frozen in eval mode; features at the final FC (loss3-classifier).
device = torch.device('mps' if torch.backends.mps.is_available()
                      else 'cuda' if torch.cuda.is_available() else 'cpu')
model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1).eval().to(device)
buf = []
model.fc.register_forward_hook(lambda m, i, o: buf.append(o.detach().cpu().numpy()))
tf = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                         transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])

def extract_split(split, fname_list):
    fnames = read_lines(fname_list)
    adir = os.path.join(DATASET_DIR, split)
    X, Y, FN = [], [], []
    batch, meta = [], []
    def flush():
        if not batch: return
        with torch.no_grad():
            model(torch.stack(batch).to(device))
        emb = buf[-1]
        for k, (lbl, fn) in enumerate(meta):
            X.append(emb[k]); Y.append(lbl); FN.append(fn)
        batch.clear(); meta.clear(); buf.clear()
    for idx, fn in enumerate(fnames):
        bmp = os.path.join(adir, f'frame_det_{idx:06d}.bmp')
        if not os.path.exists(bmp): continue
        lbl = fname_to_label(fn)
        if lbl > 50: continue
        batch.append(tf(Image.open(bmp).convert('RGB'))); meta.append((lbl, fn))
        if len(batch) >= 64: flush()
    flush()
    return np.array(X), np.array(Y), np.array(FN)

print('Extracting GoogLeNet features (training, validation) ...')
Xtr, ytr, fntr = extract_split('training', os.path.join(FEAT, 'train_filenames.txt'))
Xvl, yvl, fnvl = extract_split('validation', os.path.join(FEAT, 'val_filenames.txt'))
gX = np.vstack([Xtr, Xvl]); gfn = np.concatenate([fntr, fnvl])

def load_au(csv, fnp):
    df = pd.read_csv(csv); fns = read_lines(fnp)
    lab = np.array([fname_to_label(x) for x in fns]); keep = lab <= 50
    cols = [c for c in df.columns if 'AU' in c]
    return df[cols].values.astype(np.float32)[keep], np.array(fns)[keep]

a1, f1 = load_au(os.path.join(FEAT, 'training.csv'), os.path.join(FEAT, 'train_filenames.txt'))
a2, f2 = load_au(os.path.join(FEAT, 'validation.csv'), os.path.join(FEAT, 'val_filenames.txt'))
auX = np.vstack([a1, a2]); aufn = np.concatenate([f1, f2])

rmap = {fn: i for i, fn in enumerate(gfn)}; ri, ai = [], []
for i, fn in enumerate(aufn):
    if fn in rmap: ri.append(rmap[fn]); ai.append(i)
ri, ai = np.array(ri), np.array(ai)
X = np.hstack([gX[ri], auX[ai]]).astype(np.float32)
fn = gfn[ri]
y = np.array([fname_to_label(f) for f in fn])
subj = np.array([re.match(r'Sample_(\d+)_', f).group(1) for f in fn])

np.save(os.path.join(FEAT, 'aligned_combined_X.npy'), X)
np.save(os.path.join(FEAT, 'aligned_combined_y.npy'), y)
np.save(os.path.join(FEAT, 'aligned_combined_subj.npy'), subj)
np.save(os.path.join(FEAT, 'aligned_combined_fnames.npy'), fn)
print('Saved features/aligned_combined_*.npy', X.shape)
