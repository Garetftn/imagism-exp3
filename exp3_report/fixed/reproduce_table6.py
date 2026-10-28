"""Reproduce Table 4 (poetry rows) and Table 6 of Bizzoni et al. (2026) from the cached
Experiment III files, with and without removing duplicate lines.

Run from inside source/ (the paper's repo):
    python ../fixed/reproduce_table6.py
Needs numpy, pandas, scipy. No CLIP or HuggingFace access needed.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu

PAPER_T = {'norm': -6.69, 'entropy': 7.19, 'sparsity': 6.44,
           'imag_norm': 5.07, 'visual_norm': 4.98, 'concrete_norm': 7.93}
PAPER_U = {'norm': 567900, 'entropy': 783363, 'sparsity': 773000,
           'imag_norm': 734321.5, 'visual_norm': 749496, 'concrete_norm': 796327}


def load(tag):
    emb = np.array(json.load(open(f'data/embeddings/{tag}_sentences.json')))
    meas = pd.read_json(f'data/measures/{tag}_sentences.json')
    # identical lines give identical CLIP embeddings; keep first occurrence
    _, first = np.unique(emb.round(6), axis=0, return_index=True)
    return meas, meas.iloc[np.sort(first)]


imag_all, imag_dedup = load('imagists')
love_all, love_dedup = load('love')

for label, a, b in [('all cached lines', imag_all, love_all),
                    ('duplicates removed', imag_dedup, love_dedup)]:
    print(f'\n== {label}: Imagist {len(a)} lines, Love {len(b)} lines (paper: 1195, 1126)')
    print(f'{"measure":15}{"t":>8}{"paper t":>9}{"U":>12}{"paper U":>12}')
    for m in PAPER_T:
        x, y = a[m].dropna(), b[m].dropna()
        print(f'{m:15}{ttest_ind(x, y).statistic:8.2f}{PAPER_T[m]:9.2f}'
              f'{mannwhitneyu(x, y).statistic:12.1f}{PAPER_U[m]:12.1f}')
    for name, d in [('Imagist', a), ('Love', b)]:
        print(f'  Table 4 {name:8} imag {d.imag.mean():.1f} ± {d.imag.std():.1f}, '
              f'normalized {d.imag_norm.mean():.1f} ± {d.imag_norm.std():.1f}')
