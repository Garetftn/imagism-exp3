"""Derived tables and figures for the Experiment III report, from the cached files only.

Run from the repository root (the paper's repo):
    python exp3_report/fixed/make_figures.py
Writes exp3_report/data/derived/*.csv and exp3_report/figs/*.png. Needs numpy, pandas, scipy, matplotlib.
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu, spearmanr, gaussian_kde

OUT_DATA, OUT_FIGS = 'exp3_report/data/derived', 'exp3_report/figs'
os.makedirs(OUT_DATA, exist_ok=True)
os.makedirs(OUT_FIGS, exist_ok=True)

DV = ['norm', 'entropy', 'sparsity', 'imag_norm', 'visual_norm', 'concrete_norm']
MERGE = {  # JSON entries that are parts of one poem (same as exp3_validity_checks.py)
    **{t: 'LONDON EXCURSION' for t in ['APPROACH', 'ARRIVAL', 'WALK', 'BUS-STOP', 'TRANSPOSITION',
                                       'PERIPETEIA', 'MID-FLIGHT', 'STATION']},
    **{t: 'EPIGRAMS' for t in ['A GIRL', 'NEW LOVE', 'OCTOBER']},
    'CATULLUS.': 'DAISY',
}
IMAGIST, LOVE = '#2a78d6', '#eb6834'
INK, MUTED = '#0b0b0b', '#52514e'


def imagist_text():
    im = json.load(open('data/some_imagists.json'))
    rows = []
    for poet, works in im.items():
        for w in works:
            for s in w['poem'].replace('\n', ' ').replace('  ', ' ').strip().split('*'):
                s = s.strip()
                if s and not s.isdigit():
                    rows.append((poet, MERGE.get(w['title'], w['title']), s))
    return pd.DataFrame(rows, columns=['poet', 'poem', 'text'])


def load(tag):
    meas = pd.read_json(f'data/measures/{tag}_sentences.json')
    emb = np.array(json.load(open(f'data/embeddings/{tag}_sentences.json')))
    _, first = np.unique(emb.round(6), axis=0, return_index=True)
    meas.insert(0, 'kept_after_dedup', False)
    meas.loc[np.sort(first), 'kept_after_dedup'] = True
    meas.insert(0, 'row', range(len(meas)))
    return meas


im = load('imagists')
im = pd.concat([im.iloc[:, :2], imagist_text(), im.iloc[:, 2:]], axis=1)
lv = load('love')
cols = lambda df: [c for c in df.columns if c in ['row', 'kept_after_dedup', 'poet', 'poem', 'text'] + DV]
im[cols(im)].round(4).to_csv(f'{OUT_DATA}/imagist_lines.csv', index=False)
lv[cols(lv)].round(4).to_csv(f'{OUT_DATA}/love_lines.csv', index=False)

im, lv = im[im.kept_after_dedup], lv[lv.kept_after_dedup]

# Table 6 with what the paper leaves out: means, Cohen's d, probability of superiority
rows = []
for m in DV:
    x, y = im[m].dropna(), lv[m].dropna()
    sp = np.sqrt(((len(x) - 1) * x.var() + (len(y) - 1) * y.var()) / (len(x) + len(y) - 2))
    t = ttest_ind(x, y)
    u = mannwhitneyu(x, y)
    rows.append(dict(measure=m, n_imagist=len(x), n_love=len(y), mean_imagist=x.mean(), mean_love=y.mean(),
                     cohen_d=(x.mean() - y.mean()) / sp, t=t.statistic, p_t=t.pvalue, U=u.statistic,
                     prob_superiority=u.statistic / (len(x) * len(y)), p_U=u.pvalue))
pd.DataFrame(rows).to_csv(f'{OUT_DATA}/table6_effect_sizes.csv', index=False, float_format='%.4g')

# The two reference lines of Figure 3 are Love lines (hard-coded in experiment_3.py); the Love texts
# are not cached, so locate them by their published values
homespun = lv.loc[(lv.imag_norm - 549.75).abs() < 0.01].iloc[0]
insidious = lv.loc[(lv.imag_norm - 217.0).abs() < 0.01].iloc[0]
print('reference rows (love_lines.csv row):', homespun.row, insidious.row)

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False})

# Figure 3 reproduced: same six panels as the paper, curves only (no histogram bars), and the two
# reference lines drawn in neutral ink so they are not mistaken for the two corpora
fig, axes = plt.subplots(2, 3, figsize=(10, 5.6))
for ax, m in zip(axes.flat, DV):
    lo = min(im[m].min(), lv[m].min())
    hi = max(im[m].max(), lv[m].max())
    xs = np.linspace(lo, hi, 300)
    for df, c, lab in [(im, IMAGIST, 'Imagist'), (lv, LOVE, 'Love')]:
        ax.plot(xs, gaussian_kde(df[m].dropna())(xs), color=c, lw=2, label=lab)
        ax.axvline(df[m].mean(), color=c, lw=1, alpha=0.5)
    ax.axvline(homespun[m], color=INK, lw=1.2, ls='--')
    ax.axvline(insidious[m], color=INK, lw=1.2, ls=':')
    ax.set_title(m, color=INK, fontsize=10, loc='left')
    ax.set_yticks([])
    ax.grid(False)
axes[0, 0].legend(frameon=False, fontsize=8, loc='upper left')
fig.text(0.5, 0.005, 'Thin solid lines: group means.  Dashed: "Homespun, dyed butternuts dark gold color."  '
         'Dotted: "Of insidious intent"  (both are Love lines)', ha='center', color=MUTED, fontsize=8)
fig.suptitle('Figure 3 reproduced: Imagist (n = 1,195) vs Love (n = 1,126) lines, duplicates removed',
             x=0.01, ha='left', color=INK, fontsize=11)
fig.tight_layout(rect=(0, 0.03, 1, 0.96))
fig.savefig(f'{OUT_FIGS}/figure3_reproduced.png', dpi=150)

# Within-corpus link between norm and dictionary imageability (the plan's claim C3)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True, sharey=True)
for ax, df, c, lab in [(axes[0], im, IMAGIST, 'Imagist'), (axes[1], lv, LOVE, 'Love')]:
    d = df[['norm', 'imag_norm']].dropna()
    rho = spearmanr(d.norm, d.imag_norm).statistic
    ax.scatter(d.imag_norm, d.norm, s=8, color=c, alpha=0.35, linewidths=0)
    ax.set_title(f'{lab} lines   Spearman ρ = {rho:.2f}   (n = {len(d)})', loc='left', color=INK, fontsize=10)
    ax.set_xlabel('MRC imageability (mean over covered words)')
axes[0].set_ylabel('CLIP embedding norm')
fig.tight_layout()
fig.savefig(f'{OUT_FIGS}/norm_vs_imageability_within_corpus.png', dpi=150)
print('done')
