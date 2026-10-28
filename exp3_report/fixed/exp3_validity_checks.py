"""Experiment III claim checks on the cached files (no CLIP / HuggingFace needed).

Run from inside source/ (the paper's repo):
    python ../fixed/exp3_validity_checks.py
Needs numpy, pandas, scipy, statsmodels.

Sections:
  A  effect sizes for Table 6 (the paper reports only t and U)
  B  Imagist lines by poet and by poem (line -> poem mapping rebuilt from some_imagists.json)
  C  lines nested in poems: ICC and a poem-level test on the Imagist side
  D  leave-one-poet-out and "drop D. H. Lawrence" (he is in BOTH corpora per the paper's Tables 7/8)
  E  asymmetric line filter: Love drops lines <= 5 chars, Imagist does not
  F  "genre separability under lexical control": does norm separate the groups beyond the dictionaries?
  G  does norm track imageability *within* a corpus?
The Love texts are not cached, so Love lines cannot be grouped by poem or author here.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, spearmanr
import statsmodels.formula.api as smf

MERGE = {  # JSON entries that are parts of one poem (see issue7_data_vs_paper.md)
    **{t: 'LONDON EXCURSION' for t in ['APPROACH', 'ARRIVAL', 'WALK', 'BUS-STOP', 'TRANSPOSITION',
                                       'PERIPETEIA', 'MID-FLIGHT', 'STATION']},
    **{t: 'EPIGRAMS' for t in ['A GIRL', 'NEW LOVE', 'OCTOBER']},
    'CATULLUS.': 'DAISY',
}
DV = ['norm', 'entropy', 'sparsity', 'imag_norm', 'visual_norm', 'concrete_norm']


def imagist_lines():
    im = json.load(open('data/some_imagists.json'))
    rows = []
    for poet, works in im.items():
        for w in works:
            for s in w['poem'].replace('\n', ' ').replace('  ', ' ').strip().split('*'):
                s = s.strip()
                if s and not s.isdigit():
                    rows.append((poet, MERGE.get(w['title'], w['title']), s))
    return pd.DataFrame(rows, columns=['poet', 'poem', 'text'])


def dedup(meas, emb):
    _, first = np.unique(emb.round(6), axis=0, return_index=True)
    return meas.iloc[np.sort(first)].reset_index(drop=True)


def d(x, y):
    x, y = x.dropna(), y.dropna()
    sp = np.sqrt(((len(x) - 1) * x.var() + (len(y) - 1) * y.var()) / (len(x) + len(y) - 2))
    return (x.mean() - y.mean()) / sp


def t(x, y):
    return ttest_ind(x.dropna(), y.dropna())


im_m = pd.read_json('data/measures/imagists_sentences.json')
im_m = pd.concat([im_m, imagist_lines()], axis=1)
im = dedup(im_m, np.array(json.load(open('data/embeddings/imagists_sentences.json'))))
lv = dedup(pd.read_json('data/measures/love_sentences.json'),
           np.array(json.load(open('data/embeddings/love_sentences.json'))))
for df in (im, lv):
    df['n_hits'] = (df['imag'] / df['imag_norm']).round()  # MRC-covered tokens per line
im['chars'] = im.text.str.len()
print(f'Imagist {len(im)} lines in {im.poem.nunique()} poems, Love {len(lv)} lines')

print('\n== A. Table 6 with effect sizes (Imagist minus Love, Cohen d)')
for m in DV:
    r = t(im[m], lv[m])
    print(f'{m:14} t={r.statistic:6.2f}  d={d(im[m], lv[m]):+.2f}')

print('\n== B. Imagist poets (line means)')
g = im.groupby('poet').agg(lines=('text', 'size'), poems=('poem', 'nunique'), chars=('chars', 'mean'),
                           norm=('norm', 'mean'), imag_norm=('imag_norm', 'mean'),
                           concrete_norm=('concrete_norm', 'mean'))
g['d_norm_vs_love'] = [d(im.loc[im.poet == p, 'norm'], lv['norm']) for p in g.index]
print(g.round(2).to_string())
print(f'Love (all lines): norm={lv.norm.mean():.2f} imag_norm={lv.imag_norm.mean():.1f} '
      f'concrete_norm={lv.concrete_norm.mean():.2f}')
top = im.poem.value_counts()
print('largest poems (lines):', top.head(5).to_dict())

print('\n== C. Nesting: Imagist lines within poems')
icc_fit = smf.mixedlm('norm ~ 1', im, groups=im['poem']).fit()
tau2, s2 = float(icc_fit.cov_re.iloc[0, 0]), icc_fit.scale
icc = tau2 / (tau2 + s2)
m_bar = len(im) / im.poem.nunique()
print(f'ICC(norm by poem) = {icc:.2f}; mean poem size {m_bar:.1f} lines; '
      f'design effect ~ {1 + (m_bar - 1) * icc:.1f} -> effective Imagist n ~ {len(im) / (1 + (m_bar - 1) * icc):.0f}')
pm = im.groupby('poem')[DV].mean()
print('poem-level Imagist means (37) vs Love lines (Love still line-level, so this is only half the fix):')
for m in DV:
    r = t(pm[m], lv[m])
    print(f'  {m:14} t={r.statistic:6.2f}  p={r.pvalue:.3g}')

print('\n== D. Leave one Imagist poet out (norm, t vs Love)')
for p in im.poet.unique():
    r = t(im.loc[im.poet != p, 'norm'], lv['norm'])
    print(f'  without {p:20} t={r.statistic:6.2f}')

print('\n== E. Apply the Love filter (len > 5 chars) to Imagist lines too')
im5 = im[im.chars > 5]
print(f'Imagist lines dropped: {len(im) - len(im5)}')
for m in DV:
    print(f'  {m:14} t={t(im5[m], lv[m]).statistic:6.2f}  d={d(im5[m], lv[m]):+.2f}')

print('\n== F. Does norm separate the groups beyond the dictionaries and line length?')
both = pd.concat([im.assign(imagist=1), lv.assign(imagist=0)], ignore_index=True)
both = both.dropna(subset=['imag_norm', 'visual_norm', 'concrete_norm'])
z = lambda s: (s - s.mean()) / s.std()
for c in ['norm', 'imag_norm', 'visual_norm', 'concrete_norm', 'n_hits']:
    both['z_' + c] = z(both[c])
for f in ['imagist ~ z_norm',
          'imagist ~ z_imag_norm + z_visual_norm + z_concrete_norm + z_n_hits',
          'imagist ~ z_norm + z_imag_norm + z_visual_norm + z_concrete_norm + z_n_hits']:
    fit = smf.logit(f, both).fit(disp=0)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(both.imagist, fit.predict(both))
    co = fit.params.get('z_norm', np.nan); p = fit.pvalues.get('z_norm', np.nan)
    print(f'  {f:72} AUC={auc:.3f}  b(norm)={co:+.2f} p={p:.2g}')

print('\n== G. Within-corpus Spearman: norm vs dictionary scores (raw | partial on n_hits)')
def partial(df, a, b, c):
    r = df[[a, b, c]].dropna().rank()
    ra = r[a] - np.polyval(np.polyfit(r[c], r[a], 1), r[c])
    rb = r[b] - np.polyval(np.polyfit(r[c], r[b], 1), r[c])
    return np.corrcoef(ra, rb)[0, 1]
for name, df in [('Imagist', im), ('Love', lv)]:
    out = []
    for m in ['imag_norm', 'visual_norm', 'concrete_norm']:
        rho = spearmanr(df['norm'], df[m], nan_policy='omit').statistic
        out.append(f'{m} {rho:+.2f} | {partial(df, "norm", m, "n_hits"):+.2f}')
    print(f'  {name:8} ' + ';  '.join(out) + f';  norm~n_hits {spearmanr(df.norm, df.n_hits, nan_policy="omit").statistic:+.2f}')
