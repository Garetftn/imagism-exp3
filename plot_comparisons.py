# Comparison charts for the Experiment III runs made by run_grid.py
# reads results/exp3_runs.jsonl and data/measures/<metrics_key>_metrics.csv, writes figs/compare/
# run from the repo root:  python plot_comparisons.py
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                      # no windows, just save files
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from experiment_3_replication_commented import test

ORIG = 'imag_orig_love_orig_entropy_orig_base_computed'   # the "original result" every run is compared to
ORIG_METRICS = ORIG.rsplit('_base_', 1)[0]                # metrics CSVs are named without base_line
measures = ['norm', 'entropy', 'sparsity', 'imag_norm', 'visual_norm', 'concrete_norm']
os.makedirs('figs/compare', exist_ok=True)

# one record per run_id; a re-run appends a new line, so the last one wins
with open('results/exp3_runs.jsonl', encoding='utf-8') as f:
    runs = {r['run_id']: r for r in map(json.loads, f)}
run_ids = [ORIG] + sorted(r for r in runs if r != ORIG)

tests = pd.DataFrame([dict(t, run_id=rid) for rid in run_ids for t in runs[rid]['tests']])

# 1. heat map: how much each run changes the original result, recomputed from the metrics files
#    each kind of number is compared in the way that suits it (see the note printed on the figure)
heat_runs = ['imag_orig_love_orig_entropy_paper',     # metrics keys: only the entropy formula changed
             'imag_final_love_final_entropy_paper']   # final corpora + paper entropy

def run_tests(key):
    m = pd.read_csv(f'data/measures/{key}_metrics.csv', encoding='utf-8-sig')
    return test(m[m['group'] == 'imagist'], m[m['group'] == 'modern_love'], measures).set_index('measure').loc[measures]

base = run_tests(ORIG_METRICS)
res = {key: run_tests(key) for key in heat_runs}

# n, U, t, p and the significance flags grow with the number of lines, so they go in a table, not the heat map
pd.concat({ORIG_METRICS: base, **res}).to_csv('figs/compare/heatmap_raw_values.csv')

nlog = lambda p: -np.log10(p.clip(lower=1e-300))   # clip: a p that underflows to 0 would give inf
def panels(key, r):
    pct_mean = pd.DataFrame({'Imagist mean': (r['mean_imag'] - base['mean_imag']) / base['mean_imag'].abs() * 100,
                             'Love mean':    (r['mean_love'] - base['mean_love']) / base['mean_love'].abs() * 100})
    if key.split('_entropy_')[1] != ORIG_METRICS.split('_entropy_')[1]:
        pct_mean.loc['entropy'] = np.nan    # other entropy formula = other scale, so its mean is not comparable
    return [pd.DataFrame({"Cohen's d": r['Cohens d'] - base['Cohens d'], 'AUC': r['auc'] - base['auc']}).T,
            pd.DataFrame({'t-test': nlog(r['p']) - nlog(base['p']),
                          'Mann-Whitney U': nlog(r['p_U']) - nlog(base['p_U'])}).T,
            pct_mean.T]
tabs = {key: panels(key, r) for key, r in res.items()}

# readable labels for the figure (keys stay in the file names and the CSV)
run_labels = {'imag_orig_love_orig_entropy_paper':   'Paper entropy formula only',
              'imag_final_love_final_entropy_paper': 'Final corpora + paper entropy'}
measure_labels = ['CLIP\nnorm', 'CLIP\nentropy', 'CLIP\nsparsity', 'Imageability\n(MRC)', 'Visuality\n(Lancaster)', 'Concreteness\n(Brysbaert)']
rows = [("Change in effect size\n(run − original)", '{:+.3f}'),
        ('Change in −log10 p\n(run − original)',    '{:+.1f}'),
        ('% change of group mean',                    '{:+.1f}%')]
# diverging map from the publication palette: blue = lower than original, red = higher
cmap = LinearSegmentedColormap.from_list('blue_white_red', ['#0F4D92', '#FFFFFF', '#B64342'])

with plt.rc_context({'font.family': ['Arial', 'DejaVu Sans', 'sans-serif'], 'font.size': 13,
                     'axes.linewidth': 0, 'svg.fonttype': 'none'}):
    fig, axes = plt.subplots(3, len(heat_runs) + 1, figsize=(15, 10),
                             gridspec_kw={'width_ratios': [1] * len(heat_runs) + [0.04], 'hspace': 0.45, 'wspace': 0.08})
    for i, (row_title, fmt) in enumerate(rows):
        lim = max(np.nanmax(np.abs(tabs[k][i].values)) for k in heat_runs) or 1   # shared, symmetric scale per row
        for j, key in enumerate(heat_runs):
            ax, tab = axes[i, j], tabs[key][i]
            sns.heatmap(tab, ax=ax, cmap=cmap, center=0, vmin=-lim, vmax=lim, linewidths=1.5, linecolor='white',
                        annot=tab.map(lambda v: '' if np.isnan(v) else '0' if fmt.format(v).lstrip('+-').rstrip('%').strip('0.') == '' else fmt.format(v)), fmt='',   # unchanged cells read '0', not '+0.000' / '-0.0'
                        annot_kws={'fontsize': 12}, cbar=(j == len(heat_runs) - 1), cbar_ax=axes[i, -1],
                        cbar_kws={'label': row_title.split('\n')[0]})
            ax.patch.set(facecolor='#CFCECE', hatch='//', edgecolor='white')   # blank cells = not comparable
            ax.set_xticklabels(measure_labels if i == 2 else [], rotation=0, fontsize=11)
            ax.set_xlabel('')
            ax.tick_params(length=0)
            ax.set_ylabel(row_title if j == 0 else '', fontsize=13, labelpad=12)
            if j: ax.set_yticklabels([])
            else: ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            if i == 0: ax.set_title(run_labels.get(key, key), fontsize=15, fontweight='bold', pad=12)
    fig.suptitle('Experiment III: change against the original run (original corpora, softmax entropy)', fontsize=16, y=0.98)
    fig.text(0.02, 0.01,
             "Effect size: Δd = d_run − d_orig, d = (mean_imag − mean_love) / √((var_imag + var_love)/2);  "
             'ΔAUC = AUC_run − AUC_orig, AUC = U / (n_imag · n_love).\n'
             'Δ −log10 p = −log10 p_run + log10 p_orig: +1 = p ten times smaller (stronger evidence).  '
             '% change of mean = (mean_run − mean_orig) / |mean_orig| · 100.\n'
             'Hatched = entropy formula differs, so the means are on different scales.  '
             'Raw n, U, t, p and significance: figs/compare/heatmap_raw_values.csv.',
             fontsize=10, va='bottom', color='#4D4D4D')
    fig.subplots_adjust(left=0.17, right=0.93, top=0.88, bottom=0.16)
    for ext in ['png', 'pdf']:
        fig.savefig(f'figs/compare/heatmap_change_vs_original.{ext}', dpi=300, facecolor='white')
    plt.close(fig)

# 2. top 10 / bottom 10 lines (by imag_norm, as in take_record) of every run, as a table:
#    each line once, how many runs put it in the top/bottom 10, and its scores
ext = pd.DataFrame([dict(l, run_id=rid, which=which)
                    for rid in run_ids for which in ['top10', 'bottom10'] for l in runs[rid][which]])
ext['e_formula'] = ext['run_id'].str.extract(r'_entropy_(\w+?)_base_', expand=False)
# the orig and final corpora spell some lines differently (curly quotes, capitalised poet names); match them anyway
ext['key'] = (ext['text'].str.replace('[‘’]', "'", regex=True).str.replace('[“”]', '"', regex=True)
              .str.strip().str.casefold())
ent = ext.pivot_table(index=['which', 'group', 'key'], columns='e_formula', values='entropy', aggfunc='mean')
ent.columns = [f'entropy_{c}' for c in ent.columns]       # entropy depends on the formula, so one column per formula
table = (ext.groupby(['which', 'group', 'key'])
            .agg(line=('text', 'first'), poet=('poet', lambda p: p.str.title().iloc[0]), title=('title', 'first'),
                 n_runs=('run_id', 'nunique'),
                 imag_norm=('imag_norm', 'mean'), visual_norm=('visual_norm', 'mean'),
                 concrete_norm=('concrete_norm', 'mean'),
                 norm=('norm', 'mean'), sparsity=('sparsity', 'mean'))   # CLIP scores shift slightly between corpus versions
            .join(ent).reset_index().drop(columns='key')
            .sort_values(['which', 'imag_norm'], ascending=[False, False]))
table.to_csv('figs/compare/top_bottom10_table.csv', index=False)

# the same table as a figure, one panel for top 10 and one for bottom 10
show = ['line', 'poet', 'group', 'n_runs', 'imag_norm', 'visual_norm', 'concrete_norm', 'norm', 'sparsity'] + list(ent.columns)
with plt.rc_context({'font.family': ['Arial', 'DejaVu Sans'], 'font.size': 10}):
    sizes = [(table['which'] == w).sum() + 1 for w in ['top10', 'bottom10']]   # rows incl. header
    fig, axes = plt.subplots(2, 1, figsize=(20, 0.3 * sum(sizes) + 1.5), gridspec_kw={'height_ratios': sizes, 'hspace': 0.25})
    for ax, which in zip(axes, ['top10', 'bottom10']):
        t = table[table['which'] == which][show].copy()
        t['n_runs'] = t['n_runs'].astype(str) + f' / {len(run_ids)}'
        t['line'] = t['line'].str.slice(0, 45)
        cells = t.round(2).astype(str).replace('nan', '–').values
        tb = ax.table(cellText=cells, colLabels=show, bbox=[0, 0, 1, 1], cellLoc='left')   # fill the panel, no gaps
        tb.auto_set_font_size(False); tb.set_fontsize(9); tb.auto_set_column_width(range(len(show)))
        for (r, c), cell in tb.get_celld().items():
            cell.set_edgecolor('#CFCECE')
            if r == 0: cell.set(facecolor='#0F4D92'); cell.get_text().set(color='white', fontweight='bold')
        ax.set_axis_off()
        ax.set_title(f'{which}: lines in the {which[:-2]} 10 by imag_norm, across {len(run_ids)} runs '
                     '(n_runs = in how many runs; scores averaged across runs)', loc='left', fontweight='bold')
    fig.savefig('figs/compare/top_bottom10_table.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

# 3. dumbbell: for each measure and run, the Imagist mean joined to the Love mean,
#    with every test result written next to it (same three runs as the heat map, original first)
dumb_runs = {ORIG_METRICS: base, **res}                # recomputed with test() in section 1
dumb_labels = {ORIG_METRICS: 'Original (orig corpora,\nsoftmax entropy)', **{k: run_labels.get(k, k) for k in res}}
dumb_labels = {k: v.replace(' + ', ' +\n') for k, v in dumb_labels.items()}

# x in SD units of the original corpora: (mean - mu0) / sd0, pooled over all lines of both groups.
# The reference uses the run's own entropy formula, because softmax and |e|/sum|e| entropy are on different scales
# (the other five measures don't depend on the formula, so their reference is the same for every run).
def ref_stats(formula):
    m = pd.read_csv(f'data/measures/imag_orig_love_orig_entropy_{formula}_metrics.csv', encoding='utf-8-sig')
    return m[measures].mean(), m[measures].std()
refs = {k: ref_stats(k.split('_entropy_')[1]) for k in dumb_runs}
std = lambda k, m, v: (v - refs[k][0][m]) / refs[k][1][m]

with plt.rc_context({'font.family': ['Arial', 'DejaVu Sans'], 'font.size': 12,
                     'axes.spines.right': False, 'axes.spines.top': False, 'axes.spines.left': False,
                     'axes.linewidth': 1.5, 'svg.fonttype': 'none'}):
    fig, axes = plt.subplots(len(measures), 1, figsize=(16, 2.3 * len(measures) + 1), sharex=True)   # one x-axis for all measures
    y = np.arange(len(dumb_runs))
    for ax, m, label in zip(axes, measures, measure_labels):
        r = pd.DataFrame({k: t.loc[m] for k, t in dumb_runs.items()}).T   # rows = runs
        imag = pd.Series({k: std(k, m, v) for k, v in r['mean_imag'].astype(float).items()})
        love = pd.Series({k: std(k, m, v) for k, v in r['mean_love'].astype(float).items()})
        ax.axvline(0, color='#767676', lw=1, ls=':', zorder=0)       # 0 = pooled mean of the original corpora
        ax.hlines(y, imag, love, color='#CFCECE', lw=4, zorder=1)
        ax.scatter(imag, y, s=110, color='#0F4D92', zorder=2, label='Imagist mean')
        ax.scatter(love, y, s=110, color='#B64342', zorder=2, label='Love mean')
        for yi, (_, row) in zip(y, r.iterrows()):
            ax.annotate(f"d = {row['Cohens d']:+.3f}   AUC = {row['auc']:.3f}   "
                        f"n = {row['n_imag']:.0f} / {row['n_love']:.0f}\n"
                        f"t = {row['t']:.2f}, p = {row['p']:.1e}   U = {row['U']:.0f}, p = {row['p_U']:.1e}" + ('' if row['sig_t'] and row['sig_U'] else '   (n.s.)'),
                        xy=(1.01, yi), xycoords=('axes fraction', 'data'), va='center', fontsize=9, color='#4D4D4D', linespacing=1.3)
        ax.set_yticks(y, [dumb_labels[k] for k in r.index], fontsize=10)
        ax.set_ylim(len(y) - 0.5, -0.5)                                   # original on top
        ax.tick_params(axis='y', length=0)
        ax.set_title(label.replace('\n', ' '), loc='left', fontweight='bold', fontsize=13)
    lim = max(max(abs(l.get_offsets()[:, 0]).max() for l in ax.collections if hasattr(l, 'get_offsets')) for ax in axes)
    axes[-1].set_xlim(-1.15 * lim, 1.15 * lim)                         # symmetric, shared across all panels
    axes[-1].set_xlabel('SD units of the original corpora:  (group mean − pooled mean) / pooled SD', fontsize=12)
    for ax in axes[:-1]: ax.tick_params(axis='x', labelbottom=True)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper right', ncol=2, frameon=False, bbox_to_anchor=(0.99, 0.995))
    fig.suptitle('Experiment III: Imagist vs Love means and tests, original against two revisions', fontsize=15, x=0.01, ha='left')
    fig.text(0.01, 0.005,
             "d = (mean_imag − mean_love) / √((var_imag + var_love)/2);  AUC = U / (n_imag · n_love) = P(random Imagist line > random Love line).  "
             't: two-sample t-test; U: Mann-Whitney U.\n(n.s.) = not significant after Bonferroni (0.05 / 6).  '
             'x = (mean − μ0) / σ0, μ0 and σ0 pooled over all lines of the original corpora; the gap between the dots ≈ Cohen\'s d.\n'
             'Entropy is standardised with the run\'s own formula (softmax for the original, |e|/Σ|e| for the others), so its position shows the gap, not a shift between formulas.',
             fontsize=9.5, color='#4D4D4D', va='bottom')
    fig.subplots_adjust(left=0.17, right=0.66, top=0.93, bottom=0.09, hspace=0.75)
    for ext in ['png', 'pdf']:
        fig.savefig(f'figs/compare/dumbbell_means_tests.{ext}', dpi=300, facecolor='white')
    plt.close(fig)
# 4. original run only: scatter, strip plot, grouped box plots of the line-level metrics
orig = pd.read_csv(f'data/measures/{ORIG_METRICS}_metrics.csv', encoding='utf-8-sig')

sns.scatterplot(data=orig, x='imag_norm', y='norm', hue='group', alpha=0.4, s=12)
plt.title('original run: CLIP norm vs MRC imageability per line')
plt.savefig('figs/compare/orig_scatter_norm_vs_imag.png', dpi=150, bbox_inches='tight'); plt.close()

# one panel per measure, 2 x 3 like the paper's Figure 3, each in its own units (no z-scoring needed)
colours = {'imagist': '#0F4D92', 'modern_love': '#B64342'}
with plt.rc_context({'font.family': ['Arial', 'DejaVu Sans'], 'font.size': 12,
                     'axes.spines.right': False, 'axes.spines.top': False, 'axes.linewidth': 1.5}):
    for kind in ['strip', 'box']:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for ax, m, label in zip(axes.flat, measures, measure_labels):
            if kind == 'strip':
                sns.stripplot(data=orig, x='group', y=m, hue='group', palette=colours, order=list(colours),
                              jitter=0.3, size=2, alpha=0.35, legend=False, ax=ax)
                means = orig.groupby('group')[m].mean()
                ax.hlines([means[g] for g in colours], [-0.4, 0.6], [0.4, 1.4], color='black', lw=2.5)   # group means
            else:
                sns.boxplot(data=orig, x='group', y=m, hue='group', palette=colours, order=list(colours),
                            width=0.5, fliersize=2, linewidth=1.5, legend=False, ax=ax,
                            boxprops={'alpha': 0.6}, medianprops={'color': 'black', 'lw': 2})
            ax.set_title(label.replace('\n', ' '), fontweight='bold')
            ax.set_xticks([0, 1], ['Imagist', 'Love'])
            ax.set_xlabel(''); ax.set_ylabel('')
        note = 'black bar = group mean' if kind == 'strip' else 'box = middle 50% of lines, black line = median, dots = outliers'
        fig.suptitle(f'Experiment III, original run: Imagist vs Love lines per measure ({note})', fontsize=14)
        fig.tight_layout(pad=2)
        for ext in ['png', 'pdf']:
            fig.savefig(f'figs/compare/orig_{kind}.{ext}', dpi=300, facecolor='white')
        plt.close(fig)

print('comparison charts in figs/compare/')
