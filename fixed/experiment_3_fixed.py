# %%
# Patched copy of experiment_3.py (issue 7: data does not match the paper).
# Run from inside source/:  python ../fixed/experiment_3_fixed.py
# Change vs. the original: duplicate lines are dropped from both corpora after the
# cached measures are loaded. This reproduces the paper's line counts (1,195 / 1,126)
# and the embedding rows of Table 6 exactly.
import sys
import numpy as np
import json
import os

from scipy.special import softmax
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from datasets import load_dataset
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# from functions import generate_clip_embeddings
# from functionsimport get_dict_scores
from functions_new import generate_clip_embeddings
from functions_new import get_dict_scores

from transformers import CLIPProcessor, CLIPModel
import torch
import json
model_name = "openai/clip-vit-base-patch32"

# # correlations on the dictionary entries
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)

### Last experiment, here we compare imagist to modern love poetry

# %%
# entropy
def calculate_entropy(embedding,e_formula):
    if e_formula == 'orig':
        # Convert the embedding values to probabilities using softmax
        probabilities = softmax(embedding)
         # Calculate the entropy
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-9))  # Adding a small constant to avoid log(0)
    elif e_formula == 'paper':
        eps = 1e-9
        a = np.abs(np.asarray(embedding, dtype=float))
        total = a.sum()
        if total == 0:                      # all-zero vector: treat as uniform
            p = np.full(len(a), 1 / len(a))
        else:
            p = a / total # p_i = |e_i| / sum |e_j|, sums to 1
        entropy = -np.sum(p * np.log(p + eps)) # H(e) = -sum p_i log(p_i + eps)
    else: raise ValueError(f"unknown e_formula {e_formula}")
    
    return entropy

# sparse ratio
# Function to compute the ratio of L1 to L2 norms
def calculate_sparsity_ratio(embedding):
    embedding = np.array(embedding)
    l1_norm = np.sum(np.abs(embedding))  # L1 norm
    l2_norm = np.sqrt(np.sum(embedding**2))  # L2 norm
    sparsity_ratio = l1_norm / (np.sqrt(len(embedding)) * l2_norm)
    return sparsity_ratio

# %%

# CONFIGS

# we can do the analysis on a poem basis or a sentence basis (default, sentences)
analysis_basis = "sentences"
print(f"Analysis basis: {analysis_basis}")
from pathlib import Path

# ---- change these per run ----
RUN_NAME       = "tryrun"   # label for this try
DATA_TAG       = "new"                # which version of the text: "orig" or "new" or "final"
ANALYSIS_BASIS = "sentences"
DESCRIPTION = "tryout"
# ------------------------------
print(Path.cwd())
def make_paths(run_name, data_tag, basis, root = Path(__file__).resolve().parent.parent):
    data_dir = root / "data"
    dict_dir = root / "source" / "resources"

    p = {
        "imagist_data": data_dir / "some_imagists.json",
        "love_data":    data_dir / ("new/modern_love_final.json"if DATA_TAG == "final" else "new/modern_love.json"),
        "imagist_emb":  root / "data/derived/embeddings" / f"imagists_{basis}_{data_tag}.json",
        "love_emb":     root / "data/derived/embeddings" / f"love_{basis}_{data_tag}.json",
        "run_dir":      root / "results" / run_name,
        
    }

    p["mrc_dict"]          = dict_dir / "mrc_psychol_dict.json"
    p["sensorimotor_dict"] = dict_dir / "sensorimotor_norms_dict.json"
    p["concreteness_dict"] = dict_dir / "concreteness_brysbaert.json"
    p["lines_csv"] = p["run_dir"] / "lines.csv"
    p["tests_csv"] = p["run_dir"] / "tests.csv"
    p["run_info"]  = p["run_dir"] / "run_info.json"
    p["figs"]      = p["run_dir"] / "figs"
    for d in (p["imagist_emb"].parent, p["run_dir"], p["figs"]):
        d.mkdir(parents=True, exist_ok=True)
    return p

paths = make_paths(RUN_NAME, DATA_TAG, ANALYSIS_BASIS)

# %%
# and we get the poetry from:
# https://huggingface.co/datasets/merve/poetry
def get_merve(poets): #get poets from df_img: poets = df_imag['poet'].unique()
    ds = load_dataset("merve/poetry")


    # make df
    df = pd.DataFrame(ds['train'])
    print(len(df))
    df.head()
    # # take only the modern
    df_modern = df[df['age'] == 'Modern']
    # take only the poems where type is Love
    df_love = df_modern[df_modern['type'] == 'Love']
    print('len df', len(df_love))
    # remove any others that are in the imagist poets
    df_love = df_love[~df_love['author'].isin(poets)]
    print('len df after filtering authors', len(df_love))

    # more cleaning needed from these cause they are scraped data

        


    # take poems
    if analysis_basis == "poems":
        units = df_love['content'].tolist()
        units_clean_merve = []
        for poem in units:
            poem = poem.replace('\r', '').strip()
            # split into lines first, then filter — do not replace \n before splitting
            lines = poem.split('\n')
            lines = [line.strip().replace('*', '').replace('  ', ' ') for line in lines]
            lines = [line for line in lines if line and line != ' ']
            lines = [line for line in lines if 'copyright' not in line.lower()]
            lines = [line for line in lines if 'reprint' not in line.lower()]
            if not lines:
                continue
            poem = ' '.join(lines).strip()
            if len(poem) < 5:
                continue
            units_clean_merve.append(poem)
        print('no. of poems', len(units_clean_merve))

    # take sentences
    elif analysis_basis == "sentences":
        # get all the sentences, seperated by \n
        units = df_love['content'].str.split('\n').explode().tolist()
        # save the author and poem names
        authors = df_love['author'].str.split('\n').explode().tolist()
        print('authors', len(set(authors)))
        poems_names = df_love['poem name'].str.split('\n').explode().tolist()
        #print('poems', len(set(poems)))

        # get the sentences
        units_clean_merve = []
        for line in units:
            # we want to remove extra spaces
            line = line.replace('\n', ' ').replace('  ', ' ')
            line = line.strip().replace('*', '')
            # remove leading and trailing spaces
            line = line.strip()
            units_clean_merve.append(line)
        
        # remove emtpy strings
        units_clean_merve = [unit for unit in units_clean_merve if unit != '']
        # remove sentences that are just a number
        units_clean_merve = [unit for unit in units_clean_merve if not unit.isdigit()]
        # remove empty strings
        units_clean_merve = [unit for unit in units_clean_merve if unit != ' ']
        units_clean_merve = [unit for unit in units_clean_merve if len(unit) > 5]
        # remove \r from strings
        units_clean_merve = [unit.replace('\r', '') for unit in units_clean_merve]
        # remove all sentences containing "copyright"
        units_clean_merve = [unit for unit in units_clean_merve if 'copyright' not in unit.lower()]
        # remove all sentence containing "reprint" or "reprinted"
        units_clean_merve = [unit for unit in units_clean_merve if 'reprint' not in unit.lower()]
        print('no. of lines', len(units_clean_merve))

    else:
        raise ValueError("Invalid analysis_basis")
    return units_clean_merve
# %%
import re

def split_poem(raw: str, line_sep: str) -> list[str]:
    """按数据源各自的行分隔符切成诗行（两组数据唯一不同的地方）"""
    if line_sep == '*':
        raw = raw.replace('\n', ' ')   # imagists 的行界由 * 标记
    return raw.split(line_sep)

def clean_line(s: str) -> str:
    s = s.replace('\r', ' ').replace('\u00a0', ' ')
    return re.sub(r'\s+', ' ', s).strip()

def is_valid(s: str, min_chars: int = 0) -> bool:
    if not re.search(r'[A-Za-z]', s):          # 空串、纯数字、纯标点
        return False
    low = s.lower()
    if 'copyright' in low or 'reprint' in low:
        return False
    return len(s) >= min_chars

def load_poetry(path, group, line_sep, analysis_basis="lines", min_chars=0):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    rows = []
    poem_id = 0
    for poet, works in data.items():
        for w in works:
            lines = [clean_line(l) for l in split_poem(w['poem'], line_sep)]
            lines = [l for l in lines if is_valid(l, min_chars)]
            if not lines:
                continue
            if analysis_basis == "poems":
                rows.append(dict(group=group, poet=poet, title=w['title'],
                                 poem_id=poem_id, line_no=0, text=' '.join(lines)))
            elif analysis_basis == "lines":
                for i, l in enumerate(lines):
                    rows.append(dict(group=group, poet=poet, title=w['title'],
                                     poem_id=poem_id, line_no=i, text=l))
            else:
                raise ValueError("analysis_basis must be 'lines' or 'poems'")
            poem_id += 1
    return pd.DataFrame(rows)


# %%

# now we get the embeddings for each sentence

# check if embeddings are already saved
def get_emb(genre,units_clean):
    emb_path = paths[f"{genre}_emb"]

    if not os.path.exists(emb_path):
        sentence_embeddings = generate_clip_embeddings(units_clean)
        with open(emb_path, 'w') as f:
            json.dump(sentence_embeddings.tolist(), f)
        print(f'{genre}_embedding generated')
    else:
        with open(emb_path) as f:
            sentence_embeddings = np.array(json.load(f))
        print(f'{genre}_embedding loaded')

    assert len(sentence_embeddings) == len(units_clean), "cached embeddings don't match the lines, delete the file"
    return sentence_embeddings
# %%


"""
def score_df(units_clean,sentence_embeddings_imagist):
    norms = np.linalg.norm(sentence_embeddings_imagist, axis=1)
    entropies = [calculate_entropy(embedding) for embedding in sentence_embeddings_imagist]
    variances = np.var(sentence_embeddings_imagist, axis=1)
    sparsities = [calculate_sparsity_ratio(embedding) for embedding in sentence_embeddings_imagist]

# check if the scores are already saved !no need to save the score, all scores will be recompute in every run

    # get the imageability scores
    imag, imag_norm = get_dict_scores(units_clean, paths["mrc_dict"], 'imag', 'lemma_', normalize_by_tokens=False)
    vis, vis_norm = get_dict_scores(units_clean, paths["sensorimotor_dict"] , 'Visual.mean', 'lemma_', normalize_by_tokens=False)
    conc, conc_norm = get_dict_scores(units_clean, paths["concreteness_dict"], 'Conc.M', 'lemma_', normalize_by_tokens=False)

    # make a dataframe
    df_sentences = pd.DataFrame({'norm': norms, 'entropy': entropies, 'variance': variances, 'sparsity': sparsities})
    df_sentences['imag'] = imag
    df_sentences['visual'] = vis
    df_sentences['concrete'] = conc
    df_sentences['imag_norm'] = imag_norm
    df_sentences['visual_norm'] = vis_norm
    df_sentences['concrete_norm'] = conc_norm

    # save the measures
    return df_sentences
"""
# column name: (paths key, score key in the lexicon)
DICTS = {
    'imag':     ('mrc_dict',          'imag'),
    'visual':   ('sensorimotor_dict', 'Visual.mean'),
    'concrete': ('concreteness_dict', 'Conc.M'),
}

def score_df(df, emb, paths):
    """df: rows from load_poetry (group, poet, title, poem_id, line_no, text); emb: one vector per row."""
    emb = np.asarray(emb)
    out = df.reset_index(drop=True).copy()

    # embedding measures
    out['norm']          = np.linalg.norm(emb, axis=1)
    out['entropy_orig']  = [calculate_entropy(e, 'orig')  for e in emb]
    out['entropy_paper'] = [calculate_entropy(e, 'paper') for e in emb]
    out['variance']      = emb.var(axis=1)
    out['sparsity']      = [calculate_sparsity_ratio(e) for e in emb]

    # dictionary measures
    texts = out['text'].tolist()
    for name, (path_key, score_key) in DICTS.items():
        out[name], out[f'{name}_norm'] = get_dict_scores(
            texts, paths[path_key], score_key, 'lemma_', normalize_by_tokens=False)

    # drop repeated lines; the paper's counts exclude them
    return out[~out['text'].duplicated()].reset_index(drop=True)

df_imag = load_poetry(paths['imagist_data'], 'imagist', line_sep='*')
df_love = load_poetry(paths['love_data'], 'modern_love', line_sep='\n') if DATA_TAG != "orig" else get_merve(df_imag['poet'].unique())

df_sentences      = score_df(df_imag, get_emb('imagist', df_imag['text'].tolist()), paths)
df_sentences_love = score_df(df_love, get_emb('love',    df_love['text'].tolist()), paths)
pd.concat([df_sentences, df_sentences_love]).to_csv(paths['lines_csv'], index=False)

# %% 
# we get imageable and non imageable sentences as reference points
# cocnat the two dfs
df_sentences_all = pd.concat([df_sentences, df_sentences_love])

# make pd print all
pd.set_option('display.max_colwidth', None)

# print the first 10 most imageable sentences in both dfs
print('most imageable')
# make it a list
imageable = df_sentences_all.sort_values('imag_norm', ascending=True)['text'].tolist()
# print the first 10
for sent in imageable[:300]:
    print(sent)
    print('\n')

# %%
# we choose these sentences (from topmosts and bottommosts) as examples
imag_example = "Homespun, dyed butternuts dark gold color."
non_imag_example = "Of insidious intent"

# retrieve the measures for these from the df
print('imag example exists:', imag_example)
imag_metrics = df_sentences_all[df_sentences_all['text'] == imag_example]
print('non-imag example exists:', non_imag_example)
non_imag_metrics = df_sentences_all[df_sentences_all['text'] == non_imag_example]

# %%
# make distributions of each group for each measure
measures = ['norm', 'entropy_orig','entropy_paper', 'sparsity', 
            'imag_norm', 'visual_norm', 'concrete_norm']


def plot_distributions(df_a, df_b, measures, paths, examples=None, name="distributions.png"):
    ncols = 3
    nrows = -(-len(measures) // ncols)          # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows), squeeze=False)
    for ax, m in zip(axes.flat, measures):
        sns.histplot(df_a[m], ax=ax, label='Imagist', color='blue', kde=True, alpha=0.2, stat='density')
        sns.histplot(df_b[m], ax=ax, label='Modern Love', color='red', kde=True, alpha=0.2, stat='density')
        for label, row, colour in (examples or []):
            if len(row):                         # skip if the line isn't in this run
                ax.axvline(row[m].values[0], color=colour, linestyle='--', label=label)
        ax.set_title(m)
    for ax in axes.flat[len(measures):]:         # hide unused panels
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right')
    fig.tight_layout()
    out = paths['figs'] / name
    fig.savefig(out, dpi=200)                    # save first, then show
    plt.show()
    return out
fig_path = plot_distributions(df_sentences, df_sentences_love, measures, paths,
                              examples=[('Imageable', imag_metrics, 'blue'),
                                        ('Abstract', non_imag_metrics, 'red')])
print('figure saved to', fig_path)
# print len of both
print('len imag', len(df_sentences))
print('len love', len(df_sentences_love))
# %%
# do the t-test & mannwhitneyu test
def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1-1)*a.var(ddof=1) + (n2-1)*b.var(ddof=1)) / (n1+n2-2))
    return (a.mean() - b.mean()) / sp

alpha = 0.05
bonf = alpha / len(measures)
rows = []
for m in measures:
    a = df_sentences[m].dropna()
    b = df_sentences_love[m].dropna()
    t, p = ttest_ind(a, b)
    U, pU = mannwhitneyu(a, b)
    rows.append(dict(measure=m, n_imagist=len(a), n_love=len(b),
                     mean_imagist=a.mean(), sd_imagist=a.std(),
                     mean_love=b.mean(), sd_love=b.std(),
                     t=t, p_t=p, U=U, p_U=pU,
                     cohens_d=cohens_d(a, b),
                     auc=U / (len(a) * len(b)),   # P(Imagist > Love)
                     bonferroni_alpha=bonf, sig_t=p < bonf, sig_U=pU < bonf))
tests = pd.DataFrame(rows)
tests.to_csv(paths['tests_csv'], index=False, float_format='%.6g')
print(tests[['measure', 't', 'p_t', 'cohens_d', 'auc', 'sig_t']].to_string(index=False))


# %%
# get the numbers of means and stds per group
print('Imagists')
print(df_sentences.describe())
print('Modern Love')
print(df_sentences_love.describe())

# %%
print('All done:)')

import subprocess, datetime, platform, hashlib
from importlib.metadata import version

def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()

root = Path(__file__).resolve().parent.parent
git = lambda *a: subprocess.run(['git', *a], cwd=root, capture_output=True, text=True).stdout.strip()

info = dict(
    run_name=RUN_NAME, description=DESCRIPTION,   # add DESCRIPTION to the config block
    date=datetime.datetime.now().isoformat(timespec='seconds'),
    data_tag=DATA_TAG, analysis_basis=ANALYSIS_BASIS, dedup=True,
    imagist_data=str(paths['imagist_data']), imagist_sha=sha(paths['imagist_data']),
    love_data=str(paths['love_data']), love_sha=sha(paths['love_data']),
    clip_model=model_name, entropy_methods=['orig', 'paper'],
    git_commit=git('rev-parse', 'HEAD'), git_dirty=bool(git('status', '--porcelain')),
    python=platform.python_version(),
    versions={p: version(p) for p in ['numpy', 'pandas', 'scipy', 'torch',
                                      'transformers', 'huggingface_hub', 'spacy']},
    n_lines=dict(imagist=len(df_sentences), love=len(df_sentences_love)),
)
with open(paths['run_info'], 'w') as f:
    json.dump(info, f, indent=2)

idx = root / 'results' / 'runs.csv'
row = {k: info[k] for k in ['run_name', 'date', 'description', 'data_tag', 'git_commit', 'git_dirty']}
row |= {f'{c}_{r.measure}': r[c] for _, r in tests.iterrows() for c in ('t', 'cohens_d', 'auc')}
pd.DataFrame([row]).to_csv(idx, mode='a', header=not idx.exists(), index=False)
# %%
## gird search
corpus = ['orig','new','final']
basis = ['poem','sentence']
e_formula = ['orig','paper']
