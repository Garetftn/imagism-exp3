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

#sys.path.insert(0, os.getcwd())  # so `functions` is found when run from the repo folder

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
def calculate_entropy(embedding):
    # Convert the embedding values to probabilities using softmax
    probabilities = softmax(embedding)
    # Calculate the entropy
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-9))  # Adding a small constant to avoid log(0)
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
def get_merve():
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
df_imag = load_poetry(paths['imagist_data'], 'imagist', line_sep='*')
df_love = load_poetry(paths['love_data'], 'modern_love', line_sep='\n') if DATA_TAG != "orig" else get_merve()
df_all  = pd.concat([df_imag, df_love], ignore_index=True)

units_clean = df_all[df_all['group']=='imagist']['text'].tolist()
units_clean_merve = df_all[df_all['group']=='modern_love']['text'].tolist()

sentence_embeddings_imagist = get_emb("imagist",units_clean)
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
df_sentences = score_df(units_clean,sentence_embeddings_imagist)
# add the text
df_sentences['text'] = units_clean
# drop repeated lines (refrains such as "Zeus," or "In the dusk"); the paper's counts exclude them
df_sentences = df_sentences[~df_sentences['text'].duplicated()].reset_index(drop=True)
print('imagist lines after dropping duplicates', len(df_sentences))

# %%

        
sentence_embeddings_love = get_emb("love",units_clean_merve)

df_sentences_love = score_df(units_clean_merve,sentence_embeddings_love)

# add the text
df_sentences_love['text'] = units_clean_merve
# drop repeated lines; the paper's count (1,126) excludes them
df_sentences_love = df_sentences_love[~df_sentences_love['text'].duplicated()].reset_index(drop=True)
print('love lines after dropping duplicates', len(df_sentences_love))

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
measures = ['norm', 'entropy', 'sparsity', 
            'imag_norm', 'visual_norm', 'concrete_norm']

sns.set(style='whitegrid')

plt.figure(figsize=(12, 5))

for i, measure in enumerate(measures):
    plt.subplot(2, 3, i+1)
    sns.histplot(df_sentences[measure], label='Imagist', color='blue', kde=True, alpha=0.2, stat="density")#, shade=True)
    sns.histplot(df_sentences_love[measure], label='Modern Love', color='red', kde=True, alpha=0.2, stat="density")#, shade=True)
    # insert the lines for the examples
    plt.axvline(imag_metrics[measure].values[0], color='blue', linestyle='--', label='Imageable')
    plt.axvline(non_imag_metrics[measure].values[0], color='red', linestyle='--', label='Abstract')
    plt.title(measure)
plt.tight_layout()
# add legend
plt.legend()
plt.show()

# print len of both
print('len imag', len(df_sentences))
print('len love', len(df_sentences_love))
# %%
# do the t-test & mannwhitneyu test
for measure in measures:
    # make sure to drop any NaNs
    df_sentences_nonan = df_sentences.dropna(subset=[measure])
    df_sentences_love_nonan = df_sentences_love.dropna(subset=[measure])
    t, p = ttest_ind(df_sentences_nonan[measure], df_sentences_love_nonan[measure])

    # also make a mannwhitneyu test
    t_m, p_m = mannwhitneyu(df_sentences_nonan[measure], df_sentences_love_nonan[measure])

    print(f'{measure}: t = {t:.2f}, p = {p:.2f}', f'mannwhitneyu: t = {t_m:.2f}, p = {p_m:.2f}')
    print('--')

# %%
# do the same but with the bonferroni correction

alpha = 0.05  # Original significance level
n_tests = len(measures)  # Number of comparisons (tests)
bonferroni_alpha = alpha / n_tests  # Adjusted significance level for Bonferroni correction

for measure in measures:
    # Make sure to drop any NaNs
    df_sentences_nonan = df_sentences.dropna(subset=[measure])
    df_sentences_love_nonan = df_sentences_love.dropna(subset=[measure])
    # print the group sizes
    print('group sizes', len(df_sentences_nonan), len(df_sentences_love_nonan))

    # Perform T-test 
    t, p = ttest_ind(df_sentences_nonan[measure], df_sentences_love_nonan[measure])
    # print the group sizes
    print('group sizes', len(df_sentences_nonan), len(df_sentences_love_nonan))
    # Apply Bonferroni correction
    is_significant = p < bonferroni_alpha
    print(f'{measure}: t = {t:.2f}, p = {p:.2f}, significant: {is_significant}')

    # also make a mannwhitneyu test
    t_m, p_m = mannwhitneyu(df_sentences_nonan[measure], df_sentences_love_nonan[measure])
    # Apply Bonferroni correction
    is_significant = p_m < bonferroni_alpha
    print(f'{measure}: mannwhitneyu: t = {t_m:.2f}, p = {p_m:.2f}, significant: {is_significant}')

    print('--')

# %%
# get the numbers of means and stds per group
print('Imagists')
print(df_sentences.describe())
print('Modern Love')
print(df_sentences_love.describe())

# %%
print('All done:)')
# %%
