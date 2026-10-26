# %%
# Patched copy of experiment_3.py (issue 7: data does not match the paper).
# Run from inside source/:  python ../fixed/experiment_3_fixed.py
# Change vs. the original: duplicate lines are dropped from both corpora after the
# cached measures are loaded. This reproduces the paper's line counts (1,195 / 1,126)
# and the embedding rows of Table 6 exactly.
#
# 【中文说明｜文件概览】
# 本脚本是论文实验三（Imagist 意象派诗 vs Modern Love 现代爱情诗）的修订版。
# 整体流程：
#   1. 导入依赖、加载 CLIP 模型；
#   2. 定义两个基于嵌入向量的指标：熵（entropy）和稀疏比（sparsity ratio）；
#   3. 运行配置（RUN_NAME / DATA_TAG / ANALYSIS_BASIS）与所有输入输出路径；
#   4. 读取两组诗歌数据并切分成诗行（新数据用 load_poetry，原始数据用 get_merve）；
#   5. 为每行生成或读取缓存的 CLIP 嵌入；
#   6. 计算每行的嵌入指标 + 词典指标（意象性 / 视觉性 / 具体性），去掉重复行；
#   7. 打印参考例句、画分布图（plot_distributions）；
#   8. 做 t 检验、Mann-Whitney U 检验，计算 Cohen's d 和 AUC，结果写入 tests.csv；
#   9. 记录本次运行的元信息（run_info.json）并追加一行到 results/runs.csv。
# 注：脚本按 "# %%" 分成若干单元格，可在 VS Code / Jupyter 中逐格运行。

# ===== 第 1 部分：导入依赖 =====
import sys
import numpy as np
import json
import os

from scipy.special import softmax              # 原始熵公式用的 softmax
from scipy.stats import ttest_ind              # 独立样本 t 检验
from scipy.stats import mannwhitneyu           # 非参数 Mann-Whitney U 检验
from datasets import load_dataset              # HuggingFace 数据集（仅原始数据 merve/poetry 需要）
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# from functions import generate_clip_embeddings
# from functionsimport get_dict_scores
# 改用同目录下的 functions_new.py（兼容新版 transformers 的嵌入函数 + 词典打分函数）
from functions_new import generate_clip_embeddings
from functions_new import get_dict_scores

from transformers import CLIPProcessor, CLIPModel
import torch
import json
model_name = "openai/clip-vit-base-patch32"    # 论文使用的 CLIP 模型，文本嵌入为 512 维

# # correlations on the dictionary entries
# 在全局加载一次 CLIP 处理器和模型。
# 注意：真正生成嵌入时 generate_clip_embeddings 会自己重新加载模型，这里的 processor/model 后面并未直接使用。
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)

### Last experiment, here we compare imagist to modern love poetry

# %%
# ===== 第 2 部分：基于嵌入向量的指标 =====
# entropy
# 计算一条嵌入向量的"熵"。e_formula 选择两种公式之一：
#   'orig' ：原代码的做法，先对嵌入做 softmax 变成概率分布，再算香农熵；
#   'paper'：论文第 12 页写的公式，p_i = |e_i| / Σ|e_j|（按绝对值归一化），再算香农熵。
# 两种结果都会保留（entropy_orig / entropy_paper），方便比较代码和论文是否一致。
def calculate_entropy(embedding,e_formula):
    if e_formula == 'orig':
        # Convert the embedding values to probabilities using softmax
        probabilities = softmax(embedding)
         # Calculate the entropy
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-9))  # Adding a small constant to avoid log(0)
    elif e_formula == 'paper':
        eps = 1e-9
        a = np.abs(np.asarray(embedding, dtype=float))   # 取各维绝对值
        total = a.sum()
        if total == 0:                      # all-zero vector: treat as uniform
            p = np.full(len(a), 1 / len(a))
        else:
            p = a / total # p_i = |e_i| / sum |e_j|, sums to 1
        entropy = -np.sum(p * np.log(p + eps)) # H(e) = -sum p_i log(p_i + eps)
    else: raise ValueError(f"unknown e_formula {e_formula}")   # 公式名写错时直接报错

    return entropy

# sparse ratio
# Function to compute the ratio of L1 to L2 norms
# 稀疏比 = L1 范数 / (√维度 × L2 范数)。
# 取值范围约在 (0, 1]：各维数值越均匀越接近 1，能量越集中在少数维度（越稀疏）越小。
def calculate_sparsity_ratio(embedding):
    embedding = np.array(embedding)
    l1_norm = np.sum(np.abs(embedding))  # L1 norm
    l2_norm = np.sqrt(np.sum(embedding**2))  # L2 norm
    sparsity_ratio = l1_norm / (np.sqrt(len(embedding)) * l2_norm)
    return sparsity_ratio

# %%

# ===== 第 3 部分：运行配置与路径 =====
# CONFIGS

# we can do the analysis on a poem basis or a sentence basis (default, sentences)
# 注意：小写的 analysis_basis 只在 get_merve()（原始数据分支）里使用；
# 下面大写的 ANALYSIS_BASIS 只用于嵌入缓存文件名。两者需要手动保持一致。
analysis_basis = "sentences"
print(f"Analysis basis: {analysis_basis}")
from pathlib import Path

# ---- change these per run ----
# 每次运行前修改这几项：
#   RUN_NAME：本次运行的名字，结果保存在 results/<RUN_NAME>/；
#   DATA_TAG：使用哪个版本的爱情诗数据（"orig" 原始 HuggingFace 数据 / "new" 清洗后 / "final" 最终版）；
#   ANALYSIS_BASIS：分析单位（这里只影响嵌入缓存文件名）；
#   DESCRIPTION：对本次运行的简短描述，写入 run_info.json 和 runs.csv。
RUN_NAME       = "tryrun"   # label for this try
DATA_TAG       = "new"                # which version of the text: "orig" or "new" or "final"
ANALYSIS_BASIS = "sentences"
DESCRIPTION = "tryout"
# ------------------------------
print(Path.cwd())   # 打印当前工作目录，便于确认从哪里运行
# 生成本次运行需要的所有路径（以仓库根目录为基准，即 fixed/ 的上一级）。
# 返回一个字典 p，键为用途名称，值为 Path 对象；同时创建需要的文件夹。
def make_paths(run_name, data_tag, basis, root = Path(__file__).resolve().parent.parent):
    data_dir = root / "data"                        # 诗歌原始数据
    dict_dir = root / "source" / "resources"        # 论文仓库里的心理语言学词典

    p = {
        "imagist_data": data_dir / "some_imagists.json",
        # 爱情诗数据：DATA_TAG 为 "final" 时用 modern_love_final.json，否则用 modern_love.json
        # （注意这里读取的是全局变量 DATA_TAG，而不是参数 data_tag）
        "love_data":    data_dir / ("new/modern_love_final.json"if DATA_TAG == "final" else "new/modern_love.json"),
        # 嵌入缓存文件名里带上分析单位和数据版本，避免不同版本之间误用旧缓存
        "imagist_emb":  root / "data/derived/embeddings" / f"imagists_{basis}_{data_tag}.json",
        "love_emb":     root / "data/derived/embeddings" / f"love_{basis}_{data_tag}.json",
        "run_dir":      root / "results" / run_name,

    }

    # 三本词典：MRC 意象性、Lancaster 感知运动（视觉）、Brysbaert 具体性
    p["mrc_dict"]          = dict_dir / "mrc_psychol_dict.json"
    p["sensorimotor_dict"] = dict_dir / "sensorimotor_norms_dict.json"
    p["concreteness_dict"] = dict_dir / "concreteness_brysbaert.json"
    # 本次运行的输出文件
    p["lines_csv"] = p["run_dir"] / "lines.csv"      # 每一行诗的所有指标
    p["tests_csv"] = p["run_dir"] / "tests.csv"      # 统计检验结果
    p["run_info"]  = p["run_dir"] / "run_info.json"  # 运行元信息
    p["figs"]      = p["run_dir"] / "figs"           # 图片文件夹
    # 确保嵌入缓存目录、运行目录和图片目录都存在
    for d in (p["imagist_emb"].parent, p["run_dir"], p["figs"]):
        d.mkdir(parents=True, exist_ok=True)
    return p

paths = make_paths(RUN_NAME, DATA_TAG, ANALYSIS_BASIS)

# %%
# ===== 第 4a 部分：原始爱情诗数据（仅 DATA_TAG == "orig" 时使用）=====
# and we get the poetry from:
# https://huggingface.co/datasets/merve/poetry
# 从 HuggingFace 的 merve/poetry 数据集取出 Modern + Love 类诗歌，
# 去掉与意象派重合的诗人，再按 analysis_basis 切成整首诗或诗行。
# 这一段基本沿用原论文代码的清洗规则，用来复现原始结果。返回的是纯文本列表。
def get_merve(poets): #get poets from df_img: poets = df_imag['poet'].unique()
    ds = load_dataset("merve/poetry")


    # make df
    df = pd.DataFrame(ds['train'])
    print(len(df))
    df.head()
    # # take only the modern
    df_modern = df[df['age'] == 'Modern']          # 只保留现代诗
    # take only the poems where type is Love
    df_love = df_modern[df_modern['type'] == 'Love']   # 只保留爱情主题
    print('len df', len(df_love))
    # remove any others that are in the imagist poets
    df_love = df_love[~df_love['author'].isin(poets)]  # 排除也出现在意象派语料中的诗人
    print('len df after filtering authors', len(df_love))

    # more cleaning needed from these cause they are scraped data




    # take poems
    # 分支一：以整首诗为单位。逐行清洗后再用空格拼回一首诗。
    if analysis_basis == "poems":
        units = df_love['content'].tolist()
        units_clean_merve = []
        for poem in units:
            poem = poem.replace('\r', '').strip()
            # split into lines first, then filter — do not replace \n before splitting
            lines = poem.split('\n')
            lines = [line.strip().replace('*', '').replace('  ', ' ') for line in lines]
            lines = [line for line in lines if line and line != ' ']
            lines = [line for line in lines if 'copyright' not in line.lower()]   # 去掉版权声明行
            lines = [line for line in lines if 'reprint' not in line.lower()]     # 去掉转载声明行
            if not lines:
                continue
            poem = ' '.join(lines).strip()
            if len(poem) < 5:          # 太短的不算一首诗
                continue
            units_clean_merve.append(poem)
        print('no. of poems', len(units_clean_merve))

    # take sentences
    # 分支二（默认）：以诗行为单位。按换行符拆开所有诗，再逐行过滤。
    elif analysis_basis == "sentences":
        # get all the sentences, seperated by \n
        units = df_love['content'].str.split('\n').explode().tolist()
        # save the author and poem names
        # （作者和诗名在这里被拆开计算，但后面没有用到，也没有返回）
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

        # 以下是一连串过滤条件（与原论文代码一致）：
        # remove emtpy strings
        units_clean_merve = [unit for unit in units_clean_merve if unit != '']
        # remove sentences that are just a number
        units_clean_merve = [unit for unit in units_clean_merve if not unit.isdigit()]
        # remove empty strings
        units_clean_merve = [unit for unit in units_clean_merve if unit != ' ']
        # 只保留长度大于 5 个字符的行（注意：意象派数据没有这条过滤，两组规则不对称）
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
# ===== 第 4b 部分：新的统一读取函数（两组数据用同一套清洗规则）=====
import re

# 把一首诗的原始文本切成诗行。
# 意象派数据用 "*" 标记行界（文本中的 \n 只是排版换行，先替换成空格）；
# 爱情诗新数据直接用 "\n" 作为行界。
def split_poem(raw: str, line_sep: str) -> list[str]:
    """按数据源各自的行分隔符切成诗行（两组数据唯一不同的地方）"""
    if line_sep == '*':
        raw = raw.replace('\n', ' ')   # imagists 的行界由 * 标记
    return raw.split(line_sep)

# 清洗单行：把 \r 和不换行空格换成普通空格，再把连续空白压缩成一个空格并去掉首尾空白。
def clean_line(s: str) -> str:
    s = s.replace('\r', ' ').replace('\u00a0', ' ')
    return re.sub(r'\s+', ' ', s).strip()

# 判断一行是否保留：
#   - 必须含至少一个英文字母（排除空行、纯数字的节号、纯标点）；
#   - 不含 copyright / reprint（版权或转载说明）；
#   - 长度不少于 min_chars（默认 0，即不限制长度）。
def is_valid(s: str, min_chars: int = 0) -> bool:
    if not re.search(r'[A-Za-z]', s):          # 空串、纯数字、纯标点
        return False
    low = s.lower()
    if 'copyright' in low or 'reprint' in low:
        return False
    return len(s) >= min_chars

# 读取 {诗人: [{title, poem}, ...]} 格式的 JSON，返回一个 DataFrame。
# 每一行包含：组别 group、诗人 poet、诗名 title、诗编号 poem_id、行号 line_no、文本 text。
# analysis_basis="lines" 时每行诗一条记录；="poems" 时整首诗拼成一条记录（line_no 固定为 0）。
# 保留诗人和诗名信息，是为了以后能做按诗人/按诗的分组分析（例如混合模型）。
# 注意：调用时没有传入 ANALYSIS_BASIS，因此目前总是按默认值 "lines" 运行。
def load_poetry(path, group, line_sep, analysis_basis="lines", min_chars=0):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    rows = []
    poem_id = 0          # 跨诗人连续编号，每首有效的诗一个 id
    for poet, works in data.items():
        for w in works:
            lines = [clean_line(l) for l in split_poem(w['poem'], line_sep)]
            lines = [l for l in lines if is_valid(l, min_chars)]
            if not lines:      # 整首诗清洗后为空则跳过，不占用 poem_id
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

# ===== 第 5 部分：获取 CLIP 嵌入（有缓存则读缓存）=====
# now we get the embeddings for each sentence

# check if embeddings are already saved
# genre 为 'imagist' 或 'love'，对应 paths 里的 imagist_emb / love_emb。
# 缓存文件不存在：调用 CLIP 生成嵌入并保存为 JSON；
# 缓存文件存在：直接读取，省去重新计算的时间。
# 最后用 assert 检查嵌入条数和文本行数是否一致——数据清洗规则改了但缓存没删时会在这里报错。
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

# ===== 第 6 部分：计算每行的所有指标 =====
# 下面三引号里的是旧版 score_df，已经停用（作为字符串保留，不会执行），新版在其后。

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
# 词典配置：输出列名 -> (paths 中词典路径的键, 词典条目里分数字段的名字)
DICTS = {
    'imag':     ('mrc_dict',          'imag'),          # MRC 意象性
    'visual':   ('sensorimotor_dict', 'Visual.mean'),   # 视觉感知强度
    'concrete': ('concreteness_dict', 'Conc.M'),        # 具体性
}

# 新版打分函数：在 load_poetry 得到的 DataFrame 上直接加指标列，保留诗人/诗名等元信息。
def score_df(df, emb, paths):
    """df: rows from load_poetry (group, poet, title, poem_id, line_no, text); emb: one vector per row."""
    emb = np.asarray(emb)
    out = df.reset_index(drop=True).copy()

    # embedding measures
    # 嵌入指标：L2 范数、两种熵、各维方差、稀疏比
    out['norm']          = np.linalg.norm(emb, axis=1)
    out['entropy_orig']  = [calculate_entropy(e, 'orig')  for e in emb]
    out['entropy_paper'] = [calculate_entropy(e, 'paper') for e in emb]
    out['variance']      = emb.var(axis=1)
    out['sparsity']      = [calculate_sparsity_ratio(e) for e in emb]

    # dictionary measures
    # 词典指标：每本词典生成两列——总分（如 imag）和按"词典中查得到的词数"平均后的分数（如 imag_norm）
    texts = out['text'].tolist()
    for name, (path_key, score_key) in DICTS.items():
        out[name], out[f'{name}_norm'] = get_dict_scores(
            texts, paths[path_key], score_key, 'lemma_', normalize_by_tokens=False)

    # drop repeated lines; the paper's counts exclude them
    # 去重：同一文本只保留第一次出现的那一行（论文的行数 1,195 / 1,126 就是去重后的）
    return out[~out['text'].duplicated()].reset_index(drop=True)

# 主流程：读取两组数据。
# 意象派总是从 some_imagists.json 读取；
# 爱情诗：DATA_TAG 不是 "orig" 时读新数据，否则走原始的 get_merve()。
# 注意：get_merve() 返回的是文本列表而不是 DataFrame，所以 "orig" 分支下面的 df_love['text'] 会出错。
df_imag = load_poetry(paths['imagist_data'], 'imagist', line_sep='*')
df_love = load_poetry(paths['love_data'], 'modern_love', line_sep='\n') if DATA_TAG != "orig" else get_merve(df_imag['poet'].unique())

# 生成/读取嵌入并打分，然后把两组的逐行结果合并保存到 lines.csv
df_sentences      = score_df(df_imag, get_emb('imagist', df_imag['text'].tolist()), paths)
df_sentences_love = score_df(df_love, get_emb('love',    df_love['text'].tolist()), paths)
pd.concat([df_sentences, df_sentences_love]).to_csv(paths['lines_csv'], index=False)

# %%
# ===== 第 7 部分：参考例句 =====
# we get imageable and non imageable sentences as reference points
# cocnat the two dfs
df_sentences_all = pd.concat([df_sentences, df_sentences_love])

# make pd print all
pd.set_option('display.max_colwidth', None)

# print the first 10 most imageable sentences in both dfs
# 注意：这里是按 imag_norm 升序排列，所以打印的其实是意象性"最低"的 300 行，
# 注释里说的 "most imageable" 和 "first 10" 与实际代码不符（沿用原代码）。
print('most imageable')
# make it a list
imageable = df_sentences_all.sort_values('imag_norm', ascending=True)['text'].tolist()
# print the first 10
for sent in imageable[:300]:
    print(sent)
    print('\n')

# %%
# we choose these sentences (from topmosts and bottommosts) as examples
# 论文图 3 中的两条参考线对应的例句，是直接写死的字符串（代码中并没有计算"前十/后十"的排名）。
imag_example = "Homespun, dyed butternuts dark gold color."
non_imag_example = "Of insidious intent"

# retrieve the measures for these from the df
# 在合并后的表里找到这两句对应的行；如果新数据里没有这句，得到的是空表
print('imag example exists:', imag_example)
imag_metrics = df_sentences_all[df_sentences_all['text'] == imag_example]
print('non-imag example exists:', non_imag_example)
non_imag_metrics = df_sentences_all[df_sentences_all['text'] == non_imag_example]

# %%
# ===== 第 8 部分：分布图 =====
# make distributions of each group for each measure
# 要比较的指标列表（后面的画图和统计检验都用这个列表）
measures = ['norm', 'entropy_orig','entropy_paper', 'sparsity',
            'imag_norm', 'visual_norm', 'concrete_norm']


# 为每个指标画一张子图：两组的密度直方图 + KDE 曲线叠在一起（蓝 = 意象派，红 = 爱情诗）。
# examples 是 [(标签, 行数据, 颜色), ...]，会在每个子图上画出该例句的竖虚线。
# 图片先保存到 results/<RUN_NAME>/figs/<name>，再显示，并返回保存路径。
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
    handles, labels = axes.flat[0].get_legend_handles_labels()   # 用第一个子图的图例作为整张图的图例
    fig.legend(handles, labels, loc='lower right')
    fig.tight_layout()
    out = paths['figs'] / name
    fig.savefig(out, dpi=200)                    # save first, then show
    plt.show()
    return out
# 实际调用：画出七个指标的两组分布图，并在图上标出前面那两条参考例句
# （蓝色虚线 = 高意象例句，红色虚线 = 低意象例句；若该例句不在本次数据中则自动跳过）。
fig_path = plot_distributions(df_sentences, df_sentences_love, measures, paths,
                              examples=[('Imageable', imag_metrics, 'blue'),
                                        ('Abstract', non_imag_metrics, 'red')])
print('figure saved to', fig_path)
# print len of both
print('len imag', len(df_sentences))
print('len love', len(df_sentences_love))
# %%
# ===== 第 9 部分：统计检验 =====
# do the t-test & mannwhitneyu test
# Cohen's d：两组均值差除以合并标准差，衡量效应大小（与样本量无关，这是论文表 6 缺少的信息）
def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1-1)*a.var(ddof=1) + (n2-1)*b.var(ddof=1)) / (n1+n2-2))
    return (a.mean() - b.mean()) / sp

alpha = 0.05
bonf = alpha / len(measures)     # Bonferroni 校正后的显著性阈值（多重比较）
rows = []
# 对每个指标：去掉缺失值后做 t 检验和 Mann-Whitney U 检验，
# 并记录样本量、均值、标准差、Cohen's d、AUC 以及是否通过 Bonferroni 校正。
for m in measures:
    a = df_sentences[m].dropna()          # 意象派
    b = df_sentences_love[m].dropna()     # 爱情诗
    t, p = ttest_ind(a, b)
    U, pU = mannwhitneyu(a, b)
    rows.append(dict(measure=m, n_imagist=len(a), n_love=len(b),
                     mean_imagist=a.mean(), sd_imagist=a.std(),
                     mean_love=b.mean(), sd_love=b.std(),
                     t=t, p_t=p, U=U, p_U=pU,
                     cohens_d=cohens_d(a, b),
                     # AUC = U/(n1·n2)：随机抽一行意象派诗、一行爱情诗，前者数值更大的概率；0.5 表示两组无差别
                     auc=U / (len(a) * len(b)),   # P(Imagist > Love)
                     bonferroni_alpha=bonf, sig_t=p < bonf, sig_U=pU < bonf))
tests = pd.DataFrame(rows)
tests.to_csv(paths['tests_csv'], index=False, float_format='%.6g')   # 保存到 tests.csv
print(tests[['measure', 't', 'p_t', 'cohens_d', 'auc', 'sig_t']].to_string(index=False))


# %%
# get the numbers of means and stds per group
# 打印两组所有数值列的描述统计（计数、均值、标准差、分位数等）
print('Imagists')
print(df_sentences.describe())
print('Modern Love')
print(df_sentences_love.describe())

# %%
print('All done:)')

# ===== 第 10 部分：记录本次运行信息 =====
# 目的：每次运行都能追溯——用了哪份数据、哪个代码版本、哪些软件版本、得到什么结果。
import subprocess, datetime, platform, hashlib
from importlib.metadata import version

# 计算文件的 SHA-256 指纹，用来确认数据文件是否被改动过
def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()

root = Path(__file__).resolve().parent.parent
# 在仓库根目录执行 git 命令并返回输出（用于记录 commit 号和是否有未提交的改动）
git = lambda *a: subprocess.run(['git', *a], cwd=root, capture_output=True, text=True).stdout.strip()

# 汇总运行元信息：配置、时间、数据文件路径与指纹、模型、git 状态、Python 和各库版本、两组行数
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
    json.dump(info, f, indent=2)       # 写入 results/<RUN_NAME>/run_info.json

# 总索引 results/runs.csv：每次运行追加一行，包含基本信息和每个指标的 t、Cohen's d、AUC，
# 方便横向比较多次运行。文件不存在时才写表头。
idx = root / 'results' / 'runs.csv'
row = {k: info[k] for k in ['run_name', 'date', 'description', 'data_tag', 'git_commit', 'git_dirty']}
row |= {f'{c}_{r.measure}': r[c] for _, r in tests.iterrows() for c in ('t', 'cohens_d', 'auc')}
pd.DataFrame([row]).to_csv(idx, mode='a', header=not idx.exists(), index=False)
# %%
## gird search
# 网格搜索的计划（尚未实现）：数据版本 × 分析单位 × 熵公式 的所有组合。
# 目前只定义了这三个列表，没有循环去运行它们。
corpus = ['orig','new','final']
basis = ['poem','sentence']
e_formula = ['orig','paper']
