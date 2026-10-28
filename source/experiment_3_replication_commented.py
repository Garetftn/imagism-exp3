# %%
# =============================================================================
# Experiment III replication: Imagist poems vs Modern Love poems
# 实验三复现：意象派诗歌 vs 现代爱情诗
#
# Pipeline / 流程:
#   exp_start()
#     -> load_corpora()          load + clean both corpora, one row per poem line
#                                读取并清洗两个语料，每行诗一条记录
#     -> get_metrics()           CLIP metrics (norm, entropy, variance, sparsity)
#                                + dictionary metrics (imag, visual, concrete)
#                                计算 CLIP 指标与词典指标（有缓存则直接读取）
#     -> plot_distributions()    Figure 3 style histograms, saved to figs/
#                                画出类似论文图 3 的分布图，保存到 figs/
#     -> test()                  t-test, Mann-Whitney U, Cohen's d, AUC
#                                t 检验、Mann-Whitney U 检验、Cohen's d、AUC
#     -> take_record()           append one JSON line per run to results/exp3_runs.jsonl
#                                每次运行在 results/exp3_runs.jsonl 追加一行记录
#
# "L-numbers" in comments (e.g. L168) refer to the ORIGINAL experiment_3.py.
# 注释中的 "L+数字"（如 L168）指的是原始 experiment_3.py 的行号。
# =============================================================================
import re, json
import pandas as pd
import numpy as np
import os
from functions import generate_clip_embeddings, get_dict_scores
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from datasets import load_dataset


def load_love(exclude_poets=(), start_id=0, group='modern_love'):
    """merve/poetry -> Modern Love lines, cleaned exactly like experiment_3.py (L168-230).
    Same columns as load_poetry: group, poet, title, poem_id, line_no, text

    从 Hugging Face 的 merve/poetry 数据集读取"现代爱情诗"，并按原始
    experiment_3.py（L168-230）完全相同的规则清洗成一行一条记录。
    输出列与 load_poetry 相同：group, poet, title, poem_id, line_no, text。

    Args / 参数:
        exclude_poets: poets to drop (exact name match), normally the Imagist poets,
                       so nobody appears in both groups.
                       要排除的诗人（姓名完全匹配），通常是意象派诗人，避免同一诗人出现在两组。
        start_id:      first poem_id, so ids continue after the Imagist poems.
                       第一首诗的 poem_id，使编号接在意象派诗歌之后、不重复。
        group:         label written into the 'group' column.
                       写入 group 列的组名。

    Returns / 返回:
        pd.DataFrame, one row per kept line. 每个保留下来的诗行一行。
    """
    # Download/load the dataset and convert to a DataFrame
    # 下载/读取数据集并转成 DataFrame
    merve = load_dataset('merve/poetry', split='train').to_pandas()
    # Keep only Modern-age Love poems / 只保留"现代"时期的"爱情"类诗歌
    merve = merve[(merve['age'] == 'Modern') & (merve['type'] == 'Love')]   # L168, L170
    # Drop poets who are also in the Imagist corpus / 去掉同时出现在意象派语料里的诗人
    merve = merve[~merve['author'].isin(list(exclude_poets))]                # L173: exact match

    rows = []
    # One poem per iteration; poem_id counts up from start_id
    # 每次循环处理一首诗；poem_id 从 start_id 开始递增
    for pid, (_, r) in enumerate(merve.iterrows(), start=start_id):
        kept = []
        # Split the poem into lines and clean each one
        # 把整首诗按换行拆成诗行，逐行清洗
        for line in r['content'].split('\n'):                                # L201
            # Collapse double spaces, strip ends / 合并双空格并去掉首尾空白
            line = line.replace('\n', ' ').replace('  ', ' ').strip()        # L212
            # Remove asterisks (section markers) / 去掉星号（分节符）
            line = line.replace('*', '').strip()                             # L213-214
            # Skip empty lines and lines that are only a number (e.g. stanza numbers)
            # 跳过空行和纯数字行（如节号）
            if line == '' or line.isdigit() or line == ' ':                  # L219-223
                continue
            # Skip very short lines (5 characters or fewer)
            # 跳过很短的行（不超过 5 个字符）
            if not len(line) > 5:                                            # L224
                continue
            # Remove carriage returns / 去掉回车符
            line = line.replace('\r', '')                                    # L226
            # Skip copyright / reprint notices / 跳过版权或转载声明
            if re.search('copyright', line, re.I) or re.search('reprint', line, re.I):  # L228-230
                continue
            kept.append(line)
        # line_no is the position among the KEPT lines (after filtering)
        # line_no 是过滤之后保留诗行中的序号
        for i, line in enumerate(kept):
            rows.append(dict(group=group, poet=r['author'], title=r['poem name'],
                             poem_id=pid, line_no=i, text=line))
    # Fixed column order so it matches load_poetry (checked in load_corpora)
    # 固定列顺序，与 load_poetry 一致（load_corpora 中有 assert 检查）
    return pd.DataFrame(rows, columns=['group', 'poet', 'title', 'poem_id', 'line_no', 'text'])







def clean_line(line):
    """把一行诗里的任意连续空白归一为单个空格，并去掉首尾空白
    Collapse any run of whitespace in a line to a single space and strip both ends.

    Args / 参数:  line (str) 一行原始诗句 / one raw line
    Returns / 返回: str 清洗后的诗句 / the cleaned line
    """
    return re.sub(r'\s+', ' ', line).strip()

def is_valid(line):
    """判断一行诗句应不应该被保留
    Decide whether a line should be kept.

    A line is kept only if it contains at least one Latin letter.
    只有至少包含一个英文字母的行才保留。

    Returns / 返回: bool  True = keep 保留, False = drop 丢弃
    """
    if not re.search(r'[A-Za-z]', line):    # 空串、纯数字、纯标点 / empty, digits only, punctuation only
        return False
    return True


def load_poetry(path, group, line_sep):    # group: 即诗歌的分组 imagist, modern_love / the poem group
    """读取并清洗诗歌 JSON。analysis_basis 默认 "lines"（一行一条记录），整首诗模式调用时传 "poems"。
    Read and clean a poetry JSON file into one row per line.



    Args / 参数:
        path:     JSON file of the form {poet: [{"title": ..., "poem": ...}, ...]}
                  JSON 文件，格式为 {诗人: [{"title": ..., "poem": ...}, ...]}
        group:    label for the 'group' column ('imagist' or 'modern_love')
                  group 列的组名（'imagist' 或 'modern_love'）
        line_sep: string that separates lines inside "poem" ('*' for the Imagist
                  file, '\\n' for the Love files)
                  "poem" 字段里分隔诗行的字符串（意象派文件用 '*'，爱情诗文件用 '\\n'）

    Returns / 返回:
        pd.DataFrame with columns group, poet, title, poem_id, line_no, text
        列为 group, poet, title, poem_id, line_no, text 的 DataFrame
    """

    with open(path, encoding='utf-8') as f:
        data = json.load(f)      #把文件解析成字典：{诗人: [{title, poem}, ...]}。 / parse into {poet: [{title, poem}, ...]}

    rows = []
    poem_id = 0     # running poem counter across all poets / 跨所有诗人的诗歌编号
    for poet, works in data.items():
        for w in works:     #w 是列表中的一个元素，也就是一首诗的字典，如 {"title": "THE POOL", "poem": "..."} / w is one poem's dict
            # Split into lines and normalise whitespace / 拆分诗行并规范空白
            lines = [clean_line(line) for line in w['poem'].split(line_sep)]
            # Drop lines without any letters / 去掉不含字母的行
            lines = [line for line in lines if is_valid(line)]
            # One row per kept line / 每个保留的诗行生成一条记录
            for i, line in enumerate(lines):
                rows.append(dict(group=group, poet=poet, title=w['title'],
                poem_id=poem_id, line_no=i, text=line))
            poem_id += 1
    return pd.DataFrame(rows)

# relative paths
# 相对路径：脚本需在包含 data/、resources/ 的目录下运行
# Relative paths: run the script from the folder that contains data/ and resources/




# %%
#df_all.to_csv(f'data/cleaned_{basis}_preview.csv', index=False, encoding='utf-8-sig')

"""为每一行诗产生 CLIP 向量，并计算 L2 范数、熵、方差和稀疏度等指标。
我把这些指标都加到 df_all 里了，最后保存为 CSV。"""
def add_clip_metrics(
    df,
    e_formula,
    embeddings=None,
    model_name="openai/clip-vit-base-patch32"

):
    """Compute CLIP embeddings for each line and add embedding-based metrics.
    为每一行诗计算 CLIP 向量，并加入基于向量的指标。

    Added columns / 新增列:
        norm      L2 norm of the embedding / 向量的 L2 范数
        entropy   entropy of the embedding treated as a distribution (see e_formula)
                  把向量看作概率分布后的熵（见 e_formula）
        variance  variance of the embedding's components / 向量各分量的方差
        sparsity  L1 / (sqrt(n) * L2); range (0, 1], higher = LESS sparse
                  (1 = all components equal in size)
                  L1 / (sqrt(n) * L2)，取值 (0, 1]，越大越"不稀疏"（1 表示各分量大小相同）

    Args / 参数:
        df:         lines DataFrame with a 'text' column / 含 text 列的诗行表
        e_formula:  'paper' -> p_i = |e_i| / sum|e_j|, as defined in the paper (p.12)
                    'paper' -> 论文（第 12 页）的定义 p_i = |e_i| / Σ|e_j|
                    anything else (e.g. 'orig') -> softmax(e), as in the upstream code
                    其他值（如 'orig'）-> softmax(e)，即原始代码的做法
        embeddings: cached array (n_lines x dim); None = run CLIP now
                    已缓存的向量数组（行数 x 维度）；None 表示现在运行 CLIP 生成
        model_name: Hugging Face CLIP model / 使用的 CLIP 模型

    Returns / 返回:
        (out, embeddings): DataFrame with new columns, and the embeddings (for caching)
        (out, embeddings)：加了新列的 DataFrame，以及向量（供缓存保存）
    """
    out = df.copy()
    # Step 1: get embeddings (from cache or by running CLIP)
    # 第 1 步：获得向量（读缓存或运行 CLIP）
    if embeddings is None:                                   # no cache -> run CLIP / 无缓存则运行 CLIP
        embeddings = generate_clip_embeddings(out["text"].tolist(), model_name=model_name)
    # Cache must have exactly one vector per line / 缓存必须与诗行一一对应
    assert len(embeddings) == len(out), "cached embeddings don't match this dataframe"

    texts = out["text"].tolist()


    # Step 2: norms / 第 2 步：计算范数
    abs_embeddings = np.abs(embeddings)
    # 每个向量的 L1 和 L2 范数 / L1 and L2 norm of each vector
    l1_norms = abs_embeddings.sum(axis=1)
    l2_norms = np.linalg.norm(embeddings, axis=1)

    # 论文方法：p_i = |x_i| / Σ|x_j|
    # Paper method: p_i = |x_i| / sum|x_j| (each row sums to 1)
    probabilities = abs_embeddings / l1_norms[:, None]

    out["norm"] = l2_norms

    # Step 3: entropy, using the chosen formula / 第 3 步：按所选公式计算熵
    if e_formula == 'paper':
        # Shannon entropy of |e|/sum|e|; 1e-9 avoids log(0)
        # 对 |e|/Σ|e| 计算香农熵；加 1e-9 防止 log(0)
        out["entropy"] = -np.sum(
            probabilities * np.log(probabilities + 1e-9),
            axis=1
        )
    else:  #加上源代码方法 / upstream code method: softmax, overwrites probabilities
        probabilities = softmax(embeddings, axis=1)
        out["entropy"] = -np.sum(probabilities * np.log(probabilities + 1e-9), axis=1)

    # Step 4: variance and sparsity / 第 4 步：方差与稀疏度
    out["variance"] = np.var(embeddings, axis=1)

    # sparsity = L1 / (sqrt(dim) * L2); higher = less sparse
    # 稀疏度 = L1 / (sqrt(维度) * L2)；数值越大越不稀疏
    out["sparsity"] = (
        l1_norms
        / (np.sqrt(embeddings.shape[1]) * l2_norms)
    )

    return out, embeddings

"""为每一行诗计算字典指标：imag、visual、concrete。"""
def add_dictionary_metrics(
    df,
    mrc_path="resources/mrc_psychol_dict.json",
    visual_path="resources/sensorimotor_norms_dict.json",
    concrete_path="resources/concreteness_brysbaert.json"
):
    """Add dictionary-based scores for each line: imageability, visuality, concreteness.
    为每一行诗加入基于词典的分数：意象性、视觉性、具体性。

    For each dictionary two columns are added / 每个词典新增两列:
        <name>       sum of the scores of matched words (lemmas)
                     匹配到的词（词元）分数之和
        <name>_norm  average score over matched words (NaN if no word matched)
                     匹配词的平均分（一个词都没匹配到时为 NaN）
    Dictionaries / 词典:
        imag     MRC psycholinguistic database, key "imag" / MRC 心理语言学数据库
        visual   Lancaster sensorimotor norms, key "Visual.mean" / Lancaster 感觉运动常模
        concrete Brysbaert concreteness ratings, key "Conc.M" / Brysbaert 具体性评分

    Since <name>_norm = <name> / number of matched words, the matched-word count
    can be recovered as <name> / <name>_norm (used in plot_distributions).
    因为 <name>_norm = <name> / 匹配词数，所以匹配词数可由 <name> / <name>_norm 反推
    （plot_distributions 中用到）。

    Returns / 返回: a copy of df with six new columns / 加了六个新列的 df 副本
    """
    out = df.copy()
    texts = out["text"].tolist()

    # column name -> (dictionary file, score key inside that file)
    # 列名 -> （词典文件，文件中的分数键）
    dictionaries = {
        "imag": (mrc_path, "imag"),
        "visual": (visual_path, "Visual.mean"),
        "concrete": (concrete_path, "Conc.M"),
    }

    for name, (path, score_key) in dictionaries.items():
        # Look up each lemma (spaCy lemma_) in the dictionary;
        # returns per-line totals and per-line averages over matched words
        # 用 spaCy 的词元（lemma_）查词典；返回每行的总分和匹配词平均分
        totals, averages = get_dict_scores(
            texts,
            path,
            score_key,
            token_attr="lemma_",
            normalize_by_tokens=False
        )

        out[name] = totals
        out[f"{name}_norm"] = averages

    return out


# %%
# make distributions of each group for each measure
# 为每个指标画出两组的分布


def plot_distributions(df_all, run_id, base_line,measures,settings=None):
    """Draw Figure 3 style histograms (Imagist vs Modern Love) for each measure.
    为每个指标画出类似论文图 3 的直方图（意象派 vs 现代爱情诗）。

    Each panel shows both groups' density histograms with KDE, plus dashed
    vertical lines marking example lines.
    每个子图显示两组的密度直方图和核密度曲线，并用虚线标出示例诗行的位置。

    Args / 参数:
        df_all:    all lines with metrics / 含全部指标的所有诗行
        run_id:    used as the file name figs/<run_id>.png / 用作输出文件名
        base_line: which "computed" example lines to add / 选择如何计算示例诗行
                   'computed' -> line with highest / lowest imag_norm among all scored lines
                                 在所有有分数的诗行中取 imag_norm 最高/最低的行
                   'three'    -> same, but only lines with >= 3 MRC-matched words
                                 同上，但只考虑至少匹配 3 个 MRC 词的诗行
                   The paper's two hard-coded lines are always drawn.
                   论文中写死的两行示例总是会画出。
        measures:  up to 6 column names, one per panel (2 x 3 grid)
                   最多 6 个列名，每个子图一个（2 x 3 网格）
        settings:  optional dict printed in a box on the right of the figure
                   可选的设置字典，显示在图右侧的文本框中

    Returns / 返回: path of the saved PNG / 保存的 PNG 路径
    """
    # Split by group / 按组拆分
    df_imag = df_all[df_all['group'] == 'imagist']
    df_love = df_all[df_all['group'] == 'modern_love']
    # always: the paper's fixed lines
    # 总是画出论文里固定的两行示例（意象性高 / 抽象）
    examples = [('Imageable (paper)', df_all[df_all['text'] == "Homespun, dyed butternuts dark gold color."], 'blue'),
                ('Abstract (paper)',  df_all[df_all['text'] == "Of insidious intent"], 'red')]

    if base_line == 'computed':                       # add this run's real extremes / 加入本次运行真正的极值行
        # all lines that have an MRC imageability score / 所有有 MRC 意象性分数的行
        scored = df_all.dropna(subset=['imag_norm'])

    if base_line == 'three': #Require enough matched words / 要求匹配词足够多
        scored = df_all.dropna(subset=['imag_norm']).copy()
        # number of matched MRC words = total / average / 匹配的 MRC 词数 = 总分 / 平均分
        scored['n_matched'] = (scored['imag'] / scored['imag_norm']).round()
        # keep lines with at least 3 matched words, so one word (e.g. "of") can't decide the extreme
        # 只保留至少匹配 3 个词的行，避免单个词（如 "of"）决定极值
        scored = scored[scored['n_matched'] >= 3]

    # Add the most / least imageable line of this run / 加入本次运行意象性最高和最低的行
    examples += [('Imageable (computed)', scored.loc[[scored['imag_norm'].idxmax()]], 'purple'),
                     ('Abstract (computed)',  scored.loc[[scored['imag_norm'].idxmin()]], 'pink')]

    # One panel per measure / 每个指标一个子图
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, measure in zip(axes.flat, measures):
        # density histograms + KDE for both groups / 两组的密度直方图和核密度曲线
        sns.histplot(df_imag[measure], ax=ax, label='Imagist', color='blue', kde=True, alpha=0.2, stat='density',linewidth=0.1)
        sns.histplot(df_love[measure], ax=ax, label='Modern Love', color='red', kde=True, alpha=0.2, stat='density',linewidth=0.1)
        # dashed line at each example line's value / 在每个示例诗行的取值处画虚线
        for label, row, colour in examples:
            if len(row):                              # skip if the line isn't in this run / 本次数据里没有该行则跳过
                ax.axvline(row[measure].values[0], color=colour, linestyle='--', label=label)
        ax.set_title(measure)
    # Settings box on the right / 右侧显示本次运行的设置
    if settings:
        text = '\n'.join(f'{k}: {v}' for k, v in settings.items())
        fig.text(0.885, 0.5, text, fontsize=9, family='monospace', va='center',
                 bbox=dict(boxstyle='round', facecolor='whitesmoke', edgecolor='lightgrey'))


    # One shared legend at the bottom, taken from the first panel
    # 在底部放一个共享图例（取自第一个子图）
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4)
    # leave space at the bottom (legend) and right (settings box) / 为底部图例和右侧设置框留出空间
    fig.tight_layout(rect=(0, 0.06, 0.88, 1))
    os.makedirs('figs', exist_ok=True)
    out = f'figs/{run_id}.png'
    fig.savefig(out, dpi=200)                         # save first, then show / 先保存再显示
    #plt.show()
    return out






def load_corpora(imag_corpus,love_corpus):
    """Load and clean both corpora for one run.
    为一次运行读取并清洗两个语料。

    Args / 参数:
        imag_corpus: 'orig' -> data/some_imagists.json (the original file)
                     other  -> data/some_imagists_<imag_corpus>.json
                     'orig' 使用原始文件，其他值读取对应名字的新文件
        love_corpus: 'orig' -> merve/poetry via load_love (original cleaning)
                     other  -> data/modern_love_<love_corpus>.json
                     'orig' 通过 load_love 从 merve/poetry 读取（原始清洗规则），
                     其他值读取对应名字的 JSON 文件

    Returns / 返回: (df_imag, df_love), one row per line / 每行诗一条记录
    """
    # Imagist corpus: lines are separated by '*' / 意象派语料：诗行以 '*' 分隔
    if imag_corpus == 'orig':
        IMAG_PATH = f'data/some_imagists.json'
    else:
        IMAG_PATH = f'data/some_imagists_{imag_corpus}.json'
    df_imag = load_poetry(IMAG_PATH, 'imagist', line_sep='*')

    # Love corpus / 爱情诗语料
    if love_corpus =='orig':
        # exclude Imagist poets and continue poem_id numbering after the Imagist poems
        # 排除意象派诗人，并让 poem_id 接着意象派诗歌继续编号
        df_love = load_love(exclude_poets=df_imag['poet'].unique(),
                        start_id=df_imag['poem_id'].max() + 1)
        # both tables must have identical columns before concatenation
        # 合并前两表的列必须完全一致
        assert list(df_love.columns) == list(df_imag.columns)
    else:
        # new Love files separate lines with newlines / 新的爱情诗文件用换行分隔诗行
        LOVE_PATH = f'data/modern_love_{love_corpus}.json'
        df_love = load_poetry(LOVE_PATH, 'modern_love', line_sep='\n')
    return df_imag,df_love




    #print(len(df_all))
    #df_all = df_all.drop_duplicates(subset=['group','text'])
    #print(len(df_all))
def get_metrics(run_id,e_formula,df_imag,df_love,imag_corpus,love_corpus):
    """Return all lines with all metrics, using caches where possible.
    返回带全部指标的所有诗行，尽量使用缓存。

    Caching / 缓存:
        data/measures/<run_id>_metrics.csv  finished metrics table for this run;
                                            if present it is returned immediately
                                            本次运行的完整指标表；存在则直接返回
        data/embeddings/love_<love_corpus>_embeddings.npy
        data/embeddings/imag_<imag_corpus>_embeddings.npy
                                            CLIP embeddings per corpus, shared by all
                                            runs with the same corpus
                                            每个语料的 CLIP 向量，相同语料的运行共用
        Caution: the .npy cache is keyed only by corpus name; if the corpus file
        or its cleaning changes, delete the old .npy (the assert in
        add_clip_metrics only catches a change in the number of lines).
        注意：.npy 缓存只按语料名区分；若语料文件或清洗规则改变，需删除旧的 .npy
        （add_clip_metrics 中的 assert 只能发现行数变化）。

    Returns / 返回: df_all, Imagist rows followed by Love rows
                    df_all，意象派行在前，爱情诗行在后
    """
    # Step 1: whole run already computed? / 第 1 步：本次运行是否已算过？
    metrics_path = f"data/measures/{run_id}_metrics.csv"
    if os.path.exists(metrics_path):
        print('scores loaded')
        return pd.read_csv(metrics_path)

    # Step 2: load cached embeddings if they exist / 第 2 步：如有缓存向量则读取
    #os.makedirs('data/embeddings', exist_ok=True)
    love_path = f"data/embeddings/love_{love_corpus}_embeddings.npy"
    imag_path = f"data/embeddings/imag_{imag_corpus}_embeddings.npy"

    love_emb = np.load(love_path) if os.path.exists(love_path) else None
    imag_emb = np.load(imag_path) if os.path.exists(imag_path) else None

    # Step 3: CLIP metrics (runs CLIP only where no cache was found), then save embeddings
    # 第 3 步：计算 CLIP 指标（只有没有缓存时才运行 CLIP），然后保存向量
    df_love, love_emb = add_clip_metrics(df_love, e_formula, love_emb)   # always runs / 总会执行
    df_imag, imag_emb = add_clip_metrics(df_imag, e_formula, imag_emb)
    np.save(love_path, love_emb)
    np.save(imag_path, imag_emb)

    # Step 4: combine both groups and add dictionary metrics, then cache the table
    # 第 4 步：合并两组，加入词典指标，并缓存结果表
    print('computing the scores')
    df_all = pd.concat([df_imag, df_love], ignore_index=True)
    df_all = add_dictionary_metrics(df_all)
    df_all.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    return df_all

    # unreachable (after return); fig_path is not defined here / 在 return 之后，永远不会执行
    print('figure saved to', fig_path)

def test(df_imag,df_love,measures):
    """Compare the two groups on each measure and print a results table.
    在每个指标上比较两组，并打印结果表。

    For each measure / 对每个指标:
        t, p        independent two-sample t-test (equal variances, as scipy's default)
                    独立样本 t 检验（scipy 默认，假设方差相等）
        U, p_U      Mann-Whitney U test (two-sided) / Mann-Whitney U 检验（双侧）
        Cohens d    standardised mean difference, (mean_imag - mean_love) / pooled SD
                    标准化均值差，(意象派均值 - 爱情诗均值) / 合并标准差
        auc         U / (n_imag * n_love) = probability that a random Imagist line
                    scores higher than a random Love line (0.5 = no difference)
                    随机一行意象派诗句得分高于随机一行爱情诗句的概率（0.5 表示无差异）
        sig_t/sig_U significant after Bonferroni correction (alpha = 0.05 / number of measures)
                    经 Bonferroni 校正后是否显著（alpha = 0.05 / 指标数）

    Lines with NaN for a measure (e.g. no dictionary word matched) are dropped.
    某指标为 NaN 的行（如没有匹配到词典词）会被去掉。

    Returns / 返回: pd.DataFrame, one row per measure / 每个指标一行
    """
    # Bonferroni-corrected significance level / Bonferroni 校正后的显著性水平
    bonferroni_alpha = 0.05 / len(measures)
    rows = []
    for measure in measures:
        # drop missing values / 去掉缺失值
        a = df_imag[measure].dropna()
        b = df_love[measure].dropna()

        t, p = ttest_ind(a, b)
        U, p_U = mannwhitneyu(a, b)
        d = (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) #Cohen’s d / 效应量

        rows.append({
            'measure': measure,
            'n_imag': len(a), 'n_love': len(b),
            'mean_imag': a.mean(), 'mean_love': b.mean(),
            'sd_imag': a.std(), 'sd_love': b.std(),
            't': t, 'p': p, 'U': U, 'p_U': p_U,
            'Cohens d': d, 'auc': U / (len(a) * len(b)),
            'sig_t': p < bonferroni_alpha, 'sig_U': p_U < bonferroni_alpha,
        })

    tests_df = pd.DataFrame(rows)
    print(tests_df.round(3).to_string(index=False))
    return tests_df
# record tests:
#t,p
#t_m,p_m
#bonferroni_alpha t_m p_m
#
# # extra test : Cohen’s d d = (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
# Probability of superiority U, p = mannwhitneyu(a, b)
# auc = U / (len(a) * len(b))
# 记录的检验：t 检验、Mann-Whitney U、Bonferroni 校正；额外：Cohen's d 和优势概率（AUC）
def take_record(df_all,settings,tests_df,fig_path,run_id):
    """Append one JSON line describing this run to results/exp3_runs.jsonl.
    把描述本次运行的一行 JSON 追加到 results/exp3_runs.jsonl。

    The record holds: run_id, settings, group sizes, the 10 most and 10 least
    imageable lines (by imag_norm, with all their columns), the test results,
    the figure path and the intended scores file path.
    记录内容：run_id、设置、两组行数、imag_norm 最高和最低的各 10 行（含所有列）、
    检验结果、图片路径，以及计划中的分数文件路径。

    Notes / 注意:
        - the results/ folder must already exist (it is not created here).
          results/ 文件夹必须已存在（这里不会自动创建）。
        - 'scores_file' is only a path; df_all is not written there by this function.
          'scores_file' 只是一个路径，本函数并没有把 df_all 写到那里。
        - the file is opened in append mode, so re-running adds a new line.
          文件以追加模式打开，重复运行会新增一行。

    Returns / 返回: None
    """
    # lines that have an MRC imageability score / 有 MRC 意象性分数的诗行
    scored = df_all.dropna(subset=['imag_norm'])
    df_imag = df_all[df_all['group'] == 'imagist']        # re-split: these now have the metrics / 重新拆分（现在带有指标）
    df_love = df_all[df_all['group'] == 'modern_love']


    record = {
        'run_id': run_id,
        'settings': settings,                                   # your grid dict / 本次运行的参数组合
        'n_imag': len(df_imag), 'n_love': len(df_love),
        'top10':    scored.nlargest(10, 'imag_norm').to_dict('records'),   # most imageable / 意象性最高的 10 行
        'bottom10': scored.nsmallest(10, 'imag_norm').to_dict('records'),  # least imageable / 意象性最低的 10 行
        'tests': tests_df.to_dict('records'),                                # list of {'measure','t','p','U','p_U'}
        'figure': fig_path,
        'scores_file': f'results/scores_{run_id}.csv',     # save df_all there, path only in json / 只记录路径
    }

    # default=float converts numpy numbers / bools that json can't serialise
    # default=float 用于转换 json 无法直接序列化的 numpy 数值
    with open('results/exp3_runs.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, default=float) + '\n')

    #print(df_all.groupby('group')[['norm', 'imag_norm']].count())
    #print(df_all['group'].value_counts())

def exp_start(imag_corpus='orig', love_corpus='orig', base_line='computed', e_formula='orig'):
    """Run the whole Experiment III pipeline for one combination of settings.
    用一组设置运行完整的实验三流程。

    Args / 参数:
        imag_corpus: 'orig' or the suffix of data/some_imagists_<x>.json: 'orig','final'
                     'orig' 或 data/some_imagists_<x>.json 的后缀
        love_corpus: 'orig' (merve/poetry) or the suffix of data/modern_love_<x>.json: 'orig','new','final'
                     'orig'（merve/poetry）或 data/modern_love_<x>.json 的后缀
        base_line:   'computed' or 'three'
                     'computed' 或 'three'
        e_formula:   'orig' = upstream softmax entropy, 'paper' = |e|/sum|e| as in the paper
                     'orig' = 原始代码的 softmax 熵，'paper' = 论文中的 |e|/Σ|e|

    Returns / 返回: whatever take_record returns (None) / take_record 的返回值（None）
    """
    # Settings dict and a run_id that encodes them (used for cache / figure file names)
    # 设置字典，以及编码这些设置的 run_id（用于缓存和图片文件名）
    settings = dict(imag_corpus=imag_corpus, love_corpus=love_corpus,
                    base_line=base_line, e_formula=e_formula)
    run_id = f"imag_{imag_corpus}_love_{love_corpus}_base_{base_line}_entropy_{e_formula}"

    # Step 1: data / 第 1 步：读取数据
    df_imag, df_love = load_corpora(imag_corpus, love_corpus)
    # Step 2: metrics / 第 2 步：计算指标
    df_all = get_metrics(run_id,e_formula,df_imag,df_love,imag_corpus,love_corpus)

    df_imag = df_all[df_all['group'] == 'imagist']        # re-split: these now have the metrics / 重新拆分（现在带有指标）
    df_love = df_all[df_all['group'] == 'modern_love']
    # The six measures of the paper's Figure 3 / Table 6 / 论文图 3 和表 6 中的六个指标
    measures = ['norm', 'entropy', 'sparsity',
                'imag_norm', 'visual_norm', 'concrete_norm']
    # Step 3: figure / 第 3 步：画图
    fig_path = plot_distributions(df_all, run_id, base_line,measures,settings)
    # Step 4: statistical tests / 第 4 步：统计检验
    tests_df = test(df_imag,df_love,measures)
    # Step 5: save a record of this run / 第 5 步：保存本次运行记录
    return take_record(df_all,settings,tests_df,fig_path,run_id)


if __name__ == '__main__':
    exp_start()
