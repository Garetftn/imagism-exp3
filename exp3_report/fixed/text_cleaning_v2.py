import re, json
import pandas as pd

def clean_line(line):
    """把一行诗里的任意连续空白归一为单个空格，并去掉首尾空白"""
    return re.sub(r'\s+', ' ', line).strip()

def is_valid(line):
    """判断一行诗句应不应该被保留"""
    if not re.search(r'[A-Za-z]', line):    # 空串、纯数字、纯标点
        return False
    return True


def load_poetry(path, group, line_sep, analysis_basis="lines"):    # group: 即诗歌的分组 imagist, modern_love
    """读取并清洗诗歌 JSON。analysis_basis 默认 "lines"（一行一条记录），整首诗模式调用时传 "poems"。"""

    with open(path, encoding='utf-8') as f:
        data = json.load(f)      #把文件解析成字典：{诗人: [{title, poem}, ...]}。

    rows = []
    poem_id = 0
    for poet, works in data.items():
        for w in works:     #w 是列表中的一个元素，也就是一首诗的字典，如 {"title": "THE POOL", "poem": "..."}
            lines = [clean_line(line) for line in w['poem'].split(line_sep)]
            lines = [line for line in lines if is_valid(line)]
            if not lines:
                continue
            if analysis_basis == "lines":
                for i, line in enumerate(lines):
                    rows.append(dict(group=group, poet=poet, title=w['title'],
                                     poem_id=poem_id, line_no=i, text=line))
            poem_id += 1
    return pd.DataFrame(rows)

# relative paths
IMAG_PATH = 'data/some_imagists_final.json'
LOVE_PATH = 'data/modern_love_final.json'


df_imag = load_poetry(IMAG_PATH, 'imagist', line_sep='*', analysis_basis="lines")
df_love = load_poetry(LOVE_PATH, 'modern_love', line_sep='\n', analysis_basis="lines")
df_all  = pd.concat([df_imag, df_love], ignore_index=True)

#df_all.to_csv(f'data/cleaned_lines_preview.csv', index=False, encoding='utf-8-sig')

