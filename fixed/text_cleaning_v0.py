import re, json
import pandas as pd

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

df_imag = load_poetry('data/some_imagists_copy.json', 'imagist', line_sep='*')
df_love = load_poetry('data/modern_love_final.json', 'modern_love', line_sep='\n')
df_all  = pd.concat([df_imag, df_love], ignore_index=True)


###

df_all.to_csv('data/cleaned_lines_preview.csv', index=False, encoding='utf-8-sig')