"""Make the page images and word positions the report viewer uses for paper previews.

Run from inside paper/:
    python make_words_json.py
Needs poppler-utils (pdftoppm, pdftotext) on the PATH.
"""
import html
import json
import os
import re
import subprocess

PDF = 'Encoding_Imagism.pdf'

# one JPEG per page: pages/p-01.jpg, pages/p-02.jpg, ...
os.makedirs('pages', exist_ok=True)
subprocess.run(['pdftoppm', '-r', '120', '-jpeg', '-jpegopt', 'quality=78', '-hide-annotations', PDF, 'pages/p'],
               check=True)  # annotations (e.g. a reader's highlights) are left out of the previews

# every word with its box, as fractions of the page width and height: [word, x0, y0, x1, y1]
bbox = subprocess.run(['pdftotext', '-bbox', PDF, '-'], check=True, capture_output=True, text=True).stdout
pages = []
for width, height, body in re.findall(r'<page width="([\d.]+)" height="([\d.]+)">(.*?)</page>', bbox, re.S):
    w, h = float(width), float(height)
    pages.append([[html.unescape(word), round(float(x0) / w, 4), round(float(y0) / h, 4),
                   round(float(x1) / w, 4), round(float(y1) / h, 4)]
                  for x0, y0, x1, y1, word in re.findall(
                      r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>', body)])

with open('words.json', 'w', encoding='utf-8') as f:
    json.dump(pages, f, separators=(',', ':'), ensure_ascii=False)
print(f'{len(pages)} pages, {sum(map(len, pages))} words')
