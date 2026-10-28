# Report viewer

`index.html` (in `exp3_report/`; the root `index.html` redirects here) shows `REPORT.md` as a web page. Every link in the report that points to a file in this repository becomes a citation:

* hover over it to see a snapshot of the cited lines, rows, page passage or figure;
* click it to open the whole source in a panel beside the report, scrolled to the cited part;
* Ctrl-click (Cmd-click on a Mac) opens the file itself in a new tab, and Esc closes the panel.

## Citation formats

| Source | Write it as |
|---|---|
| code or text lines | `[label](../experiment_3.py#L172)` (the paper's code, one folder up) or `#L171-L173` |
| paper passage | `[label](paper/Encoding_Imagism.pdf#page=9 "exact words to highlight")` |
| data rows | `[label](data/derived/imagist_lines.csv#L2-L6)` (line 1 is the header row) |
| figure | `[label](figs/figure3_reproduced.png)` |

The words in quotes after a paper link are matched against the text of that page, ignoring punctuation and line-break hyphens. If they aren't found, the preview says so.

## Preview on your own machine

The page loads files with `fetch`, which browsers block for pages opened straight from disk. Start a small web server in the repository folder instead:

    python -m http.server

and open http://localhost:8000/exp3_report/.

## Settings

The top of `viewer.js` holds the repository name used for "Open on GitHub" links, the PDFs that have page previews, and the number of lines a snapshot shows. `vendor/` holds copies of [marked](https://github.com/markedjs/marked) (MIT) and [highlight.js](https://highlightjs.org/) (BSD-3-Clause), so the page needs no outside services.
