<!--
How to cite in this report. Every citation is an ordinary Markdown link to a file in this repository:
  code or text lines   [label](source/experiment_3.py#L172)   or a range: #L171-L173
  paper passage        [label](paper/Encoding_Imagism.pdf#page=9 "exact words to highlight on that page")
  data rows            [label](data/derived/imagist_lines.csv#L2-L6)   (line 1 is the header row)
  figure               [label](figs/figure3_reproduced.png)
Folders: source/ is the paper's repository, fixed/ is your code, data/derived/ and figs/ are your outputs,
notes/ holds notes. Links to anything else (https://...) open normally.
-->

# Experiment III: Imagist vs Modern Love poems

**Reproduction report** · Neil · draft

> **Dummy content.** Every paragraph below is placeholder text for testing the layout. The citations are real: they point at the right lines, rows, pages and figures. Hover over a citation to preview its source, and click it to open the source beside the report.

## 1. What they planned to do

*Placeholder:* one or two sentences on the goal of Experiment III, citing how the paper describes it ([literary case study, p. 3](paper/Encoding_Imagism.pdf#page=3 "We examine the discriminatory power of these embedding")) and where the section starts ([§6.3, p. 16](paper/Encoding_Imagism.pdf#page=16 "6.3 Experiment III: Comparing Poems")).

*Placeholder:* the two corpora and the promise to keep Imagist poets out of the Love set ([footnote 7, p. 9](paper/Encoding_Imagism.pdf#page=9 "To maintain clear genre distinctions, we excluded poets in Some Imagist Poets from the love-themed poetry")), and the unit of analysis, poem lines ([footnote 14, p. 18](paper/Encoding_Imagism.pdf#page=18 "we conduct our analysis at the poem-line level rather than at the poem level")).

## 2. What they actually did

*Placeholder:* how the Imagist poems are split into lines ([splitting on `*`](source/experiment_3.py#L86-L96)) and what the first lines look like ([first Imagist lines](data/derived/imagist_lines.csv#L2-L6)).

*Placeholder:* how entropy is computed in the code ([softmax entropy](source/experiment_3.py#L30-L35)) compared with the definition in the paper ([entropy definition, p. 12](paper/Encoding_Imagism.pdf#page=12 "taking the absolute value of each dimension and normalizing to sum to 1")).

*Placeholder:* the statistical tests behind Table 6 ([t-test and Mann-Whitney U](source/experiment_3.py#L339-L350)).

## 3. What they didn't do

*Placeholder:* the Bonferroni correction is computed but never reported ([Bonferroni block](source/experiment_3.py#L352-L380)), and lines are treated as independent although they nest in poems ([poem-level test](fixed/exp3_validity_checks.py#L85-L96)).

## 4. What went wrong

*Placeholder:* the author filter compares names by exact match ([author filter](source/experiment_3.py#L171-L173)), so D. H. Lawrence ends up in both corpora ([Lawrence in the Love set, p. 27](paper/Encoding_Imagism.pdf#page=27 "Last Words to Miriam"); [Lawrence in the Imagist set, p. 28](paper/Encoding_Imagism.pdf#page=28 "Ballad of Another Ophelia"); [his Imagist lines](data/derived/imagist_lines.csv#L837-L839)).

*Placeholder:* duplicate lines explain the line counts ([issue 7 notes](notes/issue7_data_vs_paper.md#L14-L39)), and the fix drops them in both corpora ([Imagist](fixed/experiment_3_fixed.py#L164-L165), [Love](fixed/experiment_3_fixed.py#L291-L292)).

## 5. Are the claims valid?

| Claim | Verdict | Evidence |
|---|---|---|
| Embedding metrics distinguish the two groups ([p. 19](paper/Encoding_Imagism.pdf#page=19 "embedding-based metrics do distinguish Imagist poetry")) | *Placeholder* | [effect sizes](data/derived/table6_effect_sizes.csv#L2-L4) |
| Separability holds under lexical control ([p. 20](paper/Encoding_Imagism.pdf#page=20 "genre-level separability even under lexical control")) | *Placeholder* | [within-corpus check](fixed/exp3_validity_checks.py#L124-L135) |
| Imageability is shaped by composition, not only genre ([p. 19](paper/Encoding_Imagism.pdf#page=19 "Imageability is not merely a product of genre conventions")) | *Placeholder* | [norm vs imageability](figs/norm_vs_imageability_within_corpus.png) |

## 6. Extra experiments

*Placeholder:* which extra factors to test (author, vocabulary, genre), and the script that makes the derived tables and figures ([make_figures.py](fixed/make_figures.py#L1-L6)).

## 7. How to read Figure 3 and Table 6

*Placeholder:* what the figure shows ([Figure 3 caption, p. 18](paper/Encoding_Imagism.pdf#page=18 "Figure 3: Imagist vs. love poems in terms of dictionary features and embedding-based metrics")), how the code draws it ([plotting code](source/experiment_3.py#L313-L333)), and the redrawn version ([Figure 3 reproduced](figs/figure3_reproduced.png)).

![Figure 3 reproduced from the cached data](figs/figure3_reproduced.png)

*Placeholder:* the two reference lines are hard-coded ([example lines](source/experiment_3.py#L303-L305)), although the paper says they were among the top and bottom ten ([p. 18](paper/Encoding_Imagism.pdf#page=18 "two poem lines which were found among the top ten most imageable and bottom ten least imageable lines")). Their values are in the Love table: [“Homespun…”](data/derived/love_lines.csv#L235) and [“Of insidious intent”](data/derived/love_lines.csv#L618).

*Placeholder:* reading Table 6 ([Table 6 caption, p. 18](paper/Encoding_Imagism.pdf#page=18 "T-test and Mann-Whitney U results (for comparison) for dictionary-based features")) next to the effect sizes it leaves out ([Table 6 with effect sizes](data/derived/table6_effect_sizes.csv#L1-L7)) and the script that reproduces it ([reproduce_table6.py](fixed/reproduce_table6.py#L19-L24)).

## Sources

* Paper: Bizzoni, Feldkamp & Nielbo (2026), “Encoding Imagism?”, *Journal of Computational Literary Studies* 5(1) ([PDF](paper/Encoding_Imagism.pdf#page=1)).
* Original code and data: [centre-for-humanities-computing/imageability_jcls](https://github.com/centre-for-humanities-computing/imageability_jcls) at commit `531808f`, copied into `source/`.
