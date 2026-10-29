<!--
How to cite in this report. Every citation is an ordinary Markdown link to a file in this repository:
  code or text lines   [label](../experiment_3.py#L172)   or a range: #L171-L173
  paper passage        [label](paper/Encoding_Imagism.pdf#page=9 "exact words to highlight on that page")
  data rows            [label](../data/measures/imag_orig_love_orig_base_computed_entropy_orig_metrics.csv#L2-L6)   (line 1 is the header row)
  figure               [label](../figs/imag_orig_love_orig_base_computed_entropy_orig.png)
Folders: everything except the report lives in the repository root (../): the paper's code, plus
experiment_3_replication_commented.py and run_grid.py, which write to ../data/, ../figs/ and ../results/.
Links to anything else (https://...) open normally.
-->

# Experiment III: Imagist vs Modern Love poems

**Reproduction report** · Niu Deng Meng · draft

## 1. Getting the original code running with the original data

### 1.1 Replication result

*To fill in:* whether the original run reproduces the paper's Table 6 and Figure 3, and where it does not ([Table 6, p. 18](paper/Encoding_Imagism.pdf#page=18 "T-test and Mann-Whitney U results (for comparison) for dictionary-based features")).

### 1.2 Revised, modular version of the original code

*To fill in:* how [experiment_3.py](../experiment_3.py) was split into parts in [the replication](../experiment_3_replication_commented.py), and what each part does.

## 2. Problems and areas for improvement

### 2.1 Data

*To fill in:* problems in the two corpora (authors, line splitting, duplicates, line counts).

### 2.2 Methods

#### 2.2.1 Entropy

*To fill in:* the code's entropy ([softmax entropy](../experiment_3.py#L30-L35)) versus the paper's definition ([p. 12](paper/Encoding_Imagism.pdf#page=12 "taking the absolute value of each dimension and normalizing to sum to 1")).

#### 2.2.2 Selecting the baseline lines

*To fill in:* the hard-coded example lines ([example lines](../experiment_3.py#L303-L305)) and the `base_line` options in the replication.

#### 2.2.3 Dictionary coverage (link to Experiment I)

*To fill in:* how many words the dictionaries actually match, and how this connects to Experiment I.

#### 2.2.4 Testing effect size

*To fill in:* what the t-tests and U-tests show and leave out ([tests](../experiment_3.py#L339-L350)).

### 2.3 Code

#### 2.3.1 Saving CLIP embeddings

*To fill in:* how embeddings are cached and why that matters for re-runs.

#### 2.3.2 Path control

*To fill in:* how input and output paths are handled, and what was changed.

## 3. Revised experiment

### 3.1 Runs

*To fill in:* the runs and their settings ([run_grid.py](../run_grid.py), [run log](../results/exp3_runs.jsonl)).

### 3.2 Differences between runs

*To fill in:* how the differences were calculated, and what they are.

### 3.3 Charts and tables

*To fill in:* the charts and tables that present the runs.

## 4. Critical analysis of the original visualisations

*To fill in.*

## 5. Alternative visualisations

*To fill in.*

## 6. What is in the repository

*To fill in:* where the new code, the resulting data and the figures live.

## 7. Section for the joint publication

*To fill in:* the whole process, written up for the group's paper on the original study.

## Sources

* Paper: Bizzoni, Feldkamp & Nielbo (2026), “Encoding Imagism?”, *Journal of Computational Literary Studies* 5(1) ([PDF](paper/Encoding_Imagism.pdf#page=1)).
* Original code and data: [centre-for-humanities-computing/imageability_jcls](https://github.com/centre-for-humanities-computing/imageability_jcls) at commit `531808f`, in the repository root.
