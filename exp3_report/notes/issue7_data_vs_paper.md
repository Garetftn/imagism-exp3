# Experiment III: why the data didn't match the paper (issue 7)

Scope: Experiment III only (Imagist vs Modern Love lines, Figure 3, Table 6, Table 4 poetry rows).

## Summary

| Mismatch | Cause | Status |
|---|---|---|
| 1,200 Imagist / 1,146 Love lines vs paper 1,195 / 1,126 | The paper's run dropped repeated lines (5 Imagist, 20 Love). The committed `experiment_3.py` has no such step. | **Resolved.** With duplicates dropped, Table 6's norm, entropy and sparsity rows reproduce *exactly* (t and U to the decimal). |
| 48 Imagist poems vs paper's 37 | Counting convention, plus one parsing error in `some_imagists.json`. | Explained; does not affect line-level results. |
| Dictionary rows of Table 6 off by up to 0.24 in t | Dictionary scores depend on the spaCy lemmatizer version. No version tried (3.5, 3.7, 3.8) regenerates the cached scores exactly. | Explained; can't be pinned down from the files. |
| Figure 3 examples described as top/bottom ten | Both are Love lines; "Homespun…" is rank 27, "Of insidious intent" is in the bottom ten (tied). | Minor wording issue. |

## 1. Line counts: duplicate lines

Identical lines give identical CLIP embeddings, so duplicates can be counted from the cached embeddings without the texts:

| | cached rows | unique | paper |
|---|---|---|---|
| Imagist | 1,200 | 1,195 | 1,195 |
| Love | 1,146 | 1,126 | 1,126 |

The Imagist duplicates are refrains: "Zeus,", "In the dusk", "I go on.", "There is a quiet here", "But still he dances,". The Love texts can't be listed from here (HuggingFace is blocked in this sandbox).

Table 6 from the cached files (`reproduce_table6.py`):

| measure | t, all lines | t, duplicates dropped | paper t | U, duplicates dropped | paper U |
|---|---|---|---|---|---|
| norm | −6.77 | **−6.69** | −6.69 | **567,900** | 567,900 |
| entropy | 7.27 | **7.19** | 7.19 | **783,363** | 783,363 |
| sparsity | 6.51 | **6.44** | 6.44 | **773,000** | 773,000 |
| imag_norm | 4.83 | 5.31 | 5.07 | 737,473 | 734,321.5 |
| visual_norm | 4.81 | 5.02 | 4.98 | 749,707.5 | 749,496 |
| concrete_norm | 7.31 | 7.94 | 7.93 | 795,692.5 | 796,327 |

Table 4 poetry rows with duplicates dropped: Imagist 1871.0 ± 1026.8 (paper 1870.6 ± 1027.3), normalized 376.6 ± 64.1 (paper 376.2 ± 64.5); Love 2111.5 ± 847.4 (paper 2112.53 ± 846.7), normalized 363.6 ± 53.0 (paper 363.7 ± 53.1).

Note: the entropy row matches because both the cache and the paper used **softmax** entropy, not the formula in the paper's §5 (issue 3 in `02_errors_and_issues.md`).

## 2. Poem count: 48 entries vs 37 poems

`data/some_imagists.json` stores parts of poems as separate entries. Merging them gives exactly 37:

* Fletcher's *London Excursion* is stored as 9 entries (LONDON EXCURSION + Approach, Arrival, Walk, Bus-Stop, Transposition, Peripeteia, Mid-Flight, Station) → 1 poem (−8).
* Aldington's *Epigrams* is stored as A Girl, New Love, October → 1 poem (−2).
* **Parsing error:** Aldington's *Daisy* is split in two. The entry titled "DAISY" holds only its Latin epigraph (*'Plus quam se atque suos amavit omnes, / Nunc...'*), and the poem's body sits under a fake title "CATULLUS." (the epigraph's attribution) → 1 poem (−1).

48 − 8 − 2 − 1 = 37. The parsing error also puts two **Latin** lines into the Imagist corpus. They have no dictionary hits (NaN, so they drop out of the dictionary tests) but they do enter the norm/entropy/sparsity tests.

This matters for any poem-level analysis (issue 10): grouping must use the 37 real poems, not the 48 JSON entries.

## 3. Dictionary scores depend on the spaCy version

Re-running `get_dict_scores` on the Imagist lines and comparing with the cached `imag` column:

| spaCy / model | Imagist lines whose MRC score differs from the cache |
|---|---|
| 3.5.4 / 3.5.0 (the model `requirements.txt` pins) | 28 |
| 3.7.5 / 3.7.1 | 26 |
| 3.8.16 / 3.8.0 | 18 |

The biggest single cause is the word **"leaves"**: spaCy 3.5 lemmatizes it as *leave* (not in MRC, so the word scores nothing), and 3.8 as *leaf* (MRC 608, highly imageable). Several Imagist lines are about leaves. So the dictionary side of Table 6 can't be regenerated exactly, and the small differences in the last three rows above are within this drift. The conclusions don't change: every test stays significant.

## 4. Figure 3 example lines

Both reference lines come from the **Love** corpus, though one is meant to illustrate "highly imageable":

* "Homespun, dyed butternuts dark gold color." has an imag_norm of 549.75, rank **27** of 2,293 lines. That puts it in the top 1.2%, but not in the top ten as the paper states. All 26 lines above it get their score from only 1–3 MRC words; "Homespun" is the highest-ranked line with ≥ 4. The authors likely skipped very short lines, but the paper doesn't say so (inferred). The code doesn't compute any top/bottom-ten list. The lines were picked by hand from a printed sort (`experiment_3.py` L291–300), which shows only the lowest-scoring end.
* "Of insidious intent" has an imag_norm of 217, rank 2,284 (in the bottom ten, tied). Its score comes entirely from the word **"of"**: "insidious" and "intent" aren't in MRC. This shows a wider problem: **47% of the MRC-matched tokens in the Imagist lines are function words** (a, the, of, …), which MRC also rates.

## Patched script

`experiment_3_fixed.py` is `experiment_3.py` with one change: it drops repeated lines from both corpora after loading. It works with the existing caches. Run it from inside `imageability_jcls-main/`:

```
python ../imageability/exp3/experiment_3_fixed.py
```

Tested here up to the HuggingFace download: it prints 1,195 Imagist lines. The Love step needs HuggingFace, so run it on your machine. It should report 1,126 Love lines. If it reports more, some duplicates differ only in capitalisation, which CLIP ignores but exact text matching doesn't; lower-casing before the duplicate check would fix that.
