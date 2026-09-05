# EDA: `data/private_qa.json`

**Source:** `owkin/technical_test` (private Hugging Face dataset), `train` split, as saved locally by `src/data_fetching.py`.
**Reproduce the figures:** `uv run --with matplotlib python docs/make_plots.py`

---

## Headline

The local file contains **604** (rather than 600) unique multiple-choice pairs asking whether a bladder-cancer cell line is sensitive to a drug. It is structurally clean, **exactly balanced** between `Yes` and `No`, and the label is perfectly determined by a 0.1 µM IC50 threshold.

| | |
|---|---|
| Rows / distinct questions / distinct pairs | 604 / 604 / 604 |
| Correct label | `Yes` ×302 / `No` ×302 |
| Answer key | `A` ×289 / `B` ×315 |
| Indications / templates | 1 / 1 |
| Cell lines / drugs / source studies | 17 / 298 / 4 |
| Label rule observed | `ic50_recomputed < 0.1 µM` ⇔ `Yes` |

Unlike the earlier 50-row head slice, this is a usable **binary classification** evaluation set: always answering either label scores 50%, and always choosing `A` scores 47.8%. It is still a narrow benchmark, however: every question is about `Bladder/Urinary Tract`, has the same wording, and contains no ADMET or scenario variant.

The row order is a material caveat. Records 1–302 are all `Yes`; records 303–604 are all `No`. Split or sample only after shuffling, preferably with a fixed seed, or an order-dependent evaluation will be invalid.

---

## Schema and integrity

All four top-level fields and all nine metadata fields are populated in all 604 records. Questions and `(cell_line, drug)` pairs are unique. `answer` maps to `metadata.label` after parsing `options` in all records.

| Field | Type | Notes |
|---|---|---|
| `question` | str | 14–16 words, 90–129 chars |
| `options` | str | A **JSON-encoded string**, not an object; parse it with a second `json.loads` |
| `answer` | str | `A` or `B` |
| `metadata` | dict | 9 keys, all present |

The one question template is:

> `Would the {indication} cancer cell line {cell_line} be sensitive to treatment with {drug}?`

The rendered prompt contains only indication, cell line, and drug. It does **not** expose the IC50 that determines the label.

### Metadata coverage

| Key | Distinct values | Comment |
|---|---:|---|
| `cell_line` | 17 | 14–55 questions per cell line |
| `drug` | 298 | 183 drugs appear once |
| `indication` | 1 | constant: `Bladder/Urinary Tract` |
| `label` | 2 | 302 `Yes`, 302 `No` |
| `study` | 4 | `CTRPv2_2015` supplies 515/604 rows |
| `template` | 1 | constant |
| `ic50_recomputed` | 602 | numeric µM; `1e-06` occurs three times |
| `has_admet_hint` | 1 | constant `False` |
| `test_scenario` | 1 | constant empty string |

`has_admet_hint` and `test_scenario` are dead columns in this file. The dataset tests drug/cell-line sensitivity only, not reasoning over those possible variants.

---

## The label is an IC50 threshold

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/ic50-distribution-dark.png">
  <img alt="Overlaid log-scale histograms show every Yes IC50 below the dashed 0.1 µM cut-off and every No IC50 above it, with an empty interval between 0.1 and 1 µM." src="figures/ic50-distribution.png">
</picture>

The relationship is exact in this file:

```text
ic50_recomputed < 0.1 µM  →  Yes  (302/302)
ic50_recomputed ≥ 0.1 µM  →  No   (302/302)
```

There is a clean one-order-of-magnitude gap at the boundary: the largest `Yes` IC50 is `0.0999142` µM, while the smallest `No` is `1.05901` µM. This makes the task internally unambiguous, but it also means a trace should state the threshold convention; a mechanistic narrative alone is not evidence for the answer.

| Label | min | p10 | p25 | median | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Yes` | 1.00e-06 | 0.00219 | 0.00951 | 0.02873 | 0.06058 | 0.08483 | 0.09991 |
| `No` | 1.05901 | 2.54701 | 5.56682 | 18.32389 | 67.00973 | 216.39131 | 1.73e+07 |

Within positives, 99/302 (32.8%) are in `[0.05, 0.1)` µM and 22 are at least 0.09 µM: they are sensitive by the convention, but not markedly below its boundary. Within negatives, 26/302 are below 2 µM; the rest are much less potent. That separation is conspicuous enough to make a trace that claims borderline biological uncertainty questionable unless it has information beyond this file.

---

## Coverage

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/ic50-by-cell-line-dark.png">
  <img alt="Horizontal stacked bars show question counts and the Yes/No balance for each of 17 bladder cancer cell lines." src="figures/ic50-by-cell-line.png">
</picture>

The cells have between 14 and 55 questions. Label mix varies from 30.8% `Yes` for T24 (12/39) to 72.2% for DSH1 (26/36), but every cell line has both labels. This supports comparisons across the included cell lines, subject to their unequal sample counts.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/coverage-dark.png">
  <img alt="Left: questions per source study, dominated by CTRPv2_2015 at 515. Right: a histogram of drug frequency, showing 183 drugs occur once." src="figures/coverage.png">
</picture>

- **Study skew:** `CTRPv2_2015` supplies 515/604 (85.3%) rows; `GDSC_2020_v1`, `GDSC_2020_v2`, and `CCLE_2015` supply 49, 29, and 11. Per-study comparisons are therefore weak.
- **Sparse drug coverage:** 183/298 drugs occur once, 65 occur twice, and only 17 drugs have both labels represented. A drug-specific score is usually based on one observation and should not be generalized.
- **Combination drugs:** 30 entries are fixed-ratio combinations, for example `docetaxel:tanespimycin (2:1 mol/mol)`. Drug matching must preserve lowercase names, colons, parentheses, and ratios rather than splitting the name on `:`.
- **No duplicate pairs:** all 604 pairs are unique. There is no within-pair replication for estimating measurement variability.

---

## Implications for reasoning-trace generation

1. **The label baseline is now meaningful.** The 50/50 class balance removes the previous all-`Yes` shortcut; label-only baselines score 50% and option-position-only baselines score 47.8% (`A`) or 52.2% (`B`).
2. **Shuffle before any split.** The file is ordered by label in two contiguous runs. A naïve head/tail split would create single-class partitions and make evaluation meaningless.
3. **Score biology, not retrieval of the hidden IC50.** The prompt withholds IC50, yet the label is deterministically thresholded from it. A correct answer is not, by itself, evidence that a trace has biologically justified the conclusion; require cited, external evidence or explicitly frame the response as a prediction.
4. **Keep granularity honest.** There is enough data for an overall score and perhaps cell-line-level descriptive results, but not for reliable drug-level conclusions: most drugs are singletons and only 17 appear with both outcomes.
5. **Keep the parser defensive.** Parse `options` twice and treat combination drug names as opaque identifiers.

## Suggested next step

Use a fixed-seed, stratified split by label before trace generation, retain cell-line and study counts in evaluation reporting, and reserve a test set that is not adjacent in file order to the training set. If the intended deliverable truly has 600 pairs, reconcile that expectation with the 604 records currently in `data/private_qa.json` before freezing an evaluation manifest.
