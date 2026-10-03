# Data dictionary

Each input file contains **one row per patient** (patient-level deduplication: the first encounter with both a
nucleic acid test and a complete blood count was retained; where several CBC measurements were obtained on the
same day as the nucleic acid test, their values were averaged). Column names are identical across the four
cohort files.

## Columns

| Column (as in the file) | Meaning | Unit / coding | Type |
|---|---|---|---|
| `age` | age at presentation | years | continuous |
| `sex` / `gender` | sex | 0 = female, 1 = male (see note) | binary |
| `month` | calendar month of presentation | 1–12 (year and day removed during de-identification) | ordinal |
| `RESULT` | outcome | 0 = non-influenza, 1 = influenza A, 2 = influenza B | 3-class |
| `WBC` | white blood cell count | ×10⁹/L | continuous |
| `NEU` | neutrophil count | ×10⁹/L | continuous |
| `NEU%` | neutrophil percentage | % | continuous |
| `LYM` | lymphocyte count | ×10⁹/L | continuous |
| `LYM%` | lymphocyte percentage | % | continuous |
| `MON` | monocyte count | ×10⁹/L | continuous |
| `MON%` | monocyte percentage | % | continuous |
| `EO` | eosinophil count | ×10⁹/L | continuous |
| `EO%` | eosinophil percentage | % | continuous |
| `BASO` | basophil count | ×10⁹/L | continuous |
| `BASO%` | basophil percentage | % | continuous |
| `RBC` | red blood cell count | ×10¹²/L | continuous |
| `HGB` | haemoglobin | g/L | continuous |
| `HCT` | haematocrit | % | continuous |
| `MCV` | mean corpuscular volume | fL | continuous |
| `MCH` | mean corpuscular haemoglobin | pg | continuous |
| `MCHC` | mean corpuscular haemoglobin concentration | g/L | continuous |
| `RDW-CV` | red cell distribution width (CV) | % | continuous |
| `PLT` | platelet count | ×10⁹/L | continuous |
| `MPV` | mean platelet volume | fL | continuous |
| `PCT` | plateletcrit | % | continuous |
| `PDW` | platelet distribution width | % | continuous |

*Note on sex encoding:* the released files use different conventions at different centres (`sex` 0/1 and
`gender` 0/1); the preprocessing script aligns the column name and keeps the encoding binary, and sex is
one-hot encoded before model fitting.

## Model inputs

The final model uses **25 predictors**: the 20 CBC-derived parameters listed above (all except `RESULT`), plus
`age`, `sex` and `month`. Continuous predictors are standardized with Z-scores (derivation-cohort means and
SDs); `sex` is one-hot encoded; `RESULT` is label-encoded 0/1/2.

## Missing data and plausibility screening

Missing values are completed with a K-Nearest Neighbors imputer (k = 5). Before imputation, every continuous
variable is screened against pre-specified physiological plausibility ranges (eTable 2, Panel B of the
manuscript); values outside those ranges are physiologically impossible and are therefore set to missing.
Extreme values that remain inside the ranges are retained — no ±3 SD winsorization is applied. The number of
values flagged by each rule, per feature and per centre, is reported in eTable 2.

## Reference standard

Influenza A/B status is defined by nucleic acid testing (Sansure six-respiratory-pathogen multiplex nucleic
acid detection kit, PCR-fluorescent-probe method; positive when the amplification curve shows a characteristic
S-shape with cycle threshold ≤ 40). Respiratory specimens were nasopharyngeal swabs collected by trained
nursing staff according to the manufacturers' instructions at all four centres. The non-influenza class
comprises patients with negative influenza A and B results and is therefore heterogeneous with respect to
other respiratory pathogens and non-infectious conditions.
