# Dataset

Place the Quora question-pair CSV file in this folder as:

```text
data/quora.csv
```

The notebook expects these original columns:

- `id`
- `qid1`
- `qid2`
- `question1`
- `question2`
- `is_duplicate`

The project keeps `question1`, `question2`, and the target label for modelling, and renames `is_duplicate` to `label`.

Dataset statistics reported by the completed project:

- Rows before cleaning: **404,290**
- Rows after cleaning: **404,287**
- Not duplicate (`0`): **255,024 (63.08%)**
- Duplicate (`1`): **149,263 (36.92%)**

The dataset itself is not included in this repository package.
