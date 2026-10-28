# Split files

`train.txt`, `val.txt`, `test.txt` are generated **once** by `scripts/make_splits.py`
with a fixed seed and then committed. They are the only source of truth for which
sample belongs to which split, so a reported number is reproducible from
(config, commit, split file).

Format: one sample id per line, where the id is the image/mask file stem.
Blank lines and lines starting with `#` are ignored.

```
python scripts/make_splits.py --source data/raw/masks --pattern "*.png" \
    --out-dir data/splits --seed 42 --ratios 0.8 0.1 0.1
```

If the dataset has patient or case ids, add `--group-regex` so whole cases land in
one split and slices from the same patient never leak across train/val/test.

The test split is used only by `evaluate.py --split test`.
