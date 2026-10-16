# PHEME five-fold training

PHEME now generates folds using the original TianBian95/BiGCN
`Process/rand5fold.py` Twitter algorithm instead of reusing the partitions in `folds.pkl`.
When present, the union of IDs in `folds.pkl` defines the prepared sample
roster only. CSV threads outside that roster are reported and excluded;
missing graphs or labels inside the roster still raise an error.
For classes 0, 1, 2, 3, shuffle IDs independently, then take consecutive
blocks of `int(class_size * 0.2)` as each fold's test set. Remaining IDs
form its training pool. As in the original code, rounding remainders
stay in every training pool and never enter a test fold in that iteration.
NPZ labels remain authoritative.

The random generator uses `--seed + --iteration`: all three branches
share the same partition, and changing iteration regenerates folds.
This reproduces the splitting algorithm, not the original repository's
unseeded random realization. UA-TAF still reserves 10% of each training
pool for independent Dev selection and fusion calibration.

Run one complete five-fold iteration:

```powershell
python UA-TAF/train_encoders.py --dataset PHEME --iteration 0
```

Run multiple iterations (each trains five folds):

```powershell
0..4 | ForEach-Object { python UA-TAF/train_encoders.py --dataset PHEME --iteration $_ }
```

To run one fold, add `--fold-id 0` (valid IDs: 0 through 4).
Default output is `results/UA_TAF_training/PHEME/iterN/foldK/`.
With an explicit `--output-dir`, automatic five-fold mode adds `foldK/`;
use a separate output directory per iteration.
Each fold writes `split_manifest.json` containing its seed and exact
Train/Dev/Test IDs. Evidence and checkpoints are stored per fold.
Run the existing fusion command separately on each fold's evidence.
Old PHEME results/checkpoints used a different partition; retrain all
branches before comparing or fusing results under this protocol.
Other datasets keep their existing split behavior.
