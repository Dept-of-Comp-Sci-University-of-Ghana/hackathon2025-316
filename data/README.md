This folder is where you should place your own datasets for experiments.

Each dataset should have the following structure:

```
data/<dataset_name>/train.csv    # Training set
data/<dataset_name>/dev.csv      # Development/validation set
```

Both `train.csv` and `dev.csv` must have **two columns**: `text` and `label`.

The provided `public_dev` split is a tiny synthetic example used by the continuous integration workflow. Do **not** use it for actual experiments.