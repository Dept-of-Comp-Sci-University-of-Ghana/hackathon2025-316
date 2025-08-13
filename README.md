# DCIT316 Mid‑Sem Hackathon Starter (2025)

This repository provides a minimal starter kit for the DCIT316 mid‑semester hackathon. It includes:

- A simple TF‑IDF + Logistic Regression baseline for text classification.
- A CLI to train and evaluate models.
- A tiny public development split (`data/public_dev/`) for continuous integration checks.
- Unit tests and a GitHub Actions workflow for CI.
- A stub script (`scripts/get_data.sh`) and `data/README.md` to guide you in loading your own datasets.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .               # install the project package locally
pytest -q                     # run unit tests

# train the baseline on the tiny public dev split
python -m project.train --task sentiment --lang twi --data_dir data/public_dev --out_dir models

# evaluate the trained model and write reports/metrics.json
python -m project.eval  --task sentiment --lang twi --data_dir data/public_dev --model_dir models --out reports/metrics.json
cat reports/metrics.json
```

For real experiments, place your own CSVs with columns `text,label` under `data/<dataset_name>/train.csv` and `data/<dataset_name>/dev.csv`. See `data/README.md` for more details.