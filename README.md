# Domain Adaptation for Low-Resource Biomedical Machine Translation (EN-RU Biomedical Domain Adaptation)

This project investigates various strategies for using backtranslation (BT) and pivot-based BT to improve neural machine translation (NMT) from English to Russian in the biomedical domain.

## 🔧 Project Structure and File Overview

### `run_experiment.py`
Runs full experimental pipelines for the following modes:
- `baseline`: Zero-shot evaluation of the base model without any fine-tuning.
- `finetune`: Fine-tuning on in-domain parallel data (EN-RU).
- `bt`: Fine-tuning on synthetic data created via standard backtranslation (RU → EN).
- `combined`: Fine-tuning on both synthetic and real parallel data.

Usage:
```bash
python run_experiment.py --mode bt --cache_path bt_data.json
```

### `run_pivot_bt.py`
Runs pivot-based backtranslation experiments. Supports:
- M2M100 pivoting (RU → FR → EN)
- Bilingual chaining (RU → FR → EN with two models)
- Prompted LLM-based pivoting (e.g., MT0)

Includes options to combine pivot-BT data with fine-tuning.

### `config.py`
Stores global configuration:
- Model checkpoints (`BT_MODELS`)
- Dataset paths (`RAW_DATASETS`)
- Training arguments (`TRAIN_ARGS`)
- Random seeds (`SEEDS`)

### `analysis.py`
Performs analysis on datasets and BT quality:
- Sentence length, vocabulary size
- Jaccard overlaps (domain divergence)
- Embedding distances via Sentence-BERT
- BT example inspection for semantic drift
- Heatmap generation saved to `plots/`

### `utils/`
Modular utilities used throughout the project:

- `data_utils.py`: Preprocessing routines for Hugging Face datasets.
- `decode_utils.py`: Batched translation + pivot translation functions.
- `metrics.py`: Computes terminology precision/recall and OOV counts.
- `io_utils.py`: Helpers for result printing and saving.

### `cached_standard_bt.json`
Stores the synthetic English sentences (`synthetic_en`) generated from monolingual Russian data using a RU→EN model. Paired with `mono_ru_all` as source.

### `cached_pivot_bt.json`
Stores pivot-based synthetic data generated via RU → FR → EN translation. Keys:
- `ru_original`
- `fr_intermediate`
- `en_synthetic`

Other pivot configurations are saved under `cached_pivot_data/<config_name>/`.

### `results/`
Contains all output results as `.json` files grouped by mode and seed.

---

## 📋 Summary of Supported Experimental Modes

| Mode      | Description                                           |
|-----------|-------------------------------------------------------|
| baseline  | Evaluates base model on test sets without fine-tuning |
| finetune  | Trains on parallel in-domain EN-RU data              |
| bt        | Uses backtranslated RU→EN data for training          |
| combined  | Combines BT data with real parallel EN-RU            |
| pivot     | Uses pivot-based synthetic data (via FR)             |

---

## 📂 Output
All trained models are saved under `checkpoints/`, and evaluation results are saved under `results/`.

---

## 📈 Requirements
- Python 3.8+
- Hugging Face Transformers
- Datasets
- SentenceTransformers
- sacreBLEU, COMET, evaluate

Install via:
```bash
pip install -r requirements.txt
```

