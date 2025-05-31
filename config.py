# config.py

# === Model configs ===
BT_MODELS = {
    "opus_mt": {
        "ru_en": "Helsinki-NLP/opus-mt-ru-en",
        "en_ru": "Helsinki-NLP/opus-mt-en-ru"
    },
    "wmt19": {
        "ru_en": "facebook/wmt19-ru-en",
        "en_ru": "facebook/wmt19-en-ru"
    },
    "m2m100": "facebook/m2m100_418M"
}

# === Paths ===
RAW_DATASETS = {
    "mono_ru_medline": "sethjsa/medline_ru_mono",
    "mono_ru_scipar": "sethjsa/scipar_ru_mono",
    "parallel_en_ru": "sethjsa/medline_en_ru_parallel",
    "parallel_fr_ru": "sethjsa/parapat_fr_ru_parallel",
    "parallel_en_fr": "sethjsa/medline_en_fr_parallel",
    "test_wmt_en_ru": "sethjsa/wmt20bio_en_ru_sent",
    "test_tico_en_ru": "sethjsa/tico_en_ru",
    "test_flores_en_ru": "sethjsa/flores_en_ru",
    "test_tico_en_fr": "sethjsa/tico_en_fr",
    "test_flores_en_fr": "sethjsa/flores_en_fr",
    "test_wmt_en_fr": "sethjsa/wmt20bio_en_fr_sent"
}

# === Training Hyperparameters ===
TRAIN_ARGS = {
    "num_train_epochs": 10,
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 16,
    "predict_with_generate": True,
    "generation_max_length": 128,
    "generation_num_beams": 4,
    "logging_strategy": "epoch",
    "eval_strategy": "epoch",
    "save_strategy": "epoch"
}

# === Seeds ===
SEEDS = [17, 23, 42]
