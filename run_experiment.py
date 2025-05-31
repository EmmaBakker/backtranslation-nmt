import os
import json
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq, set_seed
)
from sacrebleu.metrics import BLEU, CHRF
from evaluate import load as load_metric
from comet import download_model, load_from_checkpoint
from config import BT_MODELS, RAW_DATASETS, TRAIN_ARGS, SEEDS
from utils.data_utils import preprocess_data
from utils.decode_utils import translate_text
from utils.metrics import terminology_coverage, count_oov
from utils.io_utils import print_summary


def load_comet():
    print("Loading COMET model...")
    path = download_model("Unbabel/wmt22-comet-da")
    return load_from_checkpoint(path).eval().to("cuda" if torch.cuda.is_available() else "cpu")


def generate_or_load_backtranslated_data(cache_path):
    if Path(cache_path).exists():
        print(f"Loading cached backtranslated data from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["synthetic_en"], data["original_ru"]
    else:
        print("Generating synthetic EN translations from monolingual RU data...")
        mono_med = load_dataset(RAW_DATASETS["mono_ru_medline"])['train']
        mono_sci = load_dataset(RAW_DATASETS["mono_ru_scipar"])['train']
        mono_ru_all = [ex['ru'] for ex in mono_med] + [ex['ru'] for ex in mono_sci]
        bt_model = AutoModelForSeq2SeqLM.from_pretrained(BT_MODELS["opus_mt"]["ru_en"]).to("cuda")
        bt_tokenizer = AutoTokenizer.from_pretrained(BT_MODELS["opus_mt"]["ru_en"])
        synthetic_en = translate_text(mono_ru_all, bt_model, bt_tokenizer)
        data = {"synthetic_en": synthetic_en, "original_ru": mono_ru_all}
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return synthetic_en, mono_ru_all


def evaluate(model, tokenizer, vocab, comet_model, term_list):
    test_sets = {
        "wmt": RAW_DATASETS["test_wmt_en_ru"],
        "tico": RAW_DATASETS["test_tico_en_ru"],
        "flores": RAW_DATASETS["test_flores_en_ru"]
    }
    bleu = BLEU()
    chrf = CHRF()
    results = {}

    for name, path in test_sets.items():
        data = load_dataset(path)["test"]
        src, ref = data["en"], data["ru"]
        preds = translate_text(src, model, tokenizer)
        comet_data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(src, preds, ref)]

        results[f"bleu_{name}"] = bleu.corpus_score(preds, [ref]).score
        results[f"chrf_{name}"] = chrf.corpus_score(preds, [ref]).score
        results[f"comet_{name}"] = sum(comet_model.predict(comet_data).scores) / len(comet_data)

        if name in ["wmt", "tico"]:
            prec, rec = terminology_coverage(preds, ref, term_list)
            results[f"term_precision_{name}"] = prec
            results[f"term_recall_{name}"] = rec

        if name == "wmt":
            results["oov_count"] = count_oov(preds, vocab)

    return results


def train_and_eval(model_name, train_data, dev_data, mode, seed, term_list):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab = set(tokenizer.get_vocab().keys())
    comet_model = load_comet()

    args = Seq2SeqTrainingArguments(
        output_dir=f"checkpoints/{mode}_seed{seed}",
        seed=seed,
        **TRAIN_ARGS
    )

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        metric = load_metric("sacrebleu")
        return {"eval_bleu": metric.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])["score"]}

    trainer = Seq2SeqTrainer(
        model=AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda"),
        args=args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer),
        compute_metrics=compute_metrics
    )
    trainer.train()

    return evaluate(trainer.model, tokenizer, vocab, comet_model, term_list)


def run_experiment(mode, cache_path="bt_data.json"):
    SRC, TGT = "en", "ru"
    term_list = ["пациентов", "лечение", "терапии", "исследования", "заболевания",
                 "гемодинамики", "сетчатки", "эндометрия", "беременности", "плаценты",
                 "клеток", "воспалительных", "диагностики", "стимуляции", "новообразований",
                 "паттерн", "психоза", "кататонии", "тромбодинамики", "аневризмы"]

    if mode == "baseline":
        model_name = BT_MODELS["opus_mt"]["en_ru"]
        for seed in SEEDS:
            set_seed(seed)
            print(f"\n=== Baseline Evaluation | Seed {seed} ===")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            vocab = set(tokenizer.get_vocab().keys())
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")
            comet_model = load_comet()
            results = evaluate(model, tokenizer, vocab, comet_model, term_list)
            results.update({"mode": "baseline", "seed": seed})
            print_summary(results)

    elif mode == "finetune":
        model_name = BT_MODELS["opus_mt"]["en_ru"]
        train = preprocess_data(load_dataset(RAW_DATASETS["parallel_en_ru"]),
                                AutoTokenizer.from_pretrained(model_name), SRC, TGT, "train")
        dev = preprocess_data(load_dataset("sethjsa/tico_en_ru"),
                              AutoTokenizer.from_pretrained(model_name), SRC, TGT, "dev")
        for seed in SEEDS:
            set_seed(seed)
            results = train_and_eval(model_name, train, dev, "finetune", seed, term_list)
            results.update({"mode": "finetune", "seed": seed})
            print_summary(results)

    elif mode == "bt":
        model_name = BT_MODELS["opus_mt"]["en_ru"]
        synthetic_en, mono_ru_all = generate_or_load_backtranslated_data(cache_path)
        synthetic_dataset = Dataset.from_dict({"en": synthetic_en, "ru": mono_ru_all})
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train = preprocess_data({"train": synthetic_dataset}, tokenizer, SRC, TGT, "train")
        dev = preprocess_data(load_dataset("sethjsa/tico_en_ru"), tokenizer, SRC, TGT, "dev")
        for seed in SEEDS:
            set_seed(seed)
            results = train_and_eval(model_name, train, dev, "bt", seed, term_list)
            results.update({"mode": "bt", "seed": seed})
            print_summary(results)

    elif mode == "combined":
        model_name = BT_MODELS["opus_mt"]["en_ru"]
        synthetic_en, mono_ru_all = generate_or_load_backtranslated_data(cache_path)
        synthetic_dataset = Dataset.from_dict({"en": synthetic_en, "ru": mono_ru_all})
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        synthetic = preprocess_data({"train": synthetic_dataset}, tokenizer, SRC, TGT, "train")
        real = preprocess_data(load_dataset(RAW_DATASETS["parallel_en_ru"]), tokenizer, SRC, TGT, "train")
        combined = concatenate_datasets([synthetic, real])
        dev = preprocess_data(load_dataset("sethjsa/tico_en_ru"), tokenizer, SRC, TGT, "dev")
        for seed in SEEDS:
            set_seed(seed)
            results = train_and_eval(model_name, combined, dev, "combined", seed, term_list)
            results.update({"mode": "combined", "seed": seed})
            print_summary(results)

    else:
        raise ValueError("Unknown mode. Choose from: baseline, finetune, bt, combined")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "finetune", "bt", "combined"], required=True)
    parser.add_argument("--cache_path", type=str, default="bt_data.json")
    args = parser.parse_args()
    run_experiment(args.mode, cache_path=args.cache_path)
