import os
import json
from pathlib import Path
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer,
    M2M100ForConditionalGeneration, M2M100Tokenizer,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq, set_seed
)
from config import BT_MODELS, RAW_DATASETS, TRAIN_ARGS, SEEDS
from utils.data_utils import preprocess_data
from utils.decode_utils import translate_text as generic_translate_text 
from utils.decode_utils import pivot_translate_text as m2m_translate_text
from utils.metrics import terminology_coverage, count_oov
from comet import download_model, load_from_checkpoint
from sacrebleu.metrics import BLEU, CHRF
from evaluate import load as load_metric
import numpy as np
import torch
from tqdm import tqdm 
from typing import List, Callable, Optional, Dict, Any

torch.cuda.empty_cache()

print("Loading COMET model...")
comet_model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_model_path).to("cuda" if torch.cuda.is_available() else "cpu")
comet_model.eval()

term_list = [
    "пациентов", "лечение", "терапии", "исследования", "заболевания",
    "гемодинамики", "сетчатки", "эндометрия", "беременности", "плаценты",
    "клеток", "воспалительных", "диагностики", "стимуляции", "новообразований",
    "паттерн", "психоза", "кататонии", "тромбодинамики", "аневризмы"
]

def translate_with_prompted_llm(
    texts: List[str],
    llm_model_id: str,
    source_lang_name: str, # "Russian"
    target_lang_name: str, # "French"
    batch_size: int = 4,  
    max_length_input: int = 512, # Max length for input including prompt
    max_length_output: int = 128  # Max length for generated output
) -> List[str]:
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading LLM for translation: {llm_model_id} on {device} ({source_lang_name} -> {target_lang_name})")
    model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
    model.eval()

    all_translations = []
    print(f"Translating {len(texts)} sentences in batches of {batch_size}...")
    for i in tqdm(range(0, len(texts), batch_size), desc=f"LLM Translating {source_lang_name} to {target_lang_name}"):
        batch_texts = texts[i:i+batch_size]
        
        prompts = [f"Translate {source_lang_name} to {target_lang_name}: {text}" for text in batch_texts]
        
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding="longest", 
            truncation=True, 
            max_length=max_length_input 
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_length_output, 
                early_stopping=True,
                num_beams=1 
            )
        
        batch_translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_translations.extend(batch_translations)

    del model, tokenizer
    torch.cuda.empty_cache()
    print(f"Finished LLM translation. Generated {len(all_translations)} translations.")
    return all_translations


def compute_metrics_callback(eval_preds, tokenizer):
    metric_bleu = load_metric("sacrebleu")
    preds, labels = eval_preds
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_labels_sacre = [[label] for label in decoded_labels]
    result = metric_bleu.compute(predictions=decoded_preds, references=decoded_labels_sacre)
    return {"eval_bleu": result["score"]}

def evaluate_on_test_sets(model, tokenizer, vocab, eval_decoding_strategy="greedy"):
    test_sets = {
        "wmt": RAW_DATASETS["test_wmt_en_ru"],
        "tico": RAW_DATASETS["test_tico_en_ru"],
        "flores": RAW_DATASETS["test_flores_en_ru"]
    }
    bleu_metric = BLEU()
    chrf_metric = CHRF(word_order=2)
    results = {}
    model.eval()
    for name, ds_path in test_sets.items():
        print(f"\n>>> Evaluating on {name} ({ds_path}) with {eval_decoding_strategy} decoding")
        test_data = load_dataset(ds_path)["test"]
        src, ref = test_data["en"], test_data["ru"]
        
        preds = generic_translate_text(src, model, tokenizer, decoding=eval_decoding_strategy)

        results[f"bleu_{name}"] = bleu_metric.corpus_score(preds, [ref]).score
        results[f"chrf_{name}"] = chrf_metric.corpus_score(preds, [ref]).score
        
        comet_data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(src, preds, ref)]
        comet_scores = comet_model.predict(comet_data, batch_size=16)
        results[f"comet_{name}"] = sum(comet_scores.scores) / len(comet_data) if comet_data else 0.0

        if name in ["wmt", "tico"]:
            prec, rec = terminology_coverage(preds, ref, term_list)
            results[f"term_precision_{name}"] = prec
            results[f"term_recall_{name}"] = rec
        if name == "wmt":
            results[f"oov_count_{name}"] = count_oov(preds, vocab)
    return results


def generate_or_load_pivot_bt_data(
    ru_sentences_input: List[str],
    pivot_config: Dict[str, Any],
    cache_dir_base: str = "cached_pivot_data",
    original_m2m100_cache_path: Optional[str] = "cached_pivot_bt.json"
) -> (List[str], List[str], List[str]):
    
    config_name = pivot_config["name"]
    current_cache_dir = Path(cache_dir_base) / config_name
    current_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file_name = "pivot_bt_greedy.json" 
    cache_path = current_cache_dir / cache_file_name

    if config_name == "m2m100_default_cache" and original_m2m100_cache_path and Path(original_m2m100_cache_path).exists():
        print(f"Loading from existing original M2M100 cache: {original_m2m100_cache_path}")
        with open(original_m2m100_cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("en", data.get("en_sentences", [])), \
                   data.get("fr", data.get("fr_sentences", [])), \
                   data.get("ru", data.get("ru_sentences", [])) # Ensure keys exist

    if cache_path.exists():
        print(f"Loading cached pivot BT data from {cache_path} for config {config_name}")
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["en_synthetic"], data["fr_intermediate"], data["ru_original"]

    print(f"Generating RU -> FR -> EN translations using pivot config: {config_name} (greedy decoding)...")
    
    fr_sentences = []
    en_sentences = []
    pivot_step_decoding_strategy = "greedy" 

    if pivot_config["type"] == "m2m100":
        model_id = pivot_config["model_id"]
        print(f"Using M2M100 pivot model: {model_id}")
        pivot_model_m2m = M2M100ForConditionalGeneration.from_pretrained(model_id).to("cuda" if torch.cuda.is_available() else "cpu")
        pivot_tokenizer_m2m = M2M100Tokenizer.from_pretrained(model_id)
        pivot_model_m2m.eval()

        fr_sentences = m2m_translate_text(ru_sentences_input, pivot_model_m2m, pivot_tokenizer_m2m, "ru", "fr", decoding=pivot_step_decoding_strategy)
        en_sentences = m2m_translate_text(fr_sentences, pivot_model_m2m, pivot_tokenizer_m2m, "fr", "en", decoding=pivot_step_decoding_strategy)
        
        del pivot_model_m2m, pivot_tokenizer_m2m
        torch.cuda.empty_cache()

    elif pivot_config["type"] == "bilingual_chain":
        # RU -> FR step
        model_ru_fr_id = pivot_config["model_ru_fr_id"]
        print(f"Using RU->FR model: {model_ru_fr_id}")
        model_ru_fr = AutoModelForSeq2SeqLM.from_pretrained(model_ru_fr_id).to("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer_ru_fr = AutoTokenizer.from_pretrained(model_ru_fr_id)
        model_ru_fr.eval()
        # Using generic_translate_text for bilingual models
        fr_sentences = generic_translate_text(ru_sentences_input, model_ru_fr, tokenizer_ru_fr, decoding=pivot_step_decoding_strategy)
        del model_ru_fr, tokenizer_ru_fr
        torch.cuda.empty_cache()

        # FR -> EN step
        model_fr_en_id = pivot_config["model_fr_en_id"]
        print(f"Using FR->EN model: {model_fr_en_id}")
        model_fr_en = AutoModelForSeq2SeqLM.from_pretrained(model_fr_en_id).to("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer_fr_en = AutoTokenizer.from_pretrained(model_fr_en_id)
        model_fr_en.eval()
        en_sentences = generic_translate_text(fr_sentences, model_fr_en, tokenizer_fr_en, decoding=pivot_step_decoding_strategy)
        del model_fr_en, tokenizer_fr_en
        torch.cuda.empty_cache()

    elif pivot_config["type"] == "llm_opensource":
        llm_model_id = pivot_config["llm_model_id"]
        print(f"Using Open-Source LLM for pivot: {llm_model_id}")

        llm_batch_size = pivot_config.get("llm_batch_size", 16)
        fr_sentences = translate_with_prompted_llm(ru_sentences_input, llm_model_id, "Russian", "French", batch_size=llm_batch_size)
        en_sentences = translate_with_prompted_llm(fr_sentences, llm_model_id, "French", "English", batch_size=llm_batch_size)
    else:
        raise ValueError(f"Unknown pivot_config type: {pivot_config['type']}")

    print(f"Finished generating synthetic data. Saving to {cache_path}")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({
            "ru_original": ru_sentences_input, 
            "fr_intermediate": fr_sentences, 
            "en_synthetic": en_sentences
            }, f, ensure_ascii=False, indent=2)
    
    return en_sentences, fr_sentences, ru_sentences_input


def run_single_pivot_experiment(
    seed: int,
    experiment_pivot_config: Dict[str, Any],
    initial_ru_sentences: List[str],
    combine_with_ft: bool = False,
    num_train_epochs: int = 10, 
    learning_rate: float = 2e-5 
):

    experiment_mode_name = f"{experiment_pivot_config['name']}{'_ft' if combine_with_ft else ''}"
    print(f"\n===== Running Pivot Experiment: SEED {seed}, MODE {experiment_mode_name} =====")
    set_seed(seed)

    synthetic_en, _, corresponding_ru_for_synthetic_en = generate_or_load_pivot_bt_data(
        ru_sentences_input=initial_ru_sentences,
        pivot_config=experiment_pivot_config
    )
    
    if not synthetic_en or not corresponding_ru_for_synthetic_en: # Check both
        print(f"No synthetic data (EN: {len(synthetic_en if synthetic_en else [])}, RU: {len(corresponding_ru_for_synthetic_en if corresponding_ru_for_synthetic_en else [])}). Skipping training.")
        return {"mode": experiment_mode_name, "seed": seed, "error": "No synthetic data"}

    target_tokenizer = AutoTokenizer.from_pretrained(BT_MODELS["opus_mt"]["en_ru"])
    synthetic_dataset_dict = {"train": Dataset.from_dict({"en": synthetic_en, "ru": corresponding_ru_for_synthetic_en})}
    tokenized_synthetic_train = preprocess_data(synthetic_dataset_dict, target_tokenizer, "en", "ru", "train")
    
    final_train_dataset = tokenized_synthetic_train
    if combine_with_ft:
        print("Combining with real parallel data for fine-tuning...")
        real_parallel_data = load_dataset(RAW_DATASETS["parallel_en_ru"])
        tokenized_real_train = preprocess_data(real_parallel_data, target_tokenizer, "en", "ru", "train")
        final_train_dataset = concatenate_datasets([tokenized_synthetic_train, tokenized_real_train])
        print(f"Total combined training samples: {len(final_train_dataset)}")

    dev_data = load_dataset(RAW_DATASETS["test_tico_en_ru"])
    tokenized_dev = preprocess_data(dev_data, target_tokenizer, "en", "ru", "dev")

    model_output_dir = f"./checkpoints/{experiment_mode_name}_seed{seed}"
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Get epochs and lr from TRAIN_ARGS if not passed directly or use defaults
    actual_num_train_epochs = TRAIN_ARGS.get("num_train_epochs", num_train_epochs)
    actual_learning_rate = TRAIN_ARGS.get("learning_rate", learning_rate)


    args = Seq2SeqTrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=actual_num_train_epochs, 
        learning_rate=actual_learning_rate,
        per_device_train_batch_size=TRAIN_ARGS.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=TRAIN_ARGS.get("per_device_eval_batch_size", 16),
        predict_with_generate=TRAIN_ARGS.get("predict_with_generate", True),
        generation_max_length=TRAIN_ARGS.get("generation_max_length", 128),
        generation_num_beams=TRAIN_ARGS.get("generation_num_beams", 4),
        logging_strategy=TRAIN_ARGS.get("logging_strategy", "epoch"),
        eval_strategy=TRAIN_ARGS.get("eval_strategy", "epoch"), # Corrected key
        save_strategy=TRAIN_ARGS.get("save_strategy", "epoch"),
        save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="eval_bleu", greater_is_better=True,
        seed=seed, report_to=[]
    )

    model_to_train = AutoModelForSeq2SeqLM.from_pretrained(BT_MODELS["opus_mt"]["en_ru"])
    trainer = Seq2SeqTrainer(
        model=model_to_train, args=args,
        train_dataset=final_train_dataset, eval_dataset=tokenized_dev,
        tokenizer=target_tokenizer,
        data_collator=DataCollatorForSeq2Seq(target_tokenizer, model=model_to_train),
        compute_metrics=lambda p: compute_metrics_callback(p, target_tokenizer)
    )

    print(f"Starting training for: {experiment_mode_name}, seed {seed}")
    trainer.train()
    
    print("Evaluating the best model...")
    best_model = trainer.model
    vocab = set(target_tokenizer.get_vocab().keys())
    eval_results = evaluate_on_test_sets(best_model, target_tokenizer, vocab, eval_decoding_strategy="greedy")
    eval_results.update({"mode": experiment_mode_name, "seed": seed})
    
    print("\n--- Evaluation Summary ---")
    for k, v_val in eval_results.items():
        print(f"{k:25s}: {v_val:.4f}" if isinstance(v_val, float) else f"{k:25s}: {v_val}")
    
    del model_to_train, trainer, best_model
    torch.cuda.empty_cache()
    return eval_results

def run_all_pivot_model_ablations():
    mono_med = load_dataset(RAW_DATASETS["mono_ru_medline"])["train"]
    mono_sci = load_dataset(RAW_DATASETS["mono_ru_scipar"])["train"]
    initial_ru_sentences = [ex["ru"] for ex in mono_med] + [ex["ru"] for ex in mono_sci]

    all_experiment_results = []

    pivot_model_configurations = [
        {
            "name": "opus_chain_ru_fr_en",
            "type": "bilingual_chain",
            "model_ru_fr_id": "Helsinki-NLP/opus-mt-ru-fr", 
            "model_fr_en_id": "Helsinki-NLP/opus-mt-fr-en"  
        },
        {
            "name": "mt0_base_pivot", 
            "type": "llm_opensource",
            "llm_model_id": "bigscience/mt0-base",
            "llm_batch_size": 16
        },
        {
            "name": "m2m100_pivot",
            "type": "m2m100",
            "model_id": BT_MODELS["m2m100"]
        }    
    ]

    ft_combinations = [False, True] 
    seeds_to_run = [17, 23, 42]

    for seed_val in seeds_to_run:
        for pivot_config in pivot_model_configurations:
            for combine_ft in ft_combinations:
                results = run_single_pivot_experiment(
                    seed=seed_val,
                    experiment_pivot_config=pivot_config,
                    initial_ru_sentences=initial_ru_sentences, 
                    combine_with_ft=combine_ft
                )
                all_experiment_results.append(results)

                experiment_mode_name = f"{pivot_config['name']}{'_ft' if combine_ft else ''}"
                output_results_dir = Path(f"results/pivot_model_ablations/{experiment_mode_name}")
                output_results_dir.mkdir(parents=True, exist_ok=True)
                file_name = f"results_seed{seed_val}.json"
                with open(output_results_dir / file_name, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n\n===== All Pivot Model Ablation Runs Completed =====")
    final_summary_file = Path("results") / "all_pivot_model_ablation_summary_final.json"
    with open(final_summary_file, "w", encoding="utf-8") as f:
        json.dump(all_experiment_results, f, indent=2, ensure_ascii=False)
    print(f"Full summary saved to {final_summary_file}")


if __name__ == "__main__":
    run_all_pivot_model_ablations()