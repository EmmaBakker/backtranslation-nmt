import os
from collections import defaultdict
import argparse, random, json, sys, time

os.environ["WANDB_DISABLED"] = "true"

# imports
import torch

import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from evaluate import load
import numpy as np
from pathlib import Path
import sacrebleu
from sacrebleu.metrics import BLEU
from comet import download_model, load_from_checkpoint
import contextlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(0)

print(f"Using device: {device if device == torch.device('cpu') else device_name}")

SRC_LANG = "en"
TGT_LANG = "ru"
MODEL_NAME = "Helsinki-NLP/opus-mt-en-ru"
BT_MODEL_NAME = "Helsinki-NLP/opus-mt-ru-en"
DEV_DATASET_NAME = "sethjsa/tico_en_ru"
TEST_DATASET_NAME = "sethjsa/wmt20bio_en_ru_sent"
OUTPUT_DIR = "./results"

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = fnull
            sys.stderr = fnull  # Comment this line if you want to see error messages
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def preprocess_data(dataset_dict, tokenizer, src_lang, tgt_lang, split, max_length=128):
    def preprocess_function(examples):
        inputs = examples[src_lang]
        targets = examples[tgt_lang]

        model_inputs = tokenizer(
            inputs,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )

        labels = tokenizer(
            targets,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset_dict[split].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset_dict[split].column_names
    )

    return tokenized_dataset

def postprocess_predictions(predictions, labels, tokenizer):
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    return decoded_preds, decoded_labels

def compute_metrics_val(tokenizer, eval_preds):
    preds, labels = eval_preds
    decoded_preds, decoded_labels = postprocess_predictions(preds, labels, tokenizer)

    # Calculate BLEU score
    with suppress_output():
        bleu = load("sacrebleu")
        results = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])

    return {"bleu": results["score"]}

def compute_metrics_test(src, tgt, preds, bleu=True, comet=False):
    if bleu:
        bleu = load("sacrebleu")
        results = bleu.compute(predictions=preds, references=[[l] for l in tgt])
        score = results["score"]
    if comet:
        raise NotImplementedError("COMET not implemented yet")
        # Calculate COMET score

    return score

def train_model(model_name, tokenized_datasets, tokenizer, training_args):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Verify GPU usage
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected! Training will be slow.")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(training_args.output_dir, exist_ok=True)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["dev"] if "dev" in tokenized_datasets else None,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=lambda x: compute_metrics_val(tokenizer, x)
    )

    trainer.train()
    return model

def translate_text(texts, model, tokenizer, max_length=128, batch_size=32):
    model = model.to(device)
    model.eval()
    translations = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                early_stopping=True
            )

        batch_translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(batch_translations)

    return translations

def terminology_coverage(preds, refs, terms):
    tp = fp = fn = 0
    for hyp, ref in zip(preds, refs):
        for t in terms:
            in_ref = t in ref
            in_hyp = t in hyp
            tp += in_ref and in_hyp
            fp += (not in_ref) and in_hyp
            fn += in_ref and (not in_hyp)
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    return prec, rec

def count_oov(preds, vocab):
    return sum(tok not in vocab for sent in preds for tok in sent.split())

def evaluate_translation_model(model, tokenizer):
    metric_bleu = BLEU()
    vocab = set(tokenizer.get_vocab().keys())

    term_list = [
        "пациентов",  # patients
        "лечение",  # treatment
        "терапии",  # therapy
        "исследования",  # studies
        "заболевания",  # diseases
        "гемодинамики",  # hemodynamics
        "сетчатки",  # retina
        "эндометрия",  # endometrium
        "беременности",  # pregnancy
        "плаценты",  # placenta
        "клеток",  # cells
        "воспалительных",  # inflammatory
        "диагностики",  # diagnostics
        "стимуляции",  # stimulation
        "новообразований",  # neoplasms
        "паттерн",  # pattern (in diagnosis)
        "психоза",  # psychosis
        "кататонии",  # catatonia
        "тромбодинамики",  # thrombodynamics
        "аневризмы"  # aneurysms
    ]

    comet_model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_model_path)
    comet_model = comet_model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

    # === WMT Biomedical Test Set Evaluation ===
    wmt20bio = load_dataset("sethjsa/wmt20bio_en_ru_sent")
    wmt20bio_src = wmt20bio["test"]["en"]
    wmt20bio_ref = wmt20bio["test"]["ru"]
    preds = translate_text(wmt20bio_src, model, tokenizer)

    bleu = metric_bleu.corpus_score(preds, [wmt20bio_ref]).score
    chrF = sacrebleu.corpus_chrf(preds, [wmt20bio_ref]).score
    comet_data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(wmt20bio_src, preds, wmt20bio_ref)]
    comet_scores = comet_model.predict(comet_data, gpus=1 if torch.cuda.is_available() else 0).scores
    comet_mean = sum(comet_scores) / len(comet_scores)

    prec, rec = terminology_coverage(preds, wmt20bio_ref, term_list)
    oov = count_oov(preds, vocab)

    # === TICO-19 Evaluation ===
    tico = load_dataset("sethjsa/tico_en_ru")
    tico_src = tico["test"]["en"]
    tico_ref = tico["test"]["ru"]
    tico_preds = translate_text(tico_src, model, tokenizer)
    tico_bleu = metric_bleu.corpus_score(tico_preds, [tico_ref]).score
    tico_chrF = sacrebleu.corpus_chrf(tico_preds, [tico_ref]).score
    tico_comet_data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(tico_src, tico_preds, tico_ref)]
    tico_comet_scores = comet_model.predict(tico_comet_data, gpus=1 if torch.cuda.is_available() else 0).scores
    tico_comet = sum(tico_comet_scores) / len(tico_comet_scores)
    tico_prec, tico_rec = terminology_coverage(tico_preds, tico_ref, term_list)

    # === FLORES Evaluation ===
    flores = load_dataset("sethjsa/flores_en_ru")
    flores_src = flores["test"]["en"]
    flores_ref = flores["test"]["ru"]
    flores_preds = translate_text(flores_src, model, tokenizer)
    flores_bleu = metric_bleu.corpus_score(flores_preds, [flores_ref]).score
    flores_chrF = sacrebleu.corpus_chrf(flores_preds, [flores_ref]).score
    flores_comet_data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(flores_src, flores_preds, flores_ref)]
    flores_comet_scores = comet_model.predict(flores_comet_data, gpus=1 if torch.cuda.is_available() else 0).scores
    flores_comet = sum(flores_comet_scores) / len(flores_comet_scores)

    result = {
        # WMT
        "bleu": bleu,
        "chrf": chrF,
        "comet": comet_mean,
        "term_precision": prec,
        "term_recall": rec,
        "oov_count": oov,
        # TICO
        "bleu_tico": tico_bleu,
        "chrf_tico": tico_chrF,
        "comet_tico": tico_comet,
        "term_precision_tico": tico_prec,
        "term_recall_tico": tico_rec,
        # FLORES
        "bleu_flores": flores_bleu,
        "chrf_flores": flores_chrF,
        "comet_flores": flores_comet,
        # Outputs
        # "preds": preds,
    }

    print(result)

    return result

def cross_entropy_difference(general_model, target_model, tokenizer, sentence, target_tokenizer=None):
    inputs = tokenizer(sentence, return_tensors="pt")
    inputs = inputs.to(device)
    target_inputs = target_tokenizer(sentence, return_tensors="pt") if target_tokenizer else inputs.copy()
    target_inputs = target_inputs.to(device)

    # Compute cross-entropy loss for both models
    with torch.no_grad():
        general_model.to(device)
        g_outputs = general_model(**inputs, labels=inputs["input_ids"])
        g_ce = g_outputs.loss.item() * len(inputs["input_ids"][0])

        target_model.to(device)
        t_outputs = target_model(**inputs, labels=inputs["input_ids"])
        t_ce = t_outputs.loss.item() * len(inputs["input_ids"][0])

    return t_ce - g_ce

def get_filtered_dataset(synthetic_parallel, tokenizer, args):
    # check if required files exist
    if not os.path.exists("index_score_pre_BT.npy") and args.experiment in ["pre_bt", "both", "combined"]:
        raise FileNotFoundError("Pre-computed scores not found. Please run the scoring script first.")
    if not os.path.exists("index_score_post_BT.npy") and args.experiment in ["post_bt", "both", "combined"]:
        raise FileNotFoundError("Pre-computed scores not found. Please run the scoring script first.")
    if not os.path.exists("index_score_post_BT_pivot.npy") and args.use_pivot:
        raise FileNotFoundError("Pre-computed scores (pivot) not found. Please run the scoring script first.")

    selected_idx = []
    selected_idx_pivot = None

    if args.experiment == "baseline":
        selected_idx = np.random.choice(len(synthetic_parallel), int(len(synthetic_parallel) * args.filter_prop), replace=False)

    elif args.experiment == "pre_bt":
        scores = np.load("index_score_pre_BT.npy")
        scores.sort(order="score")
        if args.inverse_sort:
            scores.flip(axis=0)
        selected_idx = [x[0] for x in scores[:int(len(scores) * args.filter_prop)]]

    elif args.experiment == "post_bt":
        scores = np.load("index_score_post_BT.npy")
        scores.sort(order="score")
        if args.inverse_sort:
            scores.flip(axis=0)
        selected_idx = [x[0] for x in scores[:int(len(scores) * args.filter_prop)]]
        if args.use_pivot:
            scores = np.load("index_score_post_BT_pivot.npy")
            scores.sort(order="score")
            if args.inverse_sort:
                scores.flip(axis=0)
            selected_idx_pivot = [x[0] for x in scores[:int(len(scores) * args.filter_prop)]]

    elif args.experiment == "both":
        # Select pre-BT scores
        scores = np.load("index_score_pre_BT.npy")
        scores.sort(order="score")
        if args.inverse_sort:
            scores.flip(axis=0)
        selected_idx_pre = [x[0] for x in scores[:int(len(scores) * np.sqrt(args.filter_prop))]]
        # Refine selection with post-BT scores
        scores_filtered = np.load("index_score_post_BT.npy")[selected_idx_pre]
        scores_filtered.sort(order="score")
        if args.inverse_sort:
            scores_filtered.flip(axis=0)
        selected_idx = [x[0] for x in scores_filtered[:int(len(scores_filtered) * np.sqrt(args.filter_prop))]]
        # If using pivot, also select from post-BT pivot scores
        if args.use_pivot:
            scores_pivot = np.load("index_score_post_BT_pivot.npy")[selected_idx_pre]
            scores_pivot.sort(order="score")
            if args.inverse_sort:
                scores_pivot.flip(axis=0)
            selected_idx_pivot = [x[0] for x in scores_pivot[:int(len(scores_pivot) * np.sqrt(args.filter_prop))]]

    elif args.experiment == "combined":
        # Combine pre-BT and post-BT scores by summing them
        scores_pre = np.load("index_score_pre_BT.npy")
        scores_post = np.load("index_score_post_BT.npy")
        scores_pre = np.array(scores_pre.tolist())
        scores_post = np.array(scores_post.tolist())
        scores_combined = scores_pre[:, 1] + scores_post[:, 1]
        scores_combined = np.array(list(zip(range(len(scores_combined)), scores_combined)), dtype=[("index", int), ("score", float)])
        scores_combined.sort()
        if args.inverse_sort:
            scores_combined.flip(axis=0)
        selected_idx = [x[0] for x in scores_combined[:int(len(scores_combined) * args.filter_prop)]]

        if args.use_pivot:
            scores_pivot = np.load("index_score_post_BT_pivot.npy")
            scores_pivot = np.array(scores_pivot.tolist())
            scores_combined_pivot = scores_pre[:, 1] + scores_pivot[:, 1]
            scores_combined_pivot = np.array(list(zip(range(len(scores_combined_pivot)), scores_combined_pivot)), dtype=[("index", int), ("score", float)])
            scores_combined_pivot.sort()
            if args.inverse_sort:
                scores_combined_pivot.flip(axis=0)
            selected_idx_pivot = [x[0] for x in scores_combined_pivot[:int(len(scores_combined_pivot) * args.filter_prop)]]

    else:
        raise ValueError(f"Unknown experiment type: {args.experiment}")

    synthetic_parallel_filtered = synthetic_parallel.select(selected_idx)
    tokenized_synthetic_filtered = preprocess_data({"train": synthetic_parallel_filtered}, tokenizer, "en", "ru","train")
    if args.use_pivot:
        if selected_idx_pivot is not None:
            synthetic_parallel_filtered = synthetic_parallel.select(selected_idx_pivot)
        tokenized_pivot_filtered = preprocess_data({"train": synthetic_parallel_filtered}, tokenizer, "pivot", "ru","train")
        tokenized_synthetic_filtered = concatenate_datasets([tokenized_synthetic_filtered, tokenized_pivot_filtered])

    if args.seed is not None:
        tokenized_synthetic_filtered = tokenized_synthetic_filtered.shuffle(seed=args.seed)

    return tokenized_synthetic_filtered

def run_single(args, tokenizer=None, dev_dataset=None, test_dataset=None):
    tokenized_synthetic_filtered = get_filtered_dataset(synthetic_parallel, tokenizer, args)

    if args.add_parallel:
        print("Adding parallel data to training set")
        medline_parallel_dataset = load_dataset("sethjsa/medline_en_ru_parallel")
        tokenized_medline = preprocess_data(medline_parallel_dataset, tokenizer, "en", "ru", "train")
        tokenized_synthetic_filtered = concatenate_datasets([tokenized_synthetic_filtered, tokenized_medline])

    bt_datasets = DatasetDict({
        "train": tokenized_synthetic_filtered,
        "dev": dev_dataset,
        "test": test_dataset
    })

    filtered_training_args = Seq2SeqTrainingArguments(
                torch_compile=True,
                output_dir=f"filtered_{args.experiment}_{args.filter_prop}{'_pivot' if args.use_pivot else ''}{'_ft' if args.add_parallel else ''}",
                eval_strategy="epoch",
                learning_rate=args.lr,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                weight_decay=0.01,
                optim="adamw_torch",
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_epsilon=1e-8,
                save_total_limit=1,
                num_train_epochs=args.epochs,
                predict_with_generate=True,
                generation_num_beams=4,
                generation_max_length=128
    )

    print(f"Training on {len(bt_datasets['train'])} sentences after filtering")
    model_bt = train_model(MODEL_NAME, bt_datasets, tokenizer, filtered_training_args)
    print("Training complete. Evaluating model...")
    evaluate_translation_model(model_bt, tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BT model on filtered data")
    parser.add_argument("--experiment", type=str, default="baseline", help="\"baseline\", \"pre_bt\", or \"post_bt\"")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Initial learning rate")
    parser.add_argument("--filter_prop", type=float, default=0.50, help="Proportion of selected data")
    parser.add_argument("--use_pivot", action="store_true", help="Use pivoted data for back-translation")
    parser.add_argument("--add_parallel", action="store_true", help="Add parallel data to training set")
    parser.add_argument("--inverse_sort", action="store_true")
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    print(args)

    develop = True
    EPOCHS_TARGET_LM = 3
    BATCH_SIZE_TARGET_LM = 64
    LR_TARGET_LM = 2e-5

    if args.seed is not None:
        transformers.set_seed(args.seed)
        np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    bt_tokenizer = AutoTokenizer.from_pretrained(BT_MODEL_NAME)
    bt_model = AutoModelForSeq2SeqLM.from_pretrained(BT_MODEL_NAME)

    medline_parallel_dataset = load_dataset("sethjsa/medline_en_ru_parallel")
    tokenized_medline = preprocess_data(medline_parallel_dataset, tokenizer, SRC_LANG, TGT_LANG, "train")
    dev_dataset = load_dataset(DEV_DATASET_NAME)
    tokenized_dev_dataset = preprocess_data(dev_dataset, tokenizer, SRC_LANG, TGT_LANG, "dev")
    test_dataset = load_dataset(TEST_DATASET_NAME)
    tokenized_test_dataset = preprocess_data(test_dataset, tokenizer, SRC_LANG, TGT_LANG, "test")

    # Load synthetic parallel dataset if available, otherwise translate
    try:
        synthetic_dataset = load_dataset("linus-b/synthetic_parallel")
        synthetic_parallel = synthetic_dataset["train"]
    except:
        print("Synthetic parallel dataset not found. Generating synthetic data using back-translation...")
        if args.use_pivot:
            print("Pivot data missing, use linus-b/synthetic_parallel. Exiting...")
            exit(0)

        # Tokenize synthetic BT dataset
        tokenized_synthetic = preprocess_data(synthetic_dataset, tokenizer, SRC_LANG, TGT_LANG, "train")

        # Load monolingual ru biomedical data for BT
        mono_ru_medline = load_dataset("sethjsa/medline_ru_mono")
        mono_ru_scipar = load_dataset("sethjsa/scipar_ru_mono")
        # Explicitly convert to Python lists of strings
        mono_ru_medline_texts = [ex["ru"] for ex in mono_ru_medline["train"]]
        mono_ru_scipar_texts = [ex["ru"] for ex in mono_ru_scipar["train"]]

        if develop and data_proportion < 1.0:
            n_medline_subset = int(len(mono_ru_medline_texts) * data_proportion)
            n_scipar_subset = int(len(mono_ru_scipar_texts) * data_proportion)
            print(f"Selecting small subset {n_medline_subset} + {n_scipar_subset}")
            mono_ru_medline_texts = mono_ru_medline_texts[:n_medline_subset]
            mono_ru_scipar_texts = mono_ru_scipar_texts[:n_scipar_subset]

        # Concatenate the two lists
        mono_ru_all = mono_ru_medline_texts + mono_ru_scipar_texts
        # Translate monolingual Russian to English
        synthetic_en = translate_text(mono_ru_all, bt_model, bt_tokenizer, max_length=128, batch_size=32)
        print("Expected number of sentences to backtranslate:", len(mono_ru_all))
        synthetic_parallel = Dataset.from_dict({"en": synthetic_en, "ru": mono_ru_all})
        synthetic_dataset = DatasetDict({
            "train": synthetic_parallel
        })

    if not os.path.exists("index_score_pre_BT.npy") and args.experiment == "pre_bt":
        print("Scoring Russian Sentences")
        model_name = "ai-forever/rugpt3small_based_on_gpt2"
        rugpt_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        general_model = GPT2LMHeadModel.from_pretrained(model_name)
        target_model = GPT2LMHeadModel.from_pretrained(model_name)

        tokenized_mono_ru = preprocess_data(medline_parallel_dataset, rugpt_tokenizer, "ru", "ru", "train")

        training_args_target = Seq2SeqTrainingArguments(
            output_dir="./ru_target_model",
            eval_strategy="no",  # No evaluation during this training for simplicity
            learning_rate=LR_TARGET_LM,
            per_device_train_batch_size=BATCH_SIZE_TARGET_LM,
            per_device_eval_batch_size=BATCH_SIZE_TARGET_LM,
            weight_decay=0.01,
            optim="adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            save_total_limit=1,
            num_train_epochs=EPOCHS_TARGET_LM,  # Train for 1 epoch on this data
            predict_with_generate=False,  # No generation needed here
            report_to="none",  # Disable reporting if not using a logger like W&B
        )

        # Create a Dataset object from the tokenized data for the trainer
        target_train_dataset = Dataset.from_dict({
            'input_ids': tokenized_mono_ru['input_ids'],
            'attention_mask': tokenized_mono_ru['attention_mask'],
            'labels': tokenized_mono_ru['input_ids'].copy()  # For language modeling, labels are usually input_ids
        })

        # Define a DataCollator for Seq2Seq models
        data_collator = DataCollatorForSeq2Seq(tokenizer=rugpt_tokenizer, model=target_model)

        trainer_target = Seq2SeqTrainer(
            model=target_model,
            args=training_args_target,
            train_dataset=target_train_dataset,
            tokenizer=rugpt_tokenizer,
            data_collator=data_collator,
        )

        print("Training target model...")
        trainer_target.train()
        print("Target model training complete.")

        synthetic_ru = synthetic_parallel["ru"]

        print("Scoring Sentences...")
        scores = []
        for i, text in enumerate(synthetic_ru):
            scores.append((i, cross_entropy_difference(general_model, target_model, rugpt_tokenizer, text)))
        columns = [("index", int), ("score", float)]
        scores = np.array(scores, dtype=columns)
        print("Saving scores to index_score_pre_BT.npy")
        np.save("index_score_pre_BT.npy", np.array(scores))

    if not os.path.exists("index_score_post_BT.npy") and args.experiment == "post_bt":
        print("Scoring English sentences")
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

        general_model = GPT2LMHeadModel.from_pretrained('gpt2')
        target_model = GPT2LMHeadModel.from_pretrained('gpt2')

        tokenized_mono_en = preprocess_data(medline_parallel_dataset, gpt2_tokenizer, "en", "en", "train")

        training_args_target = Seq2SeqTrainingArguments(
            output_dir="./en_target_model",
            eval_strategy="no",
            learning_rate=LR_TARGET_LM,
            per_device_train_batch_size=BATCH_SIZE_TARGET_LM,
            per_device_eval_batch_size=BATCH_SIZE_TARGET_LM,
            optim="adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            save_total_limit=1,
            num_train_epochs=EPOCHS_TARGET_LM,
            predict_with_generate=False,  # No generation needed here
            report_to="none",  # Disable reporting if not using a logger like W&B
        )

        # Create a Dataset object from the tokenized data for the trainer
        target_train_dataset = Dataset.from_dict({
            'input_ids': tokenized_mono_en['input_ids'],
            'attention_mask': tokenized_mono_en['attention_mask'],
            'labels': tokenized_mono_en['input_ids'].copy()  # For language modeling, labels are usually input_ids
        })

        # Define a DataCollator for Seq2Seq models
        data_collator = DataCollatorForSeq2Seq(tokenizer=gpt2_tokenizer, model=target_model)

        trainer_target = Seq2SeqTrainer(
            model=target_model,
            args=training_args_target,
            train_dataset=target_train_dataset,
            tokenizer=gpt2_tokenizer,
            data_collator=data_collator,
        )

        print("Training target model...")
        trainer_target.train()
        print("Target model training complete.")

        synthetic_en = synthetic_parallel["en"]

        scores = []
        print("Scoring Sentences...")
        for i, text in enumerate(synthetic_en):
            scores.append((i, cross_entropy_difference(general_model, target_model, gpt2_tokenizer, text)))
        columns = [("index", int), ("score", float)]
        scores = np.array(scores, dtype=columns)
        print("Saving scores to index_score_post_BT.npy")
        np.save("index_score_post_BT.npy", np.array(scores))

    if not os.path.exists("index_score_post_BT_pivot.npy") and args.experiment == "post_bt" and args.use_pivot:
        print("Scoring pivoted English sentences")
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

        general_model = GPT2LMHeadModel.from_pretrained('gpt2')
        target_model = GPT2LMHeadModel.from_pretrained('gpt2')

        tokenized_mono_en = preprocess_data(medline_parallel_dataset, gpt2_tokenizer, "en", "en", "train")

        training_args_target = Seq2SeqTrainingArguments(
            output_dir="./en_target_model",
            eval_strategy="no",
            learning_rate=LR_TARGET_LM,
            per_device_train_batch_size=BATCH_SIZE_TARGET_LM,
            per_device_eval_batch_size=BATCH_SIZE_TARGET_LM,
            optim="adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            save_total_limit=1,
            num_train_epochs=EPOCHS_TARGET_LM,
            predict_with_generate=False,  # No generation needed here
            report_to="none",  # Disable reporting if not using a logger like W&B
        )

        # Create a Dataset object from the tokenized data for the trainer
        target_train_dataset = Dataset.from_dict({
            'input_ids': tokenized_mono_en['input_ids'],
            'attention_mask': tokenized_mono_en['attention_mask'],
            'labels': tokenized_mono_en['input_ids'].copy()  # For language modeling, labels are usually input_ids
        })

        # Define a DataCollator for Seq2Seq models
        data_collator = DataCollatorForSeq2Seq(tokenizer=gpt2_tokenizer, model=target_model)

        trainer_target = Seq2SeqTrainer(
            model=target_model,
            args=training_args_target,
            train_dataset=target_train_dataset,
            tokenizer=gpt2_tokenizer,
            data_collator=data_collator,
        )

        print("Training target model...")
        trainer_target.train()
        print("Target model training complete.")

        synthetic_en = synthetic_parallel["pivot"]

        print("Scoring Pivoted Sentences...")
        scores = []
        for i, text in enumerate(synthetic_en):
            scores.append((i, cross_entropy_difference(general_model, target_model, gpt2_tokenizer, text)))
        columns = [("index", int), ("score", float)]
        scores = np.array(scores, dtype=columns)
        print("Saving scores to index_score_post_BT_pivot.npy")
        np.save("index_score_post_BT_pivot.npy", np.array(scores))

    run_single(args, tokenizer=tokenizer, dev_dataset=tokenized_dev_dataset, test_dataset=tokenized_test_dataset)