import random
from typing import List, Dict, Tuple
import numpy as np
import nltk
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

# Import config and pivot BT loader
from config import RAW_DATASETS
from run_pivot_bt import generate_or_load_pivot_bt_data

# NLTK setup
print("Downloading/checking NLTK resource: punkt …")
nltk.download("punkt", quiet=True)

# Dataset aliases
PARALLEL_EN_RU = RAW_DATASETS["parallel_en_ru"]
MONO_RU_MEDLINE = RAW_DATASETS["mono_ru_medline"]
MONO_RU_SCIPAR = RAW_DATASETS["mono_ru_scipar"]
TEST_WMT_EN_RU = RAW_DATASETS["test_wmt_en_ru"]
TEST_TICO_EN_RU = RAW_DATASETS["test_tico_en_ru"]
TEST_FLORES_EN_RU = RAW_DATASETS["test_flores_en_ru"]

# Analysis parameters
NUM_SAMPLES_STATS = 5000
NUM_EXAMPLES = 10
EMBED_SAMPLE_SIZE = 500
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Helper functions
def _safe_load(ds_path: str, split_candidates: List[str]):
    for s in split_candidates:
        try:
            return load_dataset(ds_path, split=s, trust_remote_code=True)
        except Exception:
            continue
    ds_dict = load_dataset(ds_path, trust_remote_code=True)
    return ds_dict[list(ds_dict.keys())[0]]

def _sample(ds, column: str, k: int) -> List[str]:
    if len(ds) <= k:
        return [row[column] for row in ds]
    idx = random.sample(range(len(ds)), k)
    return [ds[i][column] for i in idx]

def _basic_stats(texts: List[str]) -> Dict[str, float]:
    toks = [word_tokenize(t.lower()) for t in texts]
    lens = [len(t) for t in toks]
    flat = [tok for sent in toks for tok in sent]
    return {
        "sents": len(texts),
        "avg_len": float(np.mean(lens)) if lens else 0.0,
        "vocab": len(set(flat)),
        "tokens": len(flat),
    }

def _jaccard(a: List[str], b: List[str]) -> float:
    va = {tok for t in a for tok in word_tokenize(t.lower())}
    vb = {tok for t in b for tok in word_tokenize(t.lower())}
    return len(va & vb) / len(va | vb) if va and vb else 0.0

def _top_ngrams(texts: List[str], n: int, k: int = 10) -> List[Tuple[Tuple[str, ...], int]]:
    cnt = Counter()
    for t in texts:
        cnt.update(ngrams(word_tokenize(t.lower()), n))
    return cnt.most_common(k)

# Load datasets
print("\n📥  Loading corpora …")
ft_ds = _safe_load(PARALLEL_EN_RU, ["train", "validation", "test"])
ft_en = _sample(ft_ds, "en", NUM_SAMPLES_STATS)
ft_ru = _sample(ft_ds, "ru", NUM_SAMPLES_STATS)
mono_med_ru = _sample(_safe_load(MONO_RU_MEDLINE, ["train"]), "ru", NUM_SAMPLES_STATS)
mono_scp_ru = _sample(_safe_load(MONO_RU_SCIPAR, ["train"]), "ru", NUM_SAMPLES_STATS)
mono_ru_all = mono_med_ru + mono_scp_ru
wmt_en = _sample(_safe_load(TEST_WMT_EN_RU, ["test"]), "en", NUM_SAMPLES_STATS)
wmt_ru = _sample(_safe_load(TEST_WMT_EN_RU, ["test"]), "ru", NUM_SAMPLES_STATS)
tico_en = _sample(_safe_load(TEST_TICO_EN_RU, ["test"]), "en", NUM_SAMPLES_STATS)
tico_ru = _sample(_safe_load(TEST_TICO_EN_RU, ["test"]), "ru", NUM_SAMPLES_STATS)
flores_en = _sample(_safe_load(TEST_FLORES_EN_RU, ["dev", "test"]), "en", NUM_SAMPLES_STATS)
flores_ru = _sample(_safe_load(TEST_FLORES_EN_RU, ["dev", "test"]), "ru", NUM_SAMPLES_STATS)

# Load pivot BT data
en_pivot_bt, fr_pivot_bt, ru_pivot_bt = generate_or_load_pivot_bt_data("cached_pivot_bt.json")
pivot_bt_en = en_pivot_bt[:NUM_SAMPLES_STATS]
pivot_bt_ru = ru_pivot_bt[:NUM_SAMPLES_STATS]

# Load standard BT data
with open("cached_standard_bt.json", encoding="utf-8") as f:
    bt_data = json.load(f)
bt_en = bt_data["synthetic_en"][:NUM_SAMPLES_STATS]
bt_ru = mono_ru_all[:NUM_SAMPLES_STATS]  # assumes matching order


print("✅  Loading complete.\n")

# Sentence statistics
print("=== Sentence / vocabulary statistics (ENGLISH) ===")
datasets_en = {
    "FT parallel EN": ft_en,
    "WMT EN": wmt_en,
    "TICO EN": tico_en,
    "FLORES EN": flores_en,
    "Pivot BT EN": pivot_bt_en,
    "BT EN": bt_en
}
for name, data in datasets_en.items():
    print(f"{name:<18} | {_basic_stats(data)}")

print("\n=== Sentence / vocabulary statistics (RUSSIAN) ===")
datasets_ru = {
    "FT parallel RU": ft_ru,
    "WMT RU": wmt_ru,
    "TICO RU": tico_ru,
    "FLORES RU": flores_ru,
    "Mono RU (BT src)": mono_ru_all,
    "Pivot BT RU": pivot_bt_ru,
    "BT RU": bt_ru
}
for name, data in datasets_ru.items():
    print(f"{name:<18} | {_basic_stats(data)}")

# Jaccard overlaps
print("\n=== Vocabulary overlap (Jaccard) vs. fine‑tune parallel ===")
for name, data in datasets_en.items():
    if name != "FT parallel EN":
        print(f"EN {name:<14}: {_jaccard(ft_en, data):.4f}")
for name, data in datasets_ru.items():
    if name != "FT parallel RU":
        print(f"RU {name:<14}: {_jaccard(ft_ru, data):.4f}")

print("\nMono‑RU (BT source) vs. each test set (RU side):")
for tgt_name, tgt_data in {"WMT RU": wmt_ru, "TICO RU": tico_ru, "FLORES RU": flores_ru}.items():
    print(f"  {tgt_name:<8}: {_jaccard(mono_ru_all, tgt_data):.4f}")

# Print some examples
print("\n=== Example hallucinations from Pivot BT ===")
for ru, en in zip(pivot_bt_ru[:10], pivot_bt_en[:10]):
    print(f"RU: {ru}\nEN: {en}\n{'-'*30}")

print("\n=== Example hallucinations from Standard BT ===")
for ru, en in zip(bt_ru[:10], bt_en[:10]):
    print(f"RU: {ru}\nEN: {en}\n{'-'*30}")

# Embedding distance analysis
print("\n=== Embedding Distance Analysis (Sentence-BERT, mean pooled) ===")
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
corpora_for_embed = {k: v[:EMBED_SAMPLE_SIZE] for k, v in datasets_en.items()}
emb_means = {}
for name, sents in corpora_for_embed.items():
    print(f"Encoding {name} ...")
    embs = model.encode(sents, batch_size=32, show_progress_bar=True)
    emb_means[name] = np.mean(embs, axis=0, keepdims=True)

# Cosine distances and heatmap
corpus_names = list(emb_means.keys())
dist_matrix = np.zeros((len(corpus_names), len(corpus_names)))
print("\nPairwise cosine distances between mean sentence embeddings:")
for i, name_i in enumerate(corpus_names):
    for j, name_j in enumerate(corpus_names):
        if j <= i:
            continue
        dist = cosine_distances(emb_means[name_i], emb_means[name_j])[0][0]
        dist_matrix[i][j] = dist_matrix[j][i] = dist
        print(f"{name_i:18s} vs {name_j:18s}: {dist:.4f}")

# Save heatmap
os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(10, 8))
sns.heatmap(
    dist_matrix,
    xticklabels=corpus_names,
    yticklabels=corpus_names,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    square=True,
    cbar_kws={"label": "Cosine Distance"}
)
plt.title("Pairwise Sentence-BERT Distances Between EN Corpora")
plt.tight_layout()
plt.savefig("plots/embedding_distance_heatmap.png", dpi=300)
plt.close()
