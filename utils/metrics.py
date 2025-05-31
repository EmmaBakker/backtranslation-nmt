import re
from collections import Counter
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords


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

