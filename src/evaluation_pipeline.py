import argparse
import csv
import re
import string
from collections import Counter

import langid
import nltk
import numpy as np
import spacy
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

# Make sure to download NLTK's punkt tokenizer if needed:
nltk.download('punkt', quiet=True)


def preprocess_text(text):
    """
    Lowercase the text, remove punctuation, and tokenize.
    """
    translator = str.maketrans('', '', string.punctuation)
    text_clean = text.lower().translate(translator)
    tokens = text_clean.split()  # simple whitespace tokenization
    return tokens


def language_classification(tokens):
    """
    Use langid to classify the language of each token.
    If the detected language is neither 'hi' (Hindi) nor 'en' (English),
    mark it as 'other'.
    """
    langs = []
    for token in tokens:
        lang, _ = langid.classify(token)
        if lang not in ['hi', 'en']:
            langs.append('other')
        else:
            langs.append(lang)
    return langs


def compute_cmi(token_langs):
    """
    Compute the Code-Mixing Index (CMI) using the formula:
    CMI = (1 - max(M, E, O) / T) * 100,
    where M = # Hindi words, E = # English words, O = # Other words,
    and T = total number of words.
    """
    total = len(token_langs)
    counts = Counter(token_langs)
    m = counts.get('hi', 0)
    e = counts.get('en', 0)
    o = counts.get('other', 0)
    if total > 0:
        cmi = (1 - max(m, e, o) / total) * 100
    else:
        cmi = 0
    return {'hi': m, 'en': e, 'other': o, 'cmi': cmi}


def compute_switch_points(langs):
    """
    Compute the number of language switch points in the token sequence.
    """
    if not langs:
        return 0
    switches = sum(1 for i in range(1, len(langs)) if langs[i] != langs[i - 1])
    return switches


def integration_index(m, e):
    """
    Compute a simple integration index as the ratio (in percentage)
    of the lower count to the higher count between Hindi and English.
    If one language is missing, the index is 0.
    """
    if max(m, e) > 0:
        return (min(m, e) / max(m, e)) * 100
    else:
        return 0


def pos_tagging(sentence, nlp):
    """
    Use spaCy's multilingual model to perform POS tagging.
    Returns a list of tuples: (token, POS tag).
    """
    doc = nlp(sentence)
    return [(token.text, token.pos_) for token in doc]


def main():
    parser = argparse.ArgumentParser(
        description="Code-Switched Sentence Evaluation Pipeline"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the file containing code-switched sentences (one sentence per line)",
    )
    parser.add_argument(
        "--reference_file",
        type=str,
        default=None,
        help="Optional: Path to reference sentences file (for BLEU evaluation). Each line must align with the input.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.csv",
        help="File to write per-sentence evaluation metrics",
    )
    args = parser.parse_args()

    # Load spaCy multilingual model for POS tagging.
    try:
        nlp = spacy.load("xx_ent_wiki_sm")
    except Exception as e:
        print("Error loading spaCy model. Please run:")
        print("    python -m spacy download xx_ent_wiki_sm")
        return

    # Read input sentences
    with open(args.input_file, "r", encoding="utf-8") as infile:
        sentences = [line.strip() for line in infile if line.strip()]

    results = []

    # Process each sentence individually.
    for sent in sentences:
        tokens = preprocess_text(sent)
        token_langs = language_classification(tokens)
        metrics = compute_cmi(token_langs)
        switches = compute_switch_points(token_langs)
        metrics["switch_points"] = switches
        metrics["total_tokens"] = len(tokens)
        metrics["integration_index"] = integration_index(metrics["hi"], metrics["en"])
        pos_tags = pos_tagging(sent, nlp)
        
        result = {
            "sentence": sent,
            "total_tokens": len(tokens),
            "hi": metrics["hi"],
            "en": metrics["en"],
            "other": metrics["other"],
            "CMI": metrics["cmi"],
            "switch_points": switches,
            "integration_index": metrics["integration_index"],
            "pos_tags": pos_tags,
        }
        results.append(result)

    # Optional: if a reference file is provided, compute BLEU score per sentence.
    bleu_score = None
    if args.reference_file:
        with open(args.reference_file, "r", encoding="utf-8") as ref_file:
            references = [line.strip() for line in ref_file if line.strip()]
        if len(references) != len(sentences):
            print("Error: The number of reference sentences does not match the number of input sentences.")
        else:
            all_bleu = []
            for i, sent in enumerate(sentences):
                hyp_tokens = preprocess_text(sent)
                ref_tokens = preprocess_text(references[i])
                # Compute sentence-level BLEU score.
                sent_bleu = sentence_bleu([ref_tokens], hyp_tokens)
                all_bleu.append(sent_bleu)
                results[i]["bleu"] = sent_bleu
            bleu_score = np.mean(all_bleu)

    # Compute overall statistics.
    avg_tokens = np.mean([r["total_tokens"] for r in results])
    avg_cmi = np.mean([r["CMI"] for r in results])
    avg_switch_points = np.mean([r["switch_points"] for r in results])
    avg_integration_index = np.mean([r["integration_index"] for r in results])

    print("\n===== Overall Evaluation Metrics =====")
    print(f"Total sentences: {len(sentences)}")
    print(f"Average tokens per sentence: {avg_tokens:.2f}")
    print(f"Average CMI: {avg_cmi:.2f}")
    print(f"Average Switch Points: {avg_switch_points:.2f}")
    print(f"Average Integration Index: {avg_integration_index:.2f}")
    if bleu_score is not None:
        print(f"Average BLEU Score: {bleu_score:.2f}")

    # Save detailed per-sentence metrics to CSV.
    fieldnames = [
        "sentence",
        "total_tokens",
        "hi",
        "en",
        "other",
        "CMI",
        "switch_points",
        "integration_index",
        "bleu",
        "pos_tags",
    ]
    with open(args.output, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "sentence": r["sentence"],
                    "total_tokens": r["total_tokens"],
                    "hi": r["hi"],
                    "en": r["en"],
                    "other": r["other"],
                    "CMI": r["CMI"],
                    "switch_points": r["switch_points"],
                    "integration_index": r["integration_index"],
                    "bleu": r.get("bleu", ""),
                    "pos_tags": r["pos_tags"],
                }
            )

    print(f"\nPer-sentence evaluation metrics saved to {args.output}")


if __name__ == "__main__":
    main()
