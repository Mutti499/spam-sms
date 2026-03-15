"""
Re-label personal SMS data using the trained BERT model.
Only touches messages with source="personal". Internet data stays unchanged.
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Config ──
MODEL_DIR = "models/bert_spam_filter"
DATA_PATH = "dataset/full_with_features.csv"
OUTPUT_PATH = "dataset/full_with_features_relabeled.csv"

LABEL2ID = {"ham": 0, "spam": 1}
ID2LABEL = {0: "ham", 1: "spam"}
CONFIDENCE_THRESHOLD = 0.85  # Only re-label when BERT is very confident

# ── Device ──
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {DEVICE}")

# ── Load model ──
print(f"Loading BERT model from {MODEL_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

# ── Load data ──
df = pd.read_csv(DATA_PATH)
print(f"Total messages: {len(df)}")
print(f"Personal: {(df['source'] == 'personal').sum()}")
print(f"Internet: {(df['source'] == 'turkish_sms_collection').sum()}")

# ── Run BERT on personal data only ──
personal_mask = df["source"] == "personal"
personal_df = df[personal_mask].copy()

print(f"\nRunning BERT on {len(personal_df)} personal messages...")

batch_size = 64
all_preds = []
all_probs = []

for i in range(0, len(personal_df), batch_size):
    batch_texts = personal_df["text"].iloc[i : i + batch_size].tolist()
    inputs = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    pred_ids = probs.argmax(dim=-1)

    all_preds.extend(pred_ids.cpu().tolist())
    all_probs.extend(probs.cpu().tolist())

    if (i // batch_size) % 10 == 0:
        print(f"  {i}/{len(personal_df)}")

personal_df["bert_pred"] = [ID2LABEL[p] for p in all_preds]
personal_df["bert_spam_prob"] = [p[1] for p in all_probs]
personal_df["bert_ham_prob"] = [p[0] for p in all_probs]

# ── Find disagreements ──
disagree_mask = personal_df["label"] != personal_df["bert_pred"]
disagree = personal_df[disagree_mask]

print(f"\n{'='*60}")
print(f"DISAGREEMENTS: {len(disagree)} out of {len(personal_df)} personal messages")
print(f"{'='*60}")

# Only re-label when BERT is confident
confident_mask = (
    (personal_df["bert_spam_prob"] > CONFIDENCE_THRESHOLD)
    | (personal_df["bert_ham_prob"] > CONFIDENCE_THRESHOLD)
)
relabel_mask = disagree_mask & confident_mask
relabel_df = personal_df[relabel_mask]

print(f"Confident disagreements (>{CONFIDENCE_THRESHOLD:.0%}): {len(relabel_df)}")

# Show what will change
was_ham_now_spam = relabel_df[
    (relabel_df["label"] == "ham") & (relabel_df["bert_pred"] == "spam")
]
was_spam_now_ham = relabel_df[
    (relabel_df["label"] == "spam") & (relabel_df["bert_pred"] == "ham")
]

print(f"\n  ham -> spam: {len(was_ham_now_spam)}")
for _, row in was_ham_now_spam.head(10).iterrows():
    print(f"    [{row['bert_spam_prob']:.2%}] {row['text'][:80]}")

print(f"\n  spam -> ham: {len(was_spam_now_ham)}")
for _, row in was_spam_now_ham.head(10).iterrows():
    print(f"    [{row['bert_ham_prob']:.2%}] {row['text'][:80]}")

# ── Apply re-labeling ──
df["original_label"] = df["label"]
relabel_indices = personal_df[relabel_mask].index
df.loc[relabel_indices, "label"] = personal_df.loc[relabel_indices, "bert_pred"]

changed = (df["label"] != df["original_label"]).sum()
print(f"\nTotal labels changed: {changed}")

# ── Save ──
df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved to {OUTPUT_PATH}")
print("\nNext step: rebuild train/val/test splits from the relabeled data and retrain SVM.")
