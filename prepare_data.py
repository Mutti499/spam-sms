"""
Data preparation pipeline for Turkish SMS spam detection.
Cleans, deduplicates, normalizes, and splits the dataset.
Outputs train/val/test CSV files ready for both SVM and BERT pipelines.
"""

import json
import re
import os
import hashlib
import pandas as pd
import numpy as np
from collections import Counter

INPUT_FILE = "dataset/combined.json"
OUTPUT_DIR = "dataset"

# ════════════════════════════════════════════════════════════
# 1. Load data
# ════════════════════════════════════════════════════════════

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)
print(f"Loaded: {len(df)} messages")
print(f"  Labels: {dict(df['label'].value_counts())}")
print(f"  Sources: {dict(df['source'].value_counts())}")


# ════════════════════════════════════════════════════════════
# 2. Text cleaning
# ════════════════════════════════════════════════════════════

def clean_text(text):
    # Replace anonymization tags with generic tokens
    text = re.sub(r'\[PERSON_\d+\]', '<PERSON>', text)
    text = re.sub(r'\[PHONE(?:_MASKED)?\]', '<PHONE>', text)
    text = re.sub(r'\[EMAIL_\d+\]', '<EMAIL>', text)
    text = re.sub(r'\[EMAIL_MASKED\]', '<EMAIL>', text)
    text = re.sub(r'\[CARD_LAST4\]', '<CARD>', text)
    text = re.sub(r'\[IMEI(?:_\d+)?\]', '<IMEI>', text)
    text = re.sub(r'\[IP_ADDRESS\]', '<IP>', text)
    text = re.sub(r'\[PRESCRIPTION\]', '<PRESCRIPTION>', text)
    text = re.sub(r'\[REDACTED_\w+\]', '<REDACTED>', text)

    # Normalize whitespace (keep newlines as spaces)
    text = re.sub(r'\s+', ' ', text).strip()

    # Normalize repeated punctuation (!!!!! → !!!, ????? → ???)
    text = re.sub(r'([!?.]){3,}', r'\1\1\1', text)

    # Normalize repeated characters (aaaaaa → aaa)
    text = re.sub(r'(.)\1{4,}', r'\1\1\1', text)

    return text


df['text_clean'] = df['text'].apply(clean_text)

# ════════════════════════════════════════════════════════════
# 3. Deduplication
# ════════════════════════════════════════════════════════════

# Exact dedup on cleaned text
before = len(df)
df['text_hash'] = df['text_clean'].apply(lambda x: hashlib.md5(x.lower().encode()).hexdigest())
df = df.drop_duplicates(subset='text_hash', keep='first')
df = df.drop(columns='text_hash')
print(f"\nDeduplication: {before} → {len(df)} (removed {before - len(df)} exact duplicates)")

# Near-duplicate removal (same text with minor variations)
# Normalize for comparison: lowercase, remove URLs, remove numbers
def normalize_for_dedup(text):
    t = text.lower()
    t = re.sub(r'https?://\S+', '', t)
    t = re.sub(r'\d+', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

df['text_norm'] = df['text_clean'].apply(normalize_for_dedup)
before = len(df)
df = df.drop_duplicates(subset='text_norm', keep='first')
df = df.drop(columns='text_norm')
print(f"Near-dedup: {before} → {len(df)} (removed {before - len(df)} near-duplicates)")

# ════════════════════════════════════════════════════════════
# 4. Feature extraction (saved alongside for analysis)
# ════════════════════════════════════════════════════════════

def extract_meta_features(text):
    """Extract structural features from SMS text."""
    features = {}
    features['length'] = len(text)
    features['word_count'] = len(text.split())
    features['url_count'] = len(re.findall(r'https?://\S+|www\.\S+', text))
    features['has_shortened_url'] = int(bool(re.search(r'bit\.ly|shorturl|tinyurl|t\.co|goo\.gl|cutt\.ly|tnn\.li|dijital\.li|turkcell\.li', text, re.IGNORECASE)))
    features['digit_ratio'] = sum(c.isdigit() for c in text) / max(len(text), 1)
    features['uppercase_ratio'] = sum(c.isupper() for c in text) / max(len(text), 1)
    features['has_currency'] = int(bool(re.search(r'TL|₺|EUR|USD', text)))
    features['has_phone_number'] = int(bool(re.search(r'(?:\d[\d\s-]{8,}\d)', text)))
    features['exclamation_count'] = text.count('!')
    features['has_opt_out'] = int(bool(re.search(r'(?i)(RET yaz|IPTAL yaz|SMS almak istemiyorsan|tanitim iptali)', text)))
    features['has_call_to_action'] = int(bool(re.search(r'(?i)(hemen.*tıkla|hemen.*kaydol|hemen.*başvur|hemen.*indir)', text)))
    features['has_discount_pattern'] = int(bool(re.search(r'%\d+', text)))

    # Spam keyword count
    spam_keywords = ['kampanya', 'indirim', 'fırsat', 'firsat', 'bonus', 'kazan',
                     'hediye', 'ücretsiz', 'ucretsiz', 'tıkla', 'tikla', 'kupon',
                     'kod', 'bedava', 'kazandınız', 'kazandiniz']
    features['spam_keyword_count'] = sum(1 for kw in spam_keywords if kw in text.lower())

    # SMS sender code pattern (B001, B016, etc.)
    features['has_sender_code'] = int(bool(re.search(r'\bB\d{3}\b', text)))

    return features


meta_features = df['text_clean'].apply(extract_meta_features).apply(pd.Series)
print(f"\nFeature correlations with spam:")
for col in meta_features.columns:
    corr = meta_features[col].corr(df['label'].map({'spam': 1, 'ham': 0}))
    bar = '█' * int(abs(corr) * 30)
    sign = '+' if corr > 0 else '-'
    print(f"  {col:25s} {sign}{abs(corr):.3f} {bar}")

# ════════════════════════════════════════════════════════════
# 5. Stratified split: 80/10/10
# ════════════════════════════════════════════════════════════

from sklearn.model_selection import train_test_split

# First split: 80% train, 20% temp
train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=42,
    stratify=df['label']
)

# Second split: 50/50 of temp → 10% val, 10% test
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42,
    stratify=temp_df['label']
)

print(f"\nSplit sizes:")
for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    ham = (split_df['label'] == 'ham').sum()
    spam = (split_df['label'] == 'spam').sum()
    print(f"  {name:5s}: {len(split_df):5d} ({ham} ham, {spam} spam, {spam/len(split_df)*100:.1f}% spam)")

# ════════════════════════════════════════════════════════════
# 6. Save
# ════════════════════════════════════════════════════════════

for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    out = split_df[['text_clean', 'label']].rename(columns={'text_clean': 'text'})
    out.to_csv(os.path.join(OUTPUT_DIR, f"{name}.csv"), index=False)

# Also save with all metadata for analysis
df_full = pd.concat([df, meta_features], axis=1)
df_full[['text_clean', 'label', 'source'] + list(meta_features.columns)].rename(
    columns={'text_clean': 'text'}
).to_csv(os.path.join(OUTPUT_DIR, "full_with_features.csv"), index=False)

print(f"\nSaved to {OUTPUT_DIR}/:")
print(f"  train.csv, val.csv, test.csv (text + label)")
print(f"  full_with_features.csv (all features for analysis)")
print(f"\nDone! Ready for training.")
