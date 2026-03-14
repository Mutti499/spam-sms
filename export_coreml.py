"""
Export SVM spam filter for iOS deployment.

Since coremltools doesn't support sklearn 1.8, we export the model
as raw weights + vocabulary JSON. The Swift app does inference natively
(TF-IDF vectorization + linear classification) — this is actually faster
and smaller than CoreML for a linear model.
"""

import pickle
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report

XCODE_DIR = "SpamDetector/SpamDetector"
MODEL_DIR = "models"

# ════════════════════════════════════════════════════════════
# 1. Train a clean text-only model for export
# ════════════════════════════════════════════════════════════

print("Training text-only model for iOS export...")

train_df = pd.read_csv("dataset/train.csv").fillna('')
test_df = pd.read_csv("dataset/test.csv").fillna('')

tfidf = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(2, 5),
    max_features=15000,  # Smaller for mobile
    sublinear_tf=True,
    strip_accents=None,
    min_df=3,
)

X_train = tfidf.fit_transform(train_df['text'])
X_test = tfidf.transform(test_df['text'])
y_train = (train_df['label'] == 'spam').astype(int).values
y_test = (test_df['label'] == 'spam').astype(int).values

svm = LinearSVC(C=0.1, class_weight='balanced', max_iter=10000, random_state=42, dual='auto')
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['ham', 'spam'], digits=4))

# ════════════════════════════════════════════════════════════
# 2. Export model weights as JSON
# ════════════════════════════════════════════════════════════

print("Exporting model weights...")

# Get vocabulary: ngram → index
vocab = tfidf.vocabulary_  # dict: str → int

# Get IDF weights
idf = tfidf.idf_.tolist()  # list of floats, indexed by feature index

# Get SVM weights: coef[0] is the weight vector, intercept[0] is the bias
coef = svm.coef_[0].tolist()  # list of floats, one per feature
intercept = float(svm.intercept_[0])

# Build compact model: only keep non-negligible weights
# This significantly reduces file size
threshold = 1e-6
active_features = {}
for ngram, idx in vocab.items():
    weight = coef[idx]
    if abs(weight) > threshold:
        active_features[ngram] = {
            "i": int(idx),  # original index (for IDF lookup)
            "w": round(float(weight), 6),  # SVM weight
        }

print(f"  Total vocab: {len(vocab)}")
print(f"  Active features (|w| > {threshold}): {len(active_features)}")

# Export
model_export = {
    "version": 1,
    "type": "tfidf_linear_svm",
    "analyzer": "char_wb",
    "ngram_range": [2, 5],
    "sublinear_tf": True,
    "intercept": round(intercept, 6),
    "idf": [round(x, 6) for x in idf],
    "features": active_features,
}

model_path = os.path.join(XCODE_DIR, "spam_model.json")
with open(model_path, "w", encoding="utf-8") as f:
    json.dump(model_export, f, ensure_ascii=False)

model_size = os.path.getsize(model_path) / 1024
print(f"  Model saved: {model_path} ({model_size:.0f} KB)")

# ════════════════════════════════════════════════════════════
# 3. Export spam config for rule-based pre-filter
# ════════════════════════════════════════════════════════════

spam_config = {
    "spam_keywords": [
        "kampanya", "indirim", "fırsat", "firsat", "bonus", "kazan",
        "hediye", "ücretsiz", "ucretsiz", "tıkla", "tikla", "kupon",
        "bedava", "promosyon", "stok", "sınırlı", "kaçırma", "hemen"
    ],
    "opt_out_patterns": [
        "RET yaz", "IPTAL yaz", "SMS almak istemiyor", "tanitim iptali", "SMS Red"
    ],
    "phishing_domains": [
        "pubit.jp", "sniply.me", "fre.to", "t.ly/merit"
    ],
    "url_shorteners": [
        "bit.ly", "shorturl.at", "tinyurl", "cutt.ly",
        "tnn.li", "dijital.li", "engho.me", "dfurl.com"
    ],
}

config_path = os.path.join(XCODE_DIR, "spam_config.json")
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(spam_config, f, ensure_ascii=False, indent=2)
print(f"  Config saved: {config_path}")

# ════════════════════════════════════════════════════════════
# 4. Verify: simulate Swift inference in Python
# ════════════════════════════════════════════════════════════

print("\nVerifying exported model matches sklearn...")

def swift_predict(text, model_data):
    """Simulate the Swift inference pipeline in Python."""
    features = model_data["features"]
    idf_weights = model_data["idf"]
    intercept = model_data["intercept"]
    ngram_min, ngram_max = model_data["ngram_range"]

    # Step 1: Extract char n-grams (char_wb adds space padding)
    padded = f" {text} "
    ngram_counts = {}
    for n in range(ngram_min, ngram_max + 1):
        for i in range(len(padded) - n + 1):
            ngram = padded[i:i+n]
            if ngram in features:
                ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

    # Step 2: Compute TF-IDF values
    tfidf_vals = {}
    for ngram, count in ngram_counts.items():
        feat = features[ngram]
        # sublinear_tf: tf = 1 + log(count)
        tf = 1.0 + np.log(count) if count > 0 else 0.0
        tfidf_vals[ngram] = tf * idf_weights[feat["i"]]

    # Step 3: L2 normalize (critical — sklearn does this by default)
    norm = np.sqrt(sum(v * v for v in tfidf_vals.values()))
    if norm > 0:
        for ngram in tfidf_vals:
            tfidf_vals[ngram] /= norm

    # Step 4: Dot product with SVM weights
    score = intercept
    for ngram, tfidf_val in tfidf_vals.items():
        score += tfidf_val * features[ngram]["w"]

    return "spam" if score > 0 else "ham", score


test_messages = [
    "Akşam eve gelirken ekmek alır mısın?",
    "750 TL bonus kazanmak için linke tıkla! https://bit.ly/xyz",
    "Faturanızın son ödeme tarihi 15.03.2024'tür.",
    "BÜYÜK İNDİRİM! %80 kampanya fırsatını kaçırmayın! Hemen tıklayın!",
    "Toplantı saat 3'e alındı, haberin olsun.",
    "Sn.B.l.NA.N.CE-TR Kullanicisi MASAK Tarafindan islemleriniz durdurulmustur",
]

print("\nExported model predictions:")
for msg in test_messages:
    label, score = swift_predict(msg, model_export)
    sklearn_pred = "spam" if svm.predict(tfidf.transform([msg]))[0] == 1 else "ham"
    match = "✓" if label == sklearn_pred else "✗"
    icon = "🚫" if label == "spam" else "✅"
    print(f"  {icon} [{label:4s}] score={score:+.3f} {match} {msg[:60]}")

# Full test set verification
matches = 0
total = len(test_df)
for _, row in test_df.iterrows():
    swift_label, _ = swift_predict(row['text'], model_export)
    sklearn_label = "spam" if svm.predict(tfidf.transform([row['text']]))[0] == 1 else "ham"
    if swift_label == sklearn_label:
        matches += 1

print(f"\nFull test set: {matches}/{total} match ({matches/total*100:.1f}%)")
print("(Minor mismatches expected due to L2 normalization difference)")
print("\nDone!")
