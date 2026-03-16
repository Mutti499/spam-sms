"""
Export SVM spam filter for iOS — FULL model with structural features.
Exports TF-IDF vocab + SVM weights + structural feature weights as JSON.
"""

import json
import os
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

XCODE_DIR = "SpamDetector/SpamDetector"

# ════════════════════════════════════════════════════════════
# 1. Train full model (TF-IDF + structural)
# ════════════════════════════════════════════════════════════

print("Training full model for iOS export...")

train_df = pd.read_csv("dataset/train.csv").fillna('')
test_df = pd.read_csv("dataset/test.csv").fillna('')

# TF-IDF
tfidf = TfidfVectorizer(
    analyzer='char_wb', ngram_range=(2, 5),
    max_features=15000, sublinear_tf=True, min_df=3,
)

X_train_tfidf = tfidf.fit_transform(train_df['text'])
X_test_tfidf = tfidf.transform(test_df['text'])

# Structural features (same as train_svm.py)
SPAM_KW = ['kampanya', 'indirim', 'fırsat', 'firsat', 'bonus', 'kazan',
           'hediye', 'ücretsiz', 'ucretsiz', 'tıkla', 'tikla', 'kupon',
           'bedava', 'promosyon', 'stok', 'sınırlı', 'kaçırma', 'hemen']

STRUCT_NAMES = [
    'length', 'word_count', 'log_length',
    'url_count', 'has_shortened_url',
    'digit_ratio', 'uppercase_ratio', 'special_char_ratio',
    'has_currency', 'has_price_pattern',
    'has_phone_number',
    'exclamation_count', 'question_count',
    'has_opt_out', 'has_call_to_action', 'has_discount_pattern', 'has_urgency',
    'spam_keyword_count', 'spam_keyword_density',
    'has_sender_code', 'has_mersis',
    'allcaps_word_ratio',
    'emoji_count',
]


def extract_structural(text):
    l = max(len(text), 1)
    return [
        len(text), len(text.split()), np.log1p(len(text)),
        len(re.findall(r'https?://\S+|www\.\S+', text)),
        int(bool(re.search(r'bit\.ly|shorturl|tinyurl|t\.co|cutt\.ly|tnn\.li|dijital\.li|dfurl|t2m\.io|engho\.me', text, re.I))),
        sum(c.isdigit() for c in text) / l,
        sum(c.isupper() for c in text) / l,
        sum(not c.isalnum() and not c.isspace() for c in text) / l,
        int(bool(re.search(r'\bTL\b|₺|EUR|USD', text))),
        int(bool(re.search(r'\d+[\.,]?\d*\s*TL', text))),
        int(bool(re.search(r'(?:\d[\d\s-]{8,}\d)', text))),
        text.count('!'), text.count('?'),
        int(bool(re.search(r'(?i)(RET yaz|IPTAL yaz|SMS almak istemiyorsan|tanitim iptali|SMS Red)', text))),
        int(bool(re.search(r'(?i)(hemen.*tıkla|hemen.*kaydol|hemen.*başvur|hemen.*indir|hemen.*ara)', text))),
        int(bool(re.search(r'%\d+', text))),
        int(bool(re.search(r'(?i)(SON \d+ GÜN|SON GÜN|SON SAATLER|SON \d+ HAFTA)', text))),
        sum(1 for kw in SPAM_KW if kw in text.lower()),
        sum(1 for kw in SPAM_KW if kw in text.lower()) / max(len(text.split()), 1),
        int(bool(re.search(r'\bB\d{3}\b', text))),
        int(bool(re.search(r'(?i)mersis', text))),
        len([w for w in text.split() if w.isupper() and len(w) > 2]) / max(len(text.split()), 1),
        len(re.findall(r'[\U0001F000-\U0001FFFF]', text)),
    ]


X_train_struct = csr_matrix(np.array([extract_structural(t) for t in train_df['text']], dtype=np.float32))
X_test_struct = csr_matrix(np.array([extract_structural(t) for t in test_df['text']], dtype=np.float32))

X_train = hstack([X_train_tfidf, X_train_struct])
X_test = hstack([X_test_tfidf, X_test_struct])

y_train = (train_df['label'] == 'spam').astype(int).values
y_test = (test_df['label'] == 'spam').astype(int).values

svm = LinearSVC(C=1.0, class_weight='balanced', max_iter=20000, random_state=42, dual='auto')
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['ham', 'spam'], digits=4))

# ════════════════════════════════════════════════════════════
# 2. Export model
# ════════════════════════════════════════════════════════════

print("Exporting model weights...")

vocab = tfidf.vocabulary_
idf = tfidf.idf_.tolist()
coef = svm.coef_[0]
intercept = float(svm.intercept_[0])

# TF-IDF features
tfidf_features = {}
for ngram, idx in vocab.items():
    w = float(coef[idx])
    if abs(w) > 1e-6:
        tfidf_features[ngram] = {"i": int(idx), "w": round(w, 6)}

# Structural feature weights (they come after TF-IDF in the coef vector)
n_tfidf = len(vocab)
struct_weights = {}
for i, name in enumerate(STRUCT_NAMES):
    w = float(coef[n_tfidf + i])
    struct_weights[name] = round(w, 6)

print(f"  TF-IDF features: {len(tfidf_features)}")
print(f"  Structural features: {len(struct_weights)}")
print(f"  Structural weights:")
for name, w in sorted(struct_weights.items(), key=lambda x: -abs(x[1])):
    print(f"    {name:25s} {w:+.4f}")

model_export = {
    "version": 2,
    "type": "tfidf_struct_linear_svm",
    "analyzer": "char_wb",
    "ngram_range": [2, 5],
    "sublinear_tf": True,
    "intercept": round(intercept, 6),
    "idf": [round(x, 6) for x in idf],
    "features": tfidf_features,
    "structural_weights": struct_weights,
}

model_path = os.path.join(XCODE_DIR, "spam_model.json")
with open(model_path, "w", encoding="utf-8") as f:
    json.dump(model_export, f, ensure_ascii=False)
model_size = os.path.getsize(model_path) / 1024
print(f"  Model saved: {model_path} ({model_size:.0f} KB)")

# ════════════════════════════════════════════════════════════
# 3. Export spam config
# ════════════════════════════════════════════════════════════

spam_config = {
    "spam_keywords": SPAM_KW,
    "opt_out_patterns": [
        "RET yaz", "IPTAL yaz", "SMS almak istemiyor", "tanitim iptali", "SMS Red"
    ],
    "phishing_domains": [
        "pubit.jp", "sniply.me", "fre.to", "t.ly/merit"
    ],
    "phishing_keywords": [
        "Varliklariniz Dondurulmus", "islemleriniz durdurulmustur",
        "Aktivasyon Adresiniz", "Finance Denetleme", "Emtia Denetleme",
        "MASAK Denetim", "MASAK Tarafindan",
        "Denetleme Kurulu Tarafindan", "JACKPOT"
    ],
    "url_shorteners": [
        "bit.ly", "shorturl.at", "tinyurl", "cutt.ly",
        "tnn.li", "dijital.li", "engho.me", "dfurl.com", "t2m.io"
    ],
}

config_path = os.path.join(XCODE_DIR, "spam_config.json")
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(spam_config, f, ensure_ascii=False, indent=2)
print(f"  Config saved: {config_path}")

# ════════════════════════════════════════════════════════════
# 4. Verify
# ════════════════════════════════════════════════════════════

print("\nVerifying exported model...")

def swift_predict(text, model_data):
    features = model_data["features"]
    idf_weights = model_data["idf"]
    intercept = model_data["intercept"]
    struct_w = model_data["structural_weights"]

    # TF-IDF
    padded = f" {text} "
    ngram_counts = {}
    for n in range(2, 6):
        for i in range(len(padded) - n + 1):
            ng = padded[i:i+n]
            if ng in features:
                ngram_counts[ng] = ngram_counts.get(ng, 0) + 1

    tfidf_vals = {}
    for ng, count in ngram_counts.items():
        feat = features[ng]
        tf = 1.0 + np.log(count) if count > 0 else 0.0
        tfidf_vals[ng] = tf * idf_weights[feat['i']]

    norm = np.sqrt(sum(v * v for v in tfidf_vals.values()))
    if norm > 0:
        for ng in tfidf_vals:
            tfidf_vals[ng] /= norm

    score = intercept
    for ng, val in tfidf_vals.items():
        score += val * features[ng]['w']

    # Structural features
    sf = extract_structural(text)
    for i, name in enumerate(STRUCT_NAMES):
        if name in struct_w:
            score += sf[i] * struct_w[name]

    return "spam" if score > 0 else "ham", score


test_messages = [
    "Akşam eve gelirken ekmek alır mısın?",
    "750 TL bonus kazanmak için linke tıkla! https://bit.ly/xyz",
    "Faturanızın son ödeme tarihi 15.03.2024'tür.",
    "BÜYÜK İNDİRİM! %80 kampanya fırsatını kaçırmayın! Hemen tıklayın!",
    "Toplantı saat 3'e alındı, haberin olsun.",
    "Sn.B.l.NA.N.CE-TR Kullanicisi MASAK Tarafindan islemleriniz durdurulmustur",
    "Netflix Casino 28.000TL DENEME BONUSU Minimum Yatirim 100 TL https://t2m.io/NetflixCasinoo",
]

print("\nExported model predictions:")
for msg in test_messages:
    label, score = swift_predict(msg, model_export)
    sklearn_pred = "spam" if svm.predict(hstack([tfidf.transform([msg]), csr_matrix(np.array([extract_structural(msg)], dtype=np.float32))]))[0] == 1 else "ham"
    match = "✓" if label == sklearn_pred else "✗"
    icon = "🚫" if label == "spam" else "✅"
    print(f"  {icon} [{label:4s}] score={score:+.3f} {match} {msg[:60]}")

# Full test set
matches = 0
for _, row in test_df.iterrows():
    swift_label, _ = swift_predict(row['text'], model_export)
    sklearn_label = "spam" if svm.predict(hstack([tfidf.transform([row['text']]), csr_matrix(np.array([extract_structural(row['text'])], dtype=np.float32))]))[0] == 1 else "ham"
    if swift_label == sklearn_label:
        matches += 1

print(f"\nFull test set: {matches}/{len(test_df)} match ({matches/len(test_df)*100:.1f}%)")
print("Done!")
