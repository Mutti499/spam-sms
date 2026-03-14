"""
Turkish SMS Spam Filter — SVM Pipeline (CPU)
Character n-gram TF-IDF + handcrafted features + Linear SVM

Runs on any machine. Produces a small, fast model suitable for mobile deployment.
"""

import os
import re
import json
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')

MODEL_DIR = "models"
DATA_DIR = "dataset"
os.makedirs(MODEL_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════
# 1. Load data
# ════════════════════════════════════════════════════════════

train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
val_df = pd.read_csv(f"{DATA_DIR}/val.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")

# Fill NaN texts
for df in [train_df, val_df, test_df]:
    df['text'] = df['text'].fillna('')

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(f"Train spam ratio: {(train_df['label']=='spam').mean():.1%}")

# ════════════════════════════════════════════════════════════
# 2. Feature extraction
# ════════════════════════════════════════════════════════════

class StructuralFeatureExtractor:
    """Extract handcrafted features from SMS text."""

    SPAM_KEYWORDS = [
        'kampanya', 'indirim', 'fırsat', 'firsat', 'bonus', 'kazan',
        'hediye', 'ücretsiz', 'ucretsiz', 'tıkla', 'tikla', 'kupon',
        'kod', 'bedava', 'kazandınız', 'kazandiniz', 'promosyon',
        'sepet', 'fiyat', 'stok', 'sınırlı', 'sinirli', 'acele',
        'kaçırma', 'kacirma', 'hemen'
    ]

    def transform(self, texts):
        features = []
        for text in texts:
            f = self._extract(text)
            features.append(f)
        return csr_matrix(np.array(features, dtype=np.float32))

    def _extract(self, text):
        length = len(text)
        safe_len = max(length, 1)

        return [
            # Length features
            length,
            len(text.split()),
            np.log1p(length),

            # URL features
            len(re.findall(r'https?://\S+|www\.\S+', text)),
            int(bool(re.search(r'bit\.ly|shorturl|tinyurl|t\.co|cutt\.ly|tnn\.li|dijital\.li|turkcell\.li|engho\.me|dfurl', text, re.IGNORECASE))),

            # Character distribution
            sum(c.isdigit() for c in text) / safe_len,
            sum(c.isupper() for c in text) / safe_len,
            sum(not c.isalnum() and not c.isspace() for c in text) / safe_len,

            # Currency / price
            int(bool(re.search(r'\bTL\b|₺|EUR|USD', text))),
            int(bool(re.search(r'\d+[\.,]?\d*\s*TL', text))),

            # Phone number
            int(bool(re.search(r'(?:\d[\d\s-]{8,}\d)', text))),

            # Punctuation
            text.count('!'),
            text.count('?'),

            # Spam structural signals
            int(bool(re.search(r'(?i)(RET yaz|IPTAL yaz|SMS almak istemiyorsan|tanitim iptali|SMS Red)', text))),
            int(bool(re.search(r'(?i)(hemen.*tıkla|hemen.*kaydol|hemen.*başvur|hemen.*indir|hemen.*ara)', text))),
            int(bool(re.search(r'%\d+', text))),
            int(bool(re.search(r'(?i)(SON \d+ GÜN|SON GÜN|SON SAATLER|SON \d+ HAFTA)', text))),

            # Spam keyword density
            sum(1 for kw in self.SPAM_KEYWORDS if kw in text.lower()),
            sum(1 for kw in self.SPAM_KEYWORDS if kw in text.lower()) / max(len(text.split()), 1),

            # Sender code (B001, B016, etc.)
            int(bool(re.search(r'\bB\d{3}\b', text))),

            # Has Mersis number (commercial messages)
            int(bool(re.search(r'(?i)mersis', text))),

            # All-caps word ratio
            len([w for w in text.split() if w.isupper() and len(w) > 2]) / max(len(text.split()), 1),

            # Emoji count
            len(re.findall(r'[\U0001F000-\U0001FFFF]', text)),
        ]

    @property
    def feature_names(self):
        return [
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


# ════════════════════════════════════════════════════════════
# 3. Build feature matrices
# ════════════════════════════════════════════════════════════

print("\nBuilding features...")

# Character n-gram TF-IDF
tfidf = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(2, 5),
    max_features=30000,
    sublinear_tf=True,
    strip_accents=None,  # Keep Turkish characters
    min_df=2,
)

X_train_tfidf = tfidf.fit_transform(train_df['text'])
X_val_tfidf = tfidf.transform(val_df['text'])
X_test_tfidf = tfidf.transform(test_df['text'])

print(f"  TF-IDF features: {X_train_tfidf.shape[1]}")

# Structural features
struct_extractor = StructuralFeatureExtractor()
X_train_struct = struct_extractor.transform(train_df['text'])
X_val_struct = struct_extractor.transform(val_df['text'])
X_test_struct = struct_extractor.transform(test_df['text'])

print(f"  Structural features: {X_train_struct.shape[1]}")

# Combine
X_train = hstack([X_train_tfidf, X_train_struct])
X_val = hstack([X_val_tfidf, X_val_struct])
X_test = hstack([X_test_tfidf, X_test_struct])

y_train = train_df['label'].values
y_val = val_df['label'].values
y_test = test_df['label'].values

print(f"  Total features: {X_train.shape[1]}")

# ════════════════════════════════════════════════════════════
# 4. Hyperparameter search
# ════════════════════════════════════════════════════════════

print("\nHyperparameter search (5-fold CV)...")

param_grid = {
    'C': [0.1, 1.0, 5.0],
    'class_weight': ['balanced'],
}

best_f1 = 0
best_params = {}

for C in param_grid['C']:
    for cw in param_grid['class_weight']:
        model = LinearSVC(C=C, class_weight=cw, max_iter=10000, random_state=42, dual='auto')
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1_macro')
        mean_f1 = scores.mean()
        print(f"    C={C}, cw={cw} → F1={mean_f1:.4f}")
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_params = {'C': C, 'class_weight': cw}

print(f"  Best params: {best_params} (CV F1: {best_f1:.4f})")

# ════════════════════════════════════════════════════════════
# 5. Train final model
# ════════════════════════════════════════════════════════════

print("\nTraining final model...")

# Use CalibratedClassifierCV for probability estimates
base_svm = LinearSVC(**best_params, max_iter=10000, random_state=42, dual='auto')
model = CalibratedClassifierCV(base_svm, cv=5)
model.fit(X_train, y_train)

# ════════════════════════════════════════════════════════════
# 6. Evaluate
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("VALIDATION SET RESULTS")
print("=" * 60)

y_val_pred = model.predict(X_val)
print(classification_report(y_val, y_val_pred, digits=4))

cm = confusion_matrix(y_val, y_val_pred, labels=['ham', 'spam'])
print(f"Confusion Matrix:")
print(f"              Pred Ham  Pred Spam")
print(f"  Actual Ham   {cm[0][0]:6d}    {cm[0][1]:6d}")
print(f"  Actual Spam  {cm[1][0]:6d}    {cm[1][1]:6d}")
print(f"\n  False positives (ham→spam): {cm[0][1]}")
print(f"  False negatives (spam→ham): {cm[1][0]}")

print("\n" + "=" * 60)
print("TEST SET RESULTS")
print("=" * 60)

y_test_pred = model.predict(X_test)
print(classification_report(y_test, y_test_pred, digits=4))

cm = confusion_matrix(y_test, y_test_pred, labels=['ham', 'spam'])
print(f"Confusion Matrix:")
print(f"              Pred Ham  Pred Spam")
print(f"  Actual Ham   {cm[0][0]:6d}    {cm[0][1]:6d}")
print(f"  Actual Spam  {cm[1][0]:6d}    {cm[1][1]:6d}")
print(f"\n  False positives (ham→spam): {cm[0][1]}")
print(f"  False negatives (spam→ham): {cm[1][0]}")

# Precision at different thresholds
y_test_proba = model.predict_proba(X_test)
spam_idx = list(model.classes_).index('spam')
spam_proba = y_test_proba[:, spam_idx]
y_test_binary = (y_test == 'spam').astype(int)

print(f"\nPrecision at different spam thresholds:")
for threshold in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
    preds = (spam_proba >= threshold).astype(int)
    tp = ((preds == 1) & (y_test_binary == 1)).sum()
    fp = ((preds == 1) & (y_test_binary == 0)).sum()
    fn = ((preds == 0) & (y_test_binary == 1)).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    print(f"  threshold={threshold:.2f}: precision={prec:.4f}, recall={rec:.4f}, flagged={preds.sum()}")

# ════════════════════════════════════════════════════════════
# 7. Error analysis
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("ERROR ANALYSIS")
print("=" * 60)

# False positives (ham predicted as spam) — most critical
fp_mask = (y_test == 'ham') & (y_test_pred == 'spam')
fp_texts = test_df.loc[fp_mask, 'text'].values
if len(fp_texts) > 0:
    print(f"\nFalse Positives ({len(fp_texts)} ham classified as spam):")
    for t in fp_texts[:5]:
        print(f"  → {t[:100]}...")
else:
    print("\nNo false positives! ✓")

# False negatives (spam predicted as ham)
fn_mask = (y_test == 'spam') & (y_test_pred == 'ham')
fn_texts = test_df.loc[fn_mask, 'text'].values
if len(fn_texts) > 0:
    print(f"\nFalse Negatives ({len(fn_texts)} spam classified as ham):")
    for t in fn_texts[:5]:
        print(f"  → {t[:100]}...")
else:
    print("\nNo false negatives! ✓")

# ════════════════════════════════════════════════════════════
# 8. Feature importance
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("TOP FEATURES")
print("=" * 60)

# Get coefficients from base estimator
base = model.calibrated_classifiers_[0].estimator
coef = base.coef_[0]

# TF-IDF feature names
tfidf_names = tfidf.get_feature_names_out()
struct_names = struct_extractor.feature_names
all_names = list(tfidf_names) + struct_names

# Top spam-indicating features
spam_top = np.argsort(coef)[-20:][::-1]
print("\nTop SPAM indicators:")
for idx in spam_top:
    name = all_names[idx] if idx < len(all_names) else f"feat_{idx}"
    print(f"  {coef[idx]:+.4f}  {repr(name)}")

# Top ham-indicating features
ham_top = np.argsort(coef)[:10]
print("\nTop HAM indicators:")
for idx in ham_top:
    name = all_names[idx] if idx < len(all_names) else f"feat_{idx}"
    print(f"  {coef[idx]:+.4f}  {repr(name)}")

# ════════════════════════════════════════════════════════════
# 9. Save model
# ════════════════════════════════════════════════════════════

model_bundle = {
    'model': model,
    'tfidf': tfidf,
    'struct_extractor': struct_extractor,
    'best_params': best_params,
    'feature_count': X_train.shape[1],
}

model_path = os.path.join(MODEL_DIR, "svm_spam_filter.pkl")
with open(model_path, 'wb') as f:
    pickle.dump(model_bundle, f)

model_size = os.path.getsize(model_path) / (1024 * 1024)
print(f"\nModel saved to {model_path} ({model_size:.1f} MB)")

# ════════════════════════════════════════════════════════════
# 10. Quick inference test
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("INFERENCE TEST")
print("=" * 60)

test_messages = [
    "Akşam eve gelirken ekmek alır mısın?",
    "750 TL bonus kazanmak için linke tıkla! https://bit.ly/xyz",
    "Faturanızın son ödeme tarihi 15.03.2024'tür.",
    "BÜYÜK İNDİRİM! %80 kampanya fırsatını kaçırmayın! Hemen tıklayın!",
    "Toplantı saat 3'e alındı, haberin olsun.",
    "Sn.B.l.NA.N.CE-TR Kullanicisi MASAK Tarafindan islemleriniz durdurulmustur",
]

for msg in test_messages:
    X_msg_tfidf = tfidf.transform([msg])
    X_msg_struct = struct_extractor.transform([msg])
    X_msg = hstack([X_msg_tfidf, X_msg_struct])

    pred = model.predict(X_msg)[0]
    proba = model.predict_proba(X_msg)[0]
    spam_p = proba[list(model.classes_).index('spam')]

    icon = "🚫" if pred == "spam" else "✅"
    print(f"  {icon} [{pred:4s}] (spam: {spam_p:.2%}) {msg[:70]}")

print("\nDone!")
