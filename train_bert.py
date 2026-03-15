"""
Turkish SMS Spam Filter — BERT Pipeline (GPU)
Fine-tunes dbmdz/bert-base-turkish-cased for SMS spam classification.

Requirements:
    pip install torch transformers datasets accelerate scikit-learn

Best with: NVIDIA GPU (16GB+ VRAM) or Apple Silicon MPS
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# ── Check GPU availability ──
import torch
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    print("Using Apple Silicon MPS")
else:
    DEVICE = "cpu"
    print("WARNING: No GPU detected. Training will be very slow.")

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict

MODEL_NAME = "dbmdz/bert-base-turkish-cased"
OUTPUT_DIR = "models/bert_spam_filter"
DATA_DIR = "dataset"
NUM_LABELS = 2
LABEL2ID = {"ham": 0, "spam": 1}
ID2LABEL = {0: "ham", 1: "spam"}

# ════════════════════════════════════════════════════════════
# 1. Load and prepare data
# ════════════════════════════════════════════════════════════

print("\nLoading data...")
train_df = pd.read_csv(f"{DATA_DIR}/train.csv").fillna('')
val_df = pd.read_csv(f"{DATA_DIR}/val.csv").fillna('')
test_df = pd.read_csv(f"{DATA_DIR}/test.csv").fillna('')

for df in [train_df, val_df, test_df]:
    df['label_id'] = df['label'].map(LABEL2ID)

dataset = DatasetDict({
    'train': Dataset.from_pandas(train_df[['text', 'label_id']].rename(columns={'label_id': 'label'})),
    'validation': Dataset.from_pandas(val_df[['text', 'label_id']].rename(columns={'label_id': 'label'})),
    'test': Dataset.from_pandas(test_df[['text', 'label_id']].rename(columns={'label_id': 'label'})),
})

print(f"Train: {len(dataset['train'])} | Val: {len(dataset['validation'])} | Test: {len(dataset['test'])}")

# ════════════════════════════════════════════════════════════
# 2. Tokenize
# ════════════════════════════════════════════════════════════

print(f"\nLoading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    return tokenizer(
        batch['text'],
        padding='max_length',
        truncation=True,
        max_length=256,  # SMS messages are short
    )

dataset = dataset.map(tokenize_fn, batched=True, batch_size=256)
dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# ════════════════════════════════════════════════════════════
# 3. Model
# ════════════════════════════════════════════════════════════

print(f"Loading model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

# Compute class weights for imbalanced data
train_labels = train_df['label_id'].values
n_spam = (train_labels == 1).sum()
n_ham = (train_labels == 0).sum()
weight_spam = len(train_labels) / (2 * n_spam)
weight_ham = len(train_labels) / (2 * n_ham)
class_weights = torch.tensor([weight_ham, weight_spam], dtype=torch.float32).to(DEVICE)
print(f"Class weights — ham: {weight_ham:.3f}, spam: {weight_spam:.3f}")


class WeightedTrainer(Trainer):
    """Trainer with class-weighted loss for imbalanced data."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ════════════════════════════════════════════════════════════
# 4. Training configuration
# ════════════════════════════════════════════════════════════

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,

    # Training
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",

    # Evaluation
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,

    # Optimization
    fp16=(DEVICE == "cuda"),  # Mixed precision on NVIDIA
    dataloader_num_workers=0,  # Windows requires __name__ guard for multiprocessing
    gradient_accumulation_steps=1,

    # Logging
    logging_steps=50,
    report_to="none",

    # Save
    save_total_limit=2,
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision = (preds[labels == 1] == 1).sum() / max((preds == 1).sum(), 1)
    recall = (preds[labels == 1] == 1).sum() / max((labels == 1).sum(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (preds == labels).mean()
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


# ════════════════════════════════════════════════════════════
# 5. Train
# ════════════════════════════════════════════════════════════

print(f"\nTraining on {DEVICE}...")

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()

# ════════════════════════════════════════════════════════════
# 6. Evaluate on test set
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("TEST SET RESULTS")
print("=" * 60)

test_results = trainer.predict(dataset['test'])
test_preds = np.argmax(test_results.predictions, axis=-1)
test_labels = test_results.label_ids

print(classification_report(
    test_labels, test_preds,
    target_names=['ham', 'spam'],
    digits=4
))

cm = confusion_matrix(test_labels, test_preds)
print(f"Confusion Matrix:")
print(f"              Pred Ham  Pred Spam")
print(f"  Actual Ham   {cm[0][0]:6d}    {cm[0][1]:6d}")
print(f"  Actual Spam  {cm[1][0]:6d}    {cm[1][1]:6d}")
print(f"\n  False positives (ham->spam): {cm[0][1]}")
print(f"  False negatives (spam->ham): {cm[1][0]}")

# ════════════════════════════════════════════════════════════
# 7. Error analysis
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("ERROR ANALYSIS")
print("=" * 60)

test_texts = test_df['text'].values

fp_mask = (test_labels == 0) & (test_preds == 1)
if fp_mask.sum() > 0:
    print(f"\nFalse Positives ({fp_mask.sum()}):")
    for t in test_texts[fp_mask][:5]:
        print(f"  > {t[:100]}...")

fn_mask = (test_labels == 1) & (test_preds == 0)
if fn_mask.sum() > 0:
    print(f"\nFalse Negatives ({fn_mask.sum()}):")
    for t in test_texts[fn_mask][:5]:
        print(f"  > {t[:100]}...")

# ════════════════════════════════════════════════════════════
# 8. Save model
# ════════════════════════════════════════════════════════════

print("\nSaving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

model_size = sum(
    os.path.getsize(os.path.join(OUTPUT_DIR, f))
    for f in os.listdir(OUTPUT_DIR)
    if os.path.isfile(os.path.join(OUTPUT_DIR, f))
) / (1024 * 1024)
print(f"Model saved to {OUTPUT_DIR}/ ({model_size:.0f} MB)")

# ════════════════════════════════════════════════════════════
# 9. Inference test
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("INFERENCE TEST")
print("=" * 60)

model.eval()
test_messages = [
    "Akşam eve gelirken ekmek alır mısın?",
    "750 TL bonus kazanmak için linke tıkla! https://bit.ly/xyz",
    "Faturanızın son ödeme tarihi 15.03.2024'tür.",
    "BÜYÜK İNDİRİM! %80 kampanya fırsatını kaçırmayın! Hemen tıklayın!",
    "Toplantı saat 3'e alındı, haberin olsun.",
    "Sn.B.l.NA.N.CE-TR Kullanicisi MASAK Tarafindan islemleriniz durdurulmustur",
]

for msg in test_messages:
    inputs = tokenizer(msg, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred_id = probs.argmax().item()
    pred_label = ID2LABEL[pred_id]
    spam_p = probs[1].item()

    icon = "[X]" if pred_label == "spam" else "[O]"
    print(f"  {icon} [{pred_label:4s}] (spam: {spam_p:.2%}) {msg[:70]}")

print("\nDone!")
