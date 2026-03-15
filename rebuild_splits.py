"""
Rebuild train/val/test splits from the relabeled dataset.
Uses the same stratified 80/10/10 split with the same random_state
so the split is reproducible.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "dataset/full_with_features_relabeled.csv"
OUTPUT_DIR = "dataset"

df = pd.read_csv(INPUT_FILE)
print(f"Loaded: {len(df)} messages")
print(f"  Labels: {dict(df['label'].value_counts())}")

# Check how many labels changed
if 'original_label' in df.columns:
    changed = (df['label'] != df['original_label']).sum()
    print(f"  Corrected labels: {changed}")

# Stratified split: 80/10/10 (same random_state as prepare_data.py)
train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=42,
    stratify=df['label']
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42,
    stratify=temp_df['label']
)

print(f"\nSplit sizes:")
for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    ham = (split_df['label'] == 'ham').sum()
    spam = (split_df['label'] == 'spam').sum()
    print(f"  {name:5s}: {len(split_df):5d} ({ham} ham, {spam} spam, {spam/len(split_df)*100:.1f}% spam)")

# Save
for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    out = split_df[['text', 'label']]
    out.to_csv(os.path.join(OUTPUT_DIR, f"{name}.csv"), index=False)
    print(f"  Saved {name}.csv")

print("\nDone! Ready to retrain SVM.")
