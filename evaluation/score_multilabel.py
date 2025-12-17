import json
import argparse
import os
import re
import numpy as np
from sklearn.metrics import f1_score, classification_report, average_precision_score
from sklearn.preprocessing import MultiLabelBinarizer

def split_labels(text):
    return [label.strip().lower() for label in re.split(r'[;,]', text) if label.strip()]

def load_multilabel_data(jsonl_path):
    true_labels = []
    pred_labels = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            true = split_labels(data['answer'])
            pred = split_labels(data['response']) 
            true_labels.append(true)
            pred_labels.append(pred)
    return true_labels, pred_labels

def main():
    parser = argparse.ArgumentParser(description="Evaluate Multi-label JSONL Results")
    parser.add_argument('--file', type=str, required=True, help="Filename prefix (without .jsonl)")
    args = parser.parse_args()

    filename = args.file
    jsonl_path = filename + '.jsonl'

    if not os.path.exists(jsonl_path):
        print(f"❌ Error: {jsonl_path} does not exist.")
        return

    true_labels, pred_labels = load_multilabel_data(jsonl_path)

    mlb = MultiLabelBinarizer()
    y_true_bin = mlb.fit_transform(true_labels)
    y_pred_bin = mlb.transform(pred_labels)

    f1_micro = f1_score(y_true_bin, y_pred_bin, average='micro')
    f1_samples = f1_score(y_true_bin, y_pred_bin, average='samples')

    print(f"\n📄 Evaluating: {jsonl_path}")
    print(f"F1 Score (micro):   {f1_micro:.4f}")
    print(f"F1 Score (samples): {f1_samples:.4f}")

    intersection = (y_true_bin & y_pred_bin).sum(axis=1)
    union = (y_true_bin | y_pred_bin).sum(axis=1)
    jaccard_scores = intersection / (union + 1e-6)
    example_based_acc = jaccard_scores.mean()
    print(f"Example-based Accuracy (Jaccard): {example_based_acc:.4f}")

    y_pred_scores = y_pred_bin.astype(float) 
    ap_per_label = average_precision_score(y_true_bin, y_pred_scores, average=None)
    map_score = average_precision_score(y_true_bin, y_pred_scores, average="macro")

    print(f"\n📊 mAP (mean Average Precision): {map_score:.4f}")
    print("AP per label:")
    for label, ap in zip(mlb.classes_, ap_per_label):
        print(f"  {label:<20}: {ap:.4f}")

    print("\n📑 Classification Report:")
    print(classification_report(y_true_bin, y_pred_bin, target_names=mlb.classes_))

if __name__ == "__main__":
    main()