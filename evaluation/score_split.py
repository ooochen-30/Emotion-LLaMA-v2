import json
import argparse
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_jsonl(file_path, split=None):
    answers = []
    responses = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if split is None or data.get("mode") == split:
                answers.append(data['answer'])
                if 'new_response' in data:
                    responses.append(data['new_response'].strip().lower())
                elif 'response' in data:
                    responses.append(data['response'].strip().lower())
    return answers, responses

# 绘制混淆矩阵
def plot_confusion_matrix(cm, labels, save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Evaluate JSONL Results")
    parser.add_argument('--file', type=str, required=True, help="Filename prefix (without .jsonl)")
    parser.add_argument('--split', type=str, default=None, help="Specify the mode to filter, e.g., 'test2'")
    args = parser.parse_args()

    filename = args.file
    split = args.split
    jsonl_path = filename + '.jsonl'

    if not os.path.exists(jsonl_path):
        print(f"Error: {jsonl_path} does not exist.")
        return

    answers, responses = load_jsonl(jsonl_path, split)

    if not answers:
        print(f"No data found for split '{split}'.")
        return

    # 计算准确率和加权F1
    acc = accuracy_score(answers, responses)
    f1 = f1_score(answers, responses, average='weighted')

    print(f"\n📄 Evaluating: {jsonl_path} (split: {split or 'all'})")
    print(f"Accuracy: {acc:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")

    # 计算混淆矩阵
    labels = sorted(list(set(answers + responses)))
    cm = confusion_matrix(answers, responses, labels=labels)

    print("\nClassification Report:")
    print(classification_report(answers, responses, labels=labels))

    # 绘制混淆矩阵并保存
    save_suffix = f"_{split}_confusion_matrix.png" if split else "_confusion_matrix.png"
    save_path = filename + save_suffix
    plot_confusion_matrix(cm, labels, save_path)

if __name__ == "__main__":
    main()

