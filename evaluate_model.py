"""
Model Evaluation and Metrics Reporting
BCA Third-Year Summer Project: Plant Disease Detection Using AI/ML

This script evaluates an existing trained model on the unseen test dataset (dataset/test):
1. Verifies model existence ('models/plant_disease_model.keras').
2. Loads test data and applies consistent normalization.
3. Computes real classification metrics:
   - Overall Accuracy
   - Macro & Weighted Precision
   - Macro & Weighted Recall
   - Macro & Weighted F1-Score
4. Generates a Confusion Matrix heatmap ('results/confusion_matrix.png').
5. Generates a Class-Wise Performance chart ('results/class_performance.png').
6. Exports 'results/classification_report.txt' and 'results/metrics.json'.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def evaluate(
    model_path: str = "models/plant_disease_model.keras",
    classes_path: str = "models/class_names.json",
    test_dir: str = "dataset/test",
    results_dir: str = "results",
    batch_size: int = 32,
    img_size: int = 224
):
    print("==================================================")
    print("Plant Disease Detection - Model Evaluation")
    print("==================================================")

    # 1. Verify model file exists
    if not os.path.isfile(model_path):
        print(f"\n[!] ERROR: Model file not found at '{model_path}'.")
        print("    Model not trained yet. Please run 'python train_model.py' first.\n")
        sys.exit(1)

    # 2. Verify class labels exist
    if not os.path.isfile(classes_path):
        print(f"\n[!] ERROR: Class names mapping not found at '{classes_path}'.")
        print("    Please ensure class_names.json exists in models directory.\n")
        sys.exit(1)

    with open(classes_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    # 3. Verify test directory exists
    if not os.path.isdir(test_dir):
        print(f"\n[!] ERROR: Test dataset directory '{test_dir}' not found.")
        print("    Please run 'python preprocess_data.py --source_dir <path>' to generate the test split.\n")
        sys.exit(1)

    subdirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    if not subdirs:
        print(f"\n[!] ERROR: No class subfolders found inside '{test_dir}'.\n")
        sys.exit(1)

    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    print(f"[*] Loading trained model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    print(f"[*] Loading test dataset from: {test_dir}")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=False
    )

    rescale_layer = tf.keras.layers.Rescaling(1.0 / 255.0)
    test_ds_rescaled = test_ds.map(lambda x, y: (rescale_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    print("[*] Running batch inference across test images...")
    y_true = []
    y_pred = []

    for images, labels in test_ds_rescaled:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) == 0:
        print("[!] ERROR: No test samples were evaluated.")
        sys.exit(1)

    # Calculate metrics
    test_acc = accuracy_score(y_true, y_pred)
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0
    )

    print("\n" + "=" * 60)
    print(f"TEST EVALUATION RESULTS (Total Test Images: {len(y_true)})")
    print("=" * 60)
    print(f"Overall Accuracy : {test_acc * 100:.2f}%")
    print(f"Macro Precision  : {report_dict['macro avg']['precision'] * 100:.2f}%")
    print(f"Macro Recall     : {report_dict['macro avg']['recall'] * 100:.2f}%")
    print(f"Macro F1-Score   : {report_dict['macro avg']['f1-score'] * 100:.2f}%")
    print("=" * 60 + "\n")
    print(report_text)

    os.makedirs(results_dir, exist_ok=True)

    # 1. Save classification report text
    report_file = os.path.join(results_dir, "classification_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("PLANT DISEASE DETECTION - MODEL TEST EVALUATION\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total Test Samples: {len(y_true)}\n")
        f.write(f"Overall Accuracy  : {test_acc * 100:.2f}%\n")
        f.write("=" * 60 + "\n\n")
        f.write(report_text)
    print(f"[+] Detailed report saved: {report_file}")

    # 2. Save structured metrics JSON
    metrics_data = {
        "test_accuracy": round(float(test_acc) * 100, 2),
        "macro_f1": round(float(report_dict["macro avg"]["f1-score"]) * 100, 2),
        "weighted_f1": round(float(report_dict["weighted avg"]["f1-score"]) * 100, 2),
        "macro_precision": round(float(report_dict["macro avg"]["precision"]) * 100, 2),
        "macro_recall": round(float(report_dict["macro avg"]["recall"]) * 100, 2),
        "total_test_samples": int(len(y_true)),
        "class_performance": {}
    }
    for c in class_names:
        if c in report_dict:
            metrics_data["class_performance"][c] = {
                "precision": round(report_dict[c]["precision"] * 100, 2),
                "recall": round(report_dict[c]["recall"] * 100, 2),
                "f1_score": round(report_dict[c]["f1-score"] * 100, 2),
                "support": report_dict[c]["support"]
            }

    metrics_file = os.path.join(results_dir, "metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"[+] Metrics JSON saved: {metrics_file}")

    # 3. Generate Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(max(8, len(class_names) * 0.6), max(6, len(class_names) * 0.5)))
    clean_labels = [c.replace("___", " - ").replace("_", " ") for c in class_names]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=clean_labels,
        yticklabels=clean_labels
    )
    plt.title("Confusion Matrix (Test Evaluation)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=10, fontweight="bold")
    plt.ylabel("True Class", fontsize=10, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    cm_file = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_file, dpi=200)
    plt.close()
    print(f"[+] Confusion matrix plot saved: {cm_file}")

    # 4. Generate Class-Wise Performance Bar Chart
    plt.figure(figsize=(10, max(5, len(class_names) * 0.4)))
    f1_scores = [report_dict[c]["f1-score"] * 100 if c in report_dict else 0.0 for c in class_names]
    y_pos = np.arange(len(class_names))
    plt.barh(y_pos, f1_scores, color="#2d6a4f", edgecolor="#1b4332")
    plt.yticks(y_pos, clean_labels, fontsize=8)
    plt.xlabel("F1-Score (%)", fontsize=10, fontweight="bold")
    plt.xlim(0, 105)
    plt.title("Class-wise F1-Score Breakdown", fontsize=12, fontweight="bold", pad=12)
    for i, v in enumerate(f1_scores):
        plt.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    perf_file = os.path.join(results_dir, "class_performance.png")
    plt.savefig(perf_file, dpi=200)
    plt.close()
    print(f"[+] Class performance chart saved: {perf_file}")

    print("\n[+] Evaluation finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate plant disease detection model.")
    parser.add_argument("--model_path", type=str, default="models/plant_disease_model.keras")
    parser.add_argument("--classes_path", type=str, default="models/class_names.json")
    parser.add_argument("--test_dir", type=str, default="dataset/test")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    evaluate(
        model_path=args.model_path,
        classes_path=args.classes_path,
        test_dir=args.test_dir,
        results_dir=args.results_dir,
        batch_size=args.batch_size
    )
