"""
Model Training Pipeline
BCA Third-Year Summer Project: Plant Disease Detection Using AI/ML

This script executes the complete model training lifecycle:
1. Verifies dataset directories (dataset/train, dataset/validation, dataset/test).
2. Dynamically scans class names directly from the folder structure.
3. Saves class index mapping to 'models/class_names.json'.
4. Loads training, validation, and test datasets with TensorFlow tf.data pipelines.
5. Injects biological data augmentation layers (flips, rotations, zooms).
6. Instantiates MobileNetV2 transfer learning model with ImageNet weights.
7. Configures EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau callbacks.
8. Trains the classification head.
9. Evaluates the trained model on unseen test images.
10. Generates and saves:
    - results/accuracy.png (Accuracy Curves)
    - results/loss.png (Loss Curves)
    - results/confusion_matrix.png (Confusion Matrix)
    - results/classification_report.txt (Precision, Recall, F1-Score)
    - results/metrics.json (JSON metrics for the Streamlit dashboard)
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

from src.model import build_transfer_learning_model
from src.preprocessing import get_data_augmentation_layers


def verify_dataset_structure(dataset_dir: str = "dataset") -> bool:
    """Checks if train and validation directories exist and contain class subfolders."""
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "validation")

    if not os.path.isdir(train_dir):
        print(f"[!] ERROR: Training folder not found at '{train_dir}'.")
        print("    Please create dataset/train/<class_folders>/ or run preprocess_data.py first.")
        return False

    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    if not classes:
        print(f"[!] ERROR: No class subdirectories found in '{train_dir}'.")
        return False

    if not os.path.isdir(val_dir):
        print(f"[!] WARNING: Validation folder '{val_dir}' not found. Splitting from train will be needed.")

    return True


def scan_and_save_classes(train_dir: str, output_path: str = "models/class_names.json") -> list:
    """
    Scans class directory names and saves an index-to-name mapping.
    This guarantees dynamic adaptation to whatever dataset is supplied.
    """
    classes = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d)) and not d.startswith(".")
    ])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=4)

    print(f"[+] Discovered {len(classes)} classes from dataset.")
    print(f"[+] Class mapping saved to: {output_path}")
    return classes


def plot_and_save_training_curves(history, output_dir: str = "results"):
    """Generates and saves professional accuracy and loss curves."""
    os.makedirs(output_dir, exist_ok=True)

    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    # 1. Accuracy Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, acc, "o-", label="Training Accuracy", color="#2d6a4f", linewidth=2)
    if val_acc:
        plt.plot(epochs_range, val_acc, "s--", label="Validation Accuracy", color="#52b788", linewidth=2)
    plt.title("Model Training vs Validation Accuracy", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Accuracy", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    acc_path = os.path.join(output_dir, "accuracy.png")
    plt.savefig(acc_path, dpi=200)
    plt.close()
    print(f"[+] Accuracy curve saved: {acc_path}")

    # 2. Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, loss, "o-", label="Training Loss", color="#d90429", linewidth=2)
    if val_loss:
        plt.plot(epochs_range, val_loss, "s--", label="Validation Loss", color="#ef233c", linewidth=2)
    plt.title("Model Training vs Validation Loss", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss (Cross-Entropy)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    loss_path = os.path.join(output_dir, "loss.png")
    plt.savefig(loss_path, dpi=200)
    plt.close()
    print(f"[+] Loss curve saved: {loss_path}")


def evaluate_test_dataset(model, test_ds, class_names, output_dir: str = "results"):
    """Evaluates the model on test set, computing accuracy, precision, recall, f1, and confusion matrix."""
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    print("\n[*] Evaluating model on test dataset...")
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

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

    print(f"\n[+] Test Accuracy: {test_acc * 100:.2f}%\n")
    print(report_text)

    # Save text report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("PLANT DISEASE DETECTION - MODEL EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall Test Accuracy: {test_acc * 100:.2f}%\n\n")
        f.write(report_text)
    print(f"[+] Classification report saved: {report_path}")

    # Save JSON metrics for Streamlit Dashboard
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

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"[+] Metrics JSON saved: {metrics_path}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(max(8, len(class_names) * 0.55), max(6, len(class_names) * 0.45)))
    clean_labels = [c.replace("___", " - ").replace("_", " ") for c in class_names]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=clean_labels,
        yticklabels=clean_labels
    )
    plt.title("Confusion Matrix (Test Dataset)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=10, fontweight="bold")
    plt.ylabel("True Label", fontsize=10, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"[+] Confusion matrix saved: {cm_path}")


def train(
    dataset_dir: str = "dataset",
    models_dir: str = "models",
    results_dir: str = "results",
    epochs: int = 15,
    batch_size: int = 32,
    img_size: int = 224,
    learning_rate: float = 0.001
):
    """Executes the full training workflow."""
    import tensorflow as tf

    print("==================================================")
    print("Plant Disease Detection - AI Model Training")
    print("==================================================")

    if not verify_dataset_structure(dataset_dir):
        sys.exit(1)

    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "validation")
    test_dir = os.path.join(dataset_dir, "test")

    # Step 1: Scan and register class names
    class_names_path = os.path.join(models_dir, "class_names.json")
    class_names = scan_and_save_classes(train_dir, class_names_path)
    num_classes = len(class_names)

    # Step 2: Load Datasets via tf.keras.utils.image_dataset_from_directory
    print("\n[*] Loading image datasets...")
    target_dim = (img_size, img_size)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=target_dim,
        shuffle=True,
        seed=42
    )

    val_ds = None
    if os.path.isdir(val_dir) and any(os.path.isdir(os.path.join(val_dir, d)) for d in os.listdir(val_dir)):
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            labels="inferred",
            label_mode="categorical",
            class_names=class_names,
            color_mode="rgb",
            batch_size=batch_size,
            image_size=target_dim,
            shuffle=False
        )

    test_ds = None
    if os.path.isdir(test_dir) and any(os.path.isdir(os.path.join(test_dir, d)) for d in os.listdir(test_dir)):
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            labels="inferred",
            label_mode="categorical",
            class_names=class_names,
            color_mode="rgb",
            batch_size=batch_size,
            image_size=target_dim,
            shuffle=False
        )

    # Step 3: Rescaling & Augmentation
    rescale_layer = tf.keras.layers.Rescaling(1.0 / 255.0)
    train_ds = train_ds.map(lambda x, y: (rescale_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    aug_pipeline = get_data_augmentation_layers()
    if aug_pipeline is not None:
        train_ds = train_ds.map(lambda x, y: (aug_pipeline(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    if val_ds is not None:
        val_ds = val_ds.map(lambda x, y: (rescale_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    if test_ds is not None:
        test_ds_rescaled = test_ds.map(lambda x, y: (rescale_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        test_ds_rescaled = test_ds_rescaled.prefetch(buffer_size=tf.data.AUTOTUNE)
    else:
        test_ds_rescaled = None

    # Step 4: Build Model Architecture
    print(f"\n[*] Building MobileNetV2 Transfer Learning Model ({num_classes} classes)...")
    model, base_model = build_transfer_learning_model(
        num_classes=num_classes,
        input_shape=(img_size, img_size, 3),
        learning_rate=learning_rate
    )
    model.summary()

    # Step 5: Configure Callbacks
    model_save_path = os.path.join(models_dir, "plant_disease_model.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss" if val_ds else "loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_accuracy" if val_ds else "accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if val_ds else "loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Step 6: Train Model
    print(f"\n[*] Training for up to {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # Save final model if checkpoint didn't trigger
    if not os.path.isfile(model_save_path):
        model.save(model_save_path)
    print(f"\n[+] Best trained model saved to: {model_save_path}")

    # Step 7: Plot Accuracy & Loss Curves
    plot_and_save_training_curves(history, results_dir)

    # Step 8: Evaluate on Test Dataset if provided
    if test_ds_rescaled is not None:
        evaluate_test_dataset(model, test_ds_rescaled, class_names, results_dir)
    else:
        print("\n[!] Note: No test set found at 'dataset/test'. To evaluate test accuracy, populate dataset/test.")

    print("\n==================================================")
    print("Training Complete!")
    print("You can now run the web interface:")
    print("  streamlit run app.py")
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train plant disease classification model.")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Path to dataset root.")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory to save trained model.")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save charts and reports.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Adam learning rate.")

    args = parser.parse_args()
    train(
        dataset_dir=args.dataset_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
