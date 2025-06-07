import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# === CONFIG ===
model_names = ["cnn", "resnet18", "mobilenetv2", "efficientnet_b0"]

for model_name in model_names:
    print(f"🔍 Evaluating: {model_name}")

    csv_file = f"model_{model_name}/{model_name}_predictions.csv"
    output_text_file = f"model_{model_name}/METRICS.txt"
    output_image_file = f"model_{model_name}/CONFUSION_MATRIX.png"

    # === LOAD CSV ===
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"❌ File not found: {csv_file}")
        continue

    y_true = df["Actual"]
    y_pred = df["Predicted"]
# === METRICS ===
    report = classification_report(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    labels = sorted(list(set(y_true)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # === SAVE REPORT TO TXT FILE ===
    with open(output_text_file, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name.upper()}\n")
        f.write("\n📊 Classification Report:\n")
        f.write(report)
        f.write(f"\n\n🎯 Accuracy: {accuracy:.4f}\n")
        f.write("\n🧮 Confusion Matrix:\n")
        for row in cm:
            f.write(" ".join(map(str, row)) + "\n")

    print(f"📄 Saved metrics report: {output_text_file}")

    # === PLOT & SAVE HEATMAP ===
    plt.figure(figsize=(20, max(14, len(labels) * 0.6)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"{model_name.upper()} Confusion Matrix")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    plt.savefig(output_image_file, dpi=300)
    plt.close()
    print(f"🖼️ Saved heatmap image: {output_image_file}")
