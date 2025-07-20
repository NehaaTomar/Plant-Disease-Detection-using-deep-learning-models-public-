import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

model_names = ["cnn", "resnet18", "mobilenetv2", "efficientnet_b0"]
colors = ["blue", "green", "orange", "red"]
history_data = {}
final_val_acc = []
rmse_values = []

# === Load training history and predictions for all models ===
for model_name in model_names:
    folder = f"model_{model_name}"

    # History file path
    if model_name == "efficientnet_b0":
        pkl_path = os.path.join(folder, "efficientnetb0_history.pkl")
    else:
        pkl_path = os.path.join(folder, f"{model_name}_history.pkl")

    # Predictions CSV path
    csv_path = os.path.join(folder, f"{model_name}_predictions.csv")

    # Load training history
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            history = pickle.load(f)
        history_data[model_name] = {
            "train_acc": history.get("train_acc", []),
            "val_acc": history.get("val_acc", []),
            "train_loss": history.get("train_loss", []),
            "val_loss": history.get("val_loss", [])
        }
    else:
        print(f"❌ History not found for {model_name}")
        continue

    # Load predictions and calculate final val acc & RMSE
    if os.path.exists(csv_path):
        import pandas as pd
        from sklearn.metrics import accuracy_score

        df = pd.read_csv(csv_path)
        if "Actual" in df.columns and "Predicted" in df.columns:
            y_true = df["Actual"]
            y_pred = df["Predicted"]

            classes = sorted(list(set(y_true) | set(y_pred)))
            class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            y_true_idx = [class_to_idx[label] for label in y_true]
            y_pred_idx = [class_to_idx[label] for label in y_pred]

            acc = accuracy_score(y_true_idx, y_pred_idx)
            rmse = np.sqrt(np.mean((np.array(y_true_idx) - np.array(y_pred_idx)) ** 2))

            final_val_acc.append(acc)
            rmse_values.append(rmse)
        else:
            print(f"⚠️ Missing columns in CSV for {model_name}")
            final_val_acc.append(0)
            rmse_values.append(0)
    else:
        print(f"❌ Predictions not found for {model_name}")
        final_val_acc.append(0)
        rmse_values.append(0)

# === Plot Training vs Validation Accuracy & Loss ===
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# --- Accuracy comparison ---
for idx, model_name in enumerate(model_names):
    if model_name in history_data:
        epochs = range(1, len(history_data[model_name]["train_acc"]) + 1)
        axes[0, 0].plot(epochs, history_data[model_name]["train_acc"], linestyle='--', color=colors[idx], label=f"{model_name} Train")
        axes[0, 0].plot(epochs, history_data[model_name]["val_acc"], linestyle='-', color=colors[idx], label=f"{model_name} Val")

axes[0, 0].set_title("Training vs Validation Accuracy")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Accuracy")
axes[0, 0].legend()
axes[0, 0].grid(True)

# --- Loss comparison ---
for idx, model_name in enumerate(model_names):
    if model_name in history_data:
        epochs = range(1, len(history_data[model_name]["train_loss"]) + 1)
        axes[0, 1].plot(epochs, history_data[model_name]["train_loss"], linestyle='--', color=colors[idx], label=f"{model_name} Train")
        axes[0, 1].plot(epochs, history_data[model_name]["val_loss"], linestyle='-', color=colors[idx], label=f"{model_name} Val")

axes[0, 1].set_title("Training vs Validation Loss")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Loss")
axes[0, 1].legend()
axes[0, 1].grid(True)

# --- Final Validation Accuracy Bar Chart ---
axes[1, 0].bar(model_names, final_val_acc, color=colors)
axes[1, 0].set_ylim(0, 1)
axes[1, 0].set_ylabel("Accuracy")
axes[1, 0].set_title("Final Validation Accuracy Comparison")
for i, acc in enumerate(final_val_acc):
    axes[1, 0].text(i, acc + 0.01, f"{acc:.2f}", ha='center')

# --- RMSE Bar Chart ---
axes[1, 1].bar(model_names, rmse_values, color=colors)
axes[1, 1].set_ylabel("RMSE")
axes[1, 1].set_title("RMSE Comparison")
for i, rmse in enumerate(rmse_values):
    axes[1, 1].text(i, rmse + 0.01, f"{rmse:.2f}", ha='center')

fig.suptitle("Model Training Performance Comparison", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("comparison_output.png", dpi=300)
plt.close()
print("✅ Saved summary plot: comparison_output.png")
