import os
import pickle
import matplotlib.pyplot as plt

# === CONFIGURATION ===
history_paths = {
    "CNN": "model_cnn/cnn_history.pkl",
    "ResNet18": "model_resnet18/resnet18_history.pkl",
    "MobileNetV2": "model_mobilenetv2/mobilenetv2_history.pkl",
    "EfficientNetB0": "model_efficientnet_b0/efficientnetb0_history.pkl"
}

# === CREATE OUTPUT FOLDER ===
os.makedirs("output_history_plot", exist_ok=True)

# === LOAD HISTORIES ===
histories = {}
for model_name, path in history_paths.items():
    if os.path.exists(path):
        with open(path, "rb") as f:
            histories[model_name] = pickle.load(f)
    else:
        print(f"❌ Missing file: {path}")

# === PLOTTING FUNCTION ===
def plot_metric(histories, metric_name, ylabel, filename):
    plt.figure(figsize=(10, 6))
    for model_name, history in histories.items():
        plt.plot(history[metric_name], label=model_name)
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"output_history_plot/{filename}")
    plt.close()
    print(f"✅ Saved: output_history_plot/{filename}")

# === GENERATE ALL PLOTS ===
plot_metric(histories, "train_loss", "Training Loss", "train_loss.png")
plot_metric(histories, "val_loss", "Validation Loss", "val_loss.png")
plot_metric(histories, "train_acc", "Training Accuracy (%)", "train_acc.png")
plot_metric(histories, "val_acc", "Validation Accuracy (%)", "val_acc.png")