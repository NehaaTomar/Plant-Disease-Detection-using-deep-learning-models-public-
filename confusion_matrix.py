import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from CNN import CNN  # Ensure CNN.py is present
from efficientnet_pytorch import EfficientNet  # pip install efficientnet_pytorch

# === CONFIG ===
VAL_PATH = "./dataset/val"
NUM_CLASSES = 39
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")

# === TRANSFORMS ===
transform_128 = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_cnn = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# === GENERAL FUNCTION FOR PREDICTIONS ===
def get_predictions(model, weights_path, transform):
    val_dataset = datasets.ImageFolder(VAL_PATH, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    class_names = val_dataset.classes

    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return y_true, y_pred, class_names

# === PLOT CONFUSION MATRIX FUNCTION ===
def plot_conf_matrix(y_true, y_pred, model_name, save_dir, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, max(10, len(class_names) * 0.25)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(include_values=False, ax=ax, xticks_rotation=90, cmap='Blues', colorbar=False)

    ax.set_title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(save_path)
    print(f"📊 Saved confusion matrix: {save_path}")
    plt.close()

# === MODEL EVALUATION FUNCTIONS ===
def evaluate_resnet():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    y_true, y_pred, class_names = get_predictions(model, "model_resnet18/resnet18_best.pth", transform_128)
    plot_conf_matrix(y_true, y_pred, "ResNet18", "model_resnet18", class_names)

def evaluate_mobilenet():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    y_true, y_pred, class_names = get_predictions(model, "model_mobilenetv2/mobilenetv2_best.pth", transform_128)
    plot_conf_matrix(y_true, y_pred, "MobileNetV2", "model_mobilenetv2", class_names)

def evaluate_efficientnet():
    weights_path = "model_efficientnet_b0/efficientnetb0_best.pth"

    model = efficientnet_b0(weights=None) # Do NOT use pretrained weights here
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    val_dataset = datasets.ImageFolder(VAL_PATH, transform=transform_128)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    class_names = val_dataset.classes

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    plot_conf_matrix(
        y_true, y_pred,
        model_name="EfficientNetB0",
        save_dir="model_efficientnet_b0",
        class_names=class_names
    )

def evaluate_cnn():
    model = CNN(K=NUM_CLASSES)
    y_true, y_pred, class_names = get_predictions(model, "model_cnn/cnn_best.pth", transform_cnn)
    plot_conf_matrix(y_true, y_pred, "CustomCNN", "model_cnn", class_names)


# === MAIN EXECUTION ===
if __name__ == "__main__":
    #evaluate_resnet()
    #evaluate_mobilenet()
    #evaluate_cnn()
    evaluate_efficientnet()
    print("\n✅ All confusion matrices generated!")