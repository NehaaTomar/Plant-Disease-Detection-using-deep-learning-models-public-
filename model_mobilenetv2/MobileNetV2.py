import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

def get_model(num_classes):
    # Load base model with pretrained weights
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)

    # Modify the classifier to match the number of output classes (e.g., 39)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model
