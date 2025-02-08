import torch
from sklearn.metrics import accuracy_score, f1_score
import os

def evaluate_model(model, dataloader, device):
    """
    Evaluates the model on the given dataloader and returns accuracy and F1 score.
    """
    all_preds = []
    all_labels = []
    model = model.to(device).to(torch.float32)  # Move model to the correct device and ensure correct data type
    model.eval()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device).to(torch.float32)  # Ensure images are on the same device and data type
            labels = labels.to(device)

            outputs = model(images)  # Perform forward pass
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

def count_parameters(model):
    """Counts the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_size_in_mb(model):
    torch.save(model.state_dict(), "temp.p")
    size_mb = os.path.getsize("temp.p") / (1024 * 1024)
    os.remove("temp.p")
    return size_mb