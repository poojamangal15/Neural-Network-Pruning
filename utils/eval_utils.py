import torch
from sklearn.metrics import accuracy_score, f1_score

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
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return accuracy, f1

