import torch
import torch.nn.utils.prune as prune
from transformers import DistilBertTokenizerFast
from datasets import load_dataset
from torchmetrics.functional import accuracy
from torch.utils.data import DataLoader
from nn_model_train import DistilBERTFineTuner  # Import your model class
import matplotlib.pyplot as plt

# Load the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Load and preprocess the ag_news dataset
dataset = load_dataset("ag_news")
test_data = dataset["test"].map(lambda batch: tokenizer(batch['text'], padding='max_length', truncation=True, max_length=64), batched=True)
test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataloader = DataLoader(test_data, batch_size=32)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define pruning percentages
pruning_percentages = [0, 0.1, 0.3, 0.4]  # 0%, 10%, 20%, 30%

# Function to evaluate model accuracy
def evaluate_model(model, dataloader):
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass through the model
            outputs = model.model(input_ids=inputs, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())  # Store predictions
            all_labels.extend(labels.cpu().numpy())  # Store true labels

    # Calculate accuracy
    all_preds_tensor = torch.tensor(all_preds)
    all_labels_tensor = torch.tensor(all_labels)
    return accuracy(all_preds_tensor, all_labels_tensor, task="multiclass", num_classes=4)

# List to store accuracies for each pruning percentage
accuracies = []

# Loop through each pruning percentage
for amount in pruning_percentages:
    # Reload the model from the checkpoint for each pruning level to start fresh
    checkpoint_path = "checkpoints/best-checkpoint.ckpt"  # Path to your saved checkpoint
    model = DistilBERTFineTuner.load_from_checkpoint(checkpoint_path)
    model.to(device)  # Move model to the appropriate device
    model.model.to(device)

    # Apply pruning if amount > 0 (skip pruning for 0% level)
    if amount > 0:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=amount)

    # Evaluate and store accuracy
    test_accuracy = evaluate_model(model, test_dataloader)
    accuracies.append(test_accuracy.item())
    print(f"Test Accuracy after Pruning {amount*100:.0f}% of weights: {test_accuracy:.4f}")

# Plot accuracy vs. pruning percentage
plt.plot([p * 100 for p in pruning_percentages], accuracies, marker='o')
plt.xlabel('Pruning Percentage (%)')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy vs. Pruning Percentage')
plt.grid(True)
plt.show()
