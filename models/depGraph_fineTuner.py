import torch
import torch.nn as nn
from .alexNet_fineTuner import AlexNetFineTuner
from utils.device_utils import get_device

class DepGraphFineTuner(AlexNetFineTuner):
    def __init__(self, learning_rate=1e-4, num_classes=10):
        super(DepGraphFineTuner, self).__init__(learning_rate=learning_rate, num_classes=num_classes)
        self.custom_device = get_device()

    def fine_tune_model(self, train_loader, val_loader, epochs=3, learning_rate=1e-5):
        # A simple post-pruning fine-tuning loop
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        self.to(self.custom_device)

        for epoch in range(epochs):
            self.train()
            train_loss, train_correct = 0, 0
            for images, labels in train_loader:
                images, labels = images.to(self.custom_device), labels.to(self.custom_device)

                optimizer.zero_grad()
                outputs = self(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_correct += (outputs.argmax(dim=1) == labels).sum().item()

            self.eval()
            val_loss, val_correct = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.custom_device), labels.to(self.custom_device)
                    outputs = self(images)
                    loss = loss_fn(outputs, labels)
                    val_loss += loss.item()
                    val_correct += (outputs.argmax(dim=1) == labels).sum().item()

            train_acc = train_correct / len(train_loader.dataset)
            val_acc = val_correct / len(val_loader.dataset)

            print(f"Post-Pruning Fine-Tune Epoch {epoch + 1}/{epochs}, "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
