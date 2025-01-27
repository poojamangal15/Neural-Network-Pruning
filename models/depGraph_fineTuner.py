import torch
import torch.nn as nn
from .alexNet_fineTuner import AlexNetFineTuner
from utils.device_utils import get_device

class DepGraphFineTuner(AlexNetFineTuner):
    def __init__(self, learning_rate=1e-4, num_classes=10):
        super(DepGraphFineTuner, self).__init__(learning_rate=learning_rate, num_classes=num_classes)
        self.custom_device = get_device()

    def fine_tune_model(self, train_dataloader, val_dataloader, device, epochs=1, learning_rate=1e-5):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        self.to(device).to(torch.float32)  # ensure model is float32 on the chosen device

        for epoch in range(epochs):
            self.train()
            for batch in train_dataloader:
                inputs, targets = batch

                inputs = inputs.to(device, dtype=torch.float32)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            self.eval()
            with torch.no_grad():
                total_loss = 0
                correct = 0
                for batch in val_dataloader:
                    inputs, targets = batch

                    inputs = inputs.to(device, dtype=torch.float32)
                    targets = targets.to(device)

                    outputs = self(inputs)
                    total_loss += criterion(outputs, targets).item()
                    correct += (outputs.argmax(dim=1) == targets).sum().item()

                val_accuracy = correct / len(val_dataloader.dataset)
                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Validation Accuracy: {val_accuracy:.4f}, "
                    f"Loss: {total_loss:.4f}"
                )
