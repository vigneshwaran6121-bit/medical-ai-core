import torch
import torch.nn as nn
import torch.optim as optim
import os
from data_loader import get_data_loader
from lstm_model import DysphagiaLSTM

# ==========================================
# 1. TRAINING CONFIGURATION
# ==========================================
BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 50  # How many times to loop through the dataset

# Auto-detect GPU (Use CUDA if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training on: {device}")


def train():
    # 1. Setup Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROCESSED_PATH = os.path.join(SCRIPT_DIR, "..", "..", "AI_Dysphagia", "Processed")

    # 2. Load Data
    train_loader = get_data_loader(PROCESSED_PATH, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Initialize Model
    model = DysphagiaLSTM(num_classes=5).to(
        device
    )  # Ensure 5 matches your CLASS_MAP count

    # 4. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("--------------------------------------------------")
    print(f"Starting Training for {EPOCHS} Epochs...")
    print("--------------------------------------------------")

    # 5. Training Loop
    for epoch in range(EPOCHS):
        model.train()  # Set to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (features, labels) in enumerate(train_loader):
            # Move data to GPU/CPU
            features = features.to(device)
            labels = labels.to(device)

            # Forward Pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward Pass (Optimization)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate Accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_acc = 100 * correct / total
        print(
            f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {epoch_acc:.2f}%"
        )

    # 6. Save the Model
    save_path = os.path.join(SCRIPT_DIR, "dysphagia_model.pth")
    torch.save(model.state_dict(), save_path)
    print("--------------------------------------------------")
    print(f"✅ Training Complete. Model saved to: {save_path}")


if __name__ == "__main__":
    train()
