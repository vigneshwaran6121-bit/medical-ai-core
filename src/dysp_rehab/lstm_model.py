import torch
import torch.nn as nn


class DysphagiaLSTM(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_layers=2, num_classes=5):
        """
        Args:
            input_size: 63 (21 landmarks * 3 coords)
            hidden_size: 128 (Neurons inside the LSTM memory)
            num_layers: 2 (Stacked LSTMs for complex patterns)
            num_classes: 5 (The number of exercises you have)
        """
        super(DysphagiaLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 1. The LSTM Layer (The "Temporal Engine")
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=0.2
        )

        # 2. The Classification Layer (The "Decision Maker")
        # Takes the last memory state and decides which exercise it is
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (Batch, Sequence_Len, Features) -> (Batch, 150, 63)

        # Initialize hidden states (h0, c0) with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        # out shape: (Batch, Seq_Len, Hidden_Size)
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        # We only care about the result after the full movement is done
        out = out[:, -1, :]

        # Pass through the classifier
        out = self.fc(out)
        return out


# Quick Test Block
if __name__ == "__main__":
    model = DysphagiaLSTM()
    dummy_input = torch.randn(4, 150, 63)  # Batch of 4 random sequences
    output = model(dummy_input)
    print(f"Model Test Output Shape: {output.shape} (Should be [4, 5])")
