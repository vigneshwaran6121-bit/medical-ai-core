# kannamma
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. CONFIGURATION
# ==========================================
# We pad/cut all videos to this length (e.g., 150 frames = ~5 seconds)
SEQUENCE_LENGTH = 150

# Map exercise names found in filenames to IDs
CLASS_MAP = {
    "ChinDown": 0,
    "Chintuck": 1,  # Note: Filenames might be case-sensitive, check your npy files
    "HeadTilt": 2,  # Matches "Headtiltexercise"
    "HeadTurn": 3,
    "ENGNeck": 4,  # Matches "ENGNeck"
    "Neck": 4,  # Alternate name
}


class DysphagiaDataset(Dataset):
    def __init__(self, processed_dir, transform=None):
        self.files = glob.glob(os.path.join(processed_dir, "*.npy"))
        self.transform = transform

        # Verify we found files
        print(f"Found {len(self.files)} processed samples.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        filename = os.path.basename(file_path).lower()  # Case insensitive matching

        # 1. Load Data (Frames, 21, 3)
        data = np.load(file_path)

        # 2. Determine Label from Filename
        label = -1
        for key, val in CLASS_MAP.items():
            if key.lower() in filename:
                label = val
                break

        # If no label found, mark as 'unknown' or error (usually 0 or specific ID)
        if label == -1:
            # print(f"Warning: No label found for {filename}")
            label = 0

        # 3. Preprocessing: Flatten Features
        # Input: (Frames, 21, 3) -> Output: (Frames, 63)
        # We flatten 21*3 = 63 features per time step
        data = data.reshape(data.shape[0], -1)

        # 4. Padding / Truncating to Fixed Length
        current_len = data.shape[0]
        feature_dim = data.shape[1]  # Should be 63

        if current_len > SEQUENCE_LENGTH:
            # Truncate (Cut off the end)
            data = data[:SEQUENCE_LENGTH, :]
        else:
            # Pad (Add zeros to the end)
            padding = np.zeros((SEQUENCE_LENGTH - current_len, feature_dim))
            data = np.vstack((data, padding))

        # 5. Convert to PyTorch Tensors
        # Data: Float32 (for Neural Net), Label: Long (for Classification)
        return torch.tensor(data, dtype=torch.float32), torch.tensor(
            label, dtype=torch.long
        )


def get_data_loader(data_dir, batch_size=16, shuffle=True):
    dataset = DysphagiaDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ==========================================
# 2. TEST BLOCK (Run this file to verify)
# ==========================================
if __name__ == "__main__":
    # Robust path to your 'Processed' folder
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROCESSED_PATH = os.path.join(SCRIPT_DIR, "..", "..", "AI_Dysphagia", "Processed")

    print(f"Checking data in: {PROCESSED_PATH}")

    if not os.path.exists(PROCESSED_PATH):
        print("❌ Error: Processed folder not found!")
    else:
        # Try loading a batch
        loader = get_data_loader(PROCESSED_PATH, batch_size=4)

        try:
            features, labels = next(iter(loader))
            print("\n✅ SUCCESS! DataLoader is working.")
            print(f"Batch Shape: {features.shape}  (BatchSize, SeqLen, Features)")
            print(f"Labels: {labels}")
            print(f"Feature Vector Size: {features.shape[2]} (Should be 63 for 21*3)")
        except Exception as e:
            print(f"\n❌ Error loading batch: {e}")
            print("Tip: Check if your .npy files are empty or corrupted.")
