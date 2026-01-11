import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class MedicalVolumeDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith(".nii.gz")]
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 1. Load the file
        file_name = self.files[idx]
        file_path = os.path.join(self.data_dir, file_name)

        # Load NIfTI and convert to Numpy array
        img_obj = nib.load(file_path)
        img_data = img_obj.get_fdata().astype(np.float32)  # Ensure it's float, not int

        # 2. Normalization (Min-Max Scaling)
        # We want values between 0 and 1 to make training stable
        min_val = np.min(img_data)
        max_val = np.max(img_data)

        if max_val > 0:  # Avoid division by zero
            img_data = (img_data - min_val) / (max_val - min_val)

        # 3. Convert to PyTorch Tensor
        tensor = torch.from_numpy(img_data)

        # 4. Add Channel Dimension
        # Current shape: (Height, Width, Depth) -> e.g., (192, 224, 176)
        # Target shape: (Channels, Height, Width, Depth) -> (1, 192, 224, 176)
        # We use unsqueeze(0) to add that "1" at the start.
        tensor = tensor.unsqueeze(0)

        return tensor, file_name


# --- Self-Test Block ---
if __name__ == "__main__":
    test_dir = "../data/raw"

    try:
        dataset = MedicalVolumeDataset(test_dir)
        print(f"✅ Dataset initialized with {len(dataset)} files.")

        # Let's get the first item
        first_scan, filename = dataset[0]

        print(f"\n🔬 File Loaded: {filename}")
        print(f"   Tensor Shape: {first_scan.shape}")
        print(f"   Data Type: {first_scan.dtype}")
        print(f"   Max Value (should be 1.0): {first_scan.max()}")
        print(f"   Min Value (should be 0.0): {first_scan.min()}")

        # Check if shape is correct (Should start with 1, e.g., [1, ...])
        if first_scan.shape[0] == 1:
            print("✅ Channel dimension added correctly.")
        else:
            print("❌ Error: Missing channel dimension!")

    except Exception as e:
        print(f"❌ Error: {e}")
