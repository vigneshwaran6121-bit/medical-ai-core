import os
import requests

# 1. NEW URL: Standard MNI152 Brain Atlas (reliable and small)
url = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/avg152T1_LR_nifti.nii.gz"

# 2. Save location
save_folder = "data/raw"
filename = "brain_mri.nii.gz"
save_path = os.path.join(save_folder, filename)

# 3. Create folder
os.makedirs(save_folder, exist_ok=True)

print(f"🚀 Starting download from: {url}")

# 4. Download
try:
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print(f"✅ Success! Brain MRI saved at: {save_path}")
    else:
        print(f"❌ Error: Server returned status code {response.status_code}")
except Exception as e:
    print(f"❌ Connection Error: {e}")
