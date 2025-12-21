import nibabel as nib
import matplotlib.pyplot as plt
import os

# load the NIFTI file
current_dir = os.getcwd()
file_path = os.path.join(current_dir, "src", "day5_task", "day5_nii file", "mni152.nii")
img = nib.load(file_path)
data = img.get_fdata()

print(f"Image shape: {data.shape}")

# find the center slices
x_center = data.shape[0] // 2
y_center = data.shape[1] // 2
z_center = data.shape[2] // 2

# plotting
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# sagital
ax1.imshow(data[x_center, :, :].T, cmap="gray", origin="lower")
ax1.set_title("sagital")

# coronal
ax2.imshow(data[:, y_center, :].T, cmap="gray", origin="lower")
ax2.set_title("coronal")

# axial
ax3.imshow(data[:, :, z_center].T, cmap="gray", origin="lower")
ax3.set_title("axial")

plt.show()
