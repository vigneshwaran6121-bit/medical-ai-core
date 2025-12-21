import pydicom
from pydicom.data import get_testdata_files

# load a sample DICOM file provided by the library (.dcm file)
filename = get_testdata_files("CT_small.dcm")[0]

# read the file
ds = pydicom.dcmread(filename)

# print specific metadata (" the header")
print(" --- DICOM HEADER INFO ---")
print(f"Patient ID: {ds.PatientID}")
print(f"Modality:{ds.Modality}")
print(f"Study Date: {ds.StudyDate}")

# access the pixel data(the image)
# it converts raw bytes into a workable numpy array
image_data = ds.pixel_array

print("\n--- IMAGE DATA ---")
print(f"Image shape:{image_data.shape}")
print(f"Data Type: {image_data.dtype}")
