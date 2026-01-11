import pydicom
import os
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)


# Get the directory where the script is currently running
current_dir = os.path.dirname(os.path.abspath(__file__))
dicom_path = os.path.join(current_dir, "ID_0000aee4b.dcm")
ds = pydicom.dcmread(dicom_path)

print("Patient ID:", ds.PatientID)
print("Study UID:", ds.StudyInstanceUID)
print("Series UID:", ds.SeriesInstanceUID)
print("Modality:", ds.Modality)
# print("Slice Thickness:", ds.SliceThickness)
print("Pixel Spacing:", ds.PixelSpacing)
print("Image Position:", ds.ImagePositionPatient)
print("Image Orientation:", ds.ImageOrientationPatient)
