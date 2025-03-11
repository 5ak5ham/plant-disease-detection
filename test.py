import os

file_path = "C:/Users/SAKSHAM/Downloads/CAPSTONE/trained_model.keras"
file_size = os.path.getsize(file_path)
print(f"File size: {file_size} bytes")

# Check if it's a zip file (Keras files should be zip archives)
import zipfile
try:
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        print("Contents of the zip file:")
        for file in zip_ref.namelist():
            print(f" - {file}")
except zipfile.BadZipFile:
    print("Not a valid zip file - this is not a proper .keras file")
except Exception as e:
    print(f"Error: {e}")