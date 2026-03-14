from PIL import Image
import os

input_folder = r"C:\Users\Naman V Shetty\Documents\WizWorks\###SIDDHI\AbDo-CTRG\AbDo-CT Dataset (Sample)\Patient 30"
output_folder = r"C:\Users\Naman V Shetty\Documents\WizWorks\###SIDDHI\AbDo-CTRG\AbDo-CT Dataset (Sample)\Patient_30"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith(".jpg") or file.endswith(".jpeg"):
        img = Image.open(os.path.join(input_folder, file))
        
        png_name = file.split(".")[0] + ".png"
        img.save(os.path.join(output_folder, png_name), "PNG")

print("Conversion complete!")