import requests
import os

url = "https://github.com/Selhia/image-segmentation-app/raw/main/image-segmentation-app/model/unet_resnet50_cityscapes.tflite"
output_path = "prediction_api/model/unet_resnet50_cityscapes.tflite"

# Crée le dossier model s'il n'existe pas
os.makedirs(os.path.dirname(output_path), exist_ok=True)

response = requests.get(url)
with open(output_path, "wb") as f:
    f.write(response.content)