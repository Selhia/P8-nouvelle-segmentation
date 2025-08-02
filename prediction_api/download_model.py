import requests

url = "https://github.com/Selhia/image-segmentation-app/raw/main/image-segmentation-app/model/unet_resnet50_cityscapes.tflite"
output_path = "prediction_api/model/unet_resnet50_cityscapes.tflite"

response = requests.get(url)
with open(output_path, "wb") as f:
    f.write(response.content)