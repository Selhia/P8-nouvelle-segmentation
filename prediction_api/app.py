from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# --- Configuration ---
# ✅ Nom du modèle mis à jour
MODEL_PATH = os.path.join('model', 'unet_resnet50_cityscapes.tflite') 
IMAGE_DIR = os.path.join('static', 'uploads')
IMAGE_SIZE = (256, 256) # ❗️ Adaptez si la taille attendue par ce modèle est différente

# --- Chargement de l'Interprète TFLite ---
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors() # Alloue la mémoire pour le modèle
    
    # Récupère les détails des tenseurs d'entrée et de sortie
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("Modèle TFLite chargé et initialisé avec succès.")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle TFLite : {e}")
    interpreter = None

# --- Fonctions Utilitaires ---
def preprocess_image(image_path):
    """Charge et pré-traite une image pour le modèle."""
    img = Image.open(image_path).convert('RGB').resize(IMAGE_SIZE) # .convert('RGB') pour forcer 3 canaux
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def save_predicted_mask(mask_array, output_path):
    """Sauvegarde le masque prédit en tant qu'image."""
    mask = np.squeeze(mask_array)
    mask = (mask * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask)
    mask_img.save(output_path)

# --- Routes de l'API ---
@app.route('/predict/<string:image_id>', methods=['GET'])
def predict(image_id):
    if not interpreter:
        return jsonify({'error': 'Modèle non disponible'}), 500

    image_filename = f"{image_id}.jpg" 
    image_path = os.path.join(IMAGE_DIR, image_filename)

    if not os.path.exists(image_path):
        return jsonify({'error': 'Image non trouvée'}), 404

    try:
        processed_image = preprocess_image(image_path)
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        predicted_mask_array = interpreter.get_tensor(output_details[0]['index'])

        predicted_mask_filename = f"predicted_{image_id}.png"
        predicted_mask_path = os.path.join(IMAGE_DIR, predicted_mask_filename)
        save_predicted_mask(predicted_mask_array, predicted_mask_path)

        mask_url = request.host_url + f"uploads/{predicted_mask_filename}"
        return jsonify({'predicted_mask_url': mask_url})

    except Exception as e:
        return jsonify({'error': f'Erreur lors de la prédiction : {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Sert les fichiers statiques (images et masques)."""
    return send_from_directory(IMAGE_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5001)