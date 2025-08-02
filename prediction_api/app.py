from flask import Flask, request, jsonify, send_from_directory, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# --- Configuration ---
MODEL_PATH = os.path.join('model', 'unet_resnet50_cityscapes.tflite')
IMAGE_DIR = os.path.join('static', 'uploads')
# La taille d'entrée attendue par le modèle TFLite
INPUT_SHAPE = (256, 512)

# --- Palette de couleurs pour Cityscapes ---
# ✅ Palette ajoutée pour correspondre à votre code original
CITYSCAPES_COLOR_PALETTE = [
    (0, 0, 0),        # 0: arrière-plan / void
    (128, 64, 128),   # 1: route
    (244, 35, 232),   # 2: trottoir
    (70, 70, 70),     # 3: bâtiment
    (102, 102, 156),  # 4: mur
    (190, 153, 153),  # 5: clôture
    (153, 153, 153),  # 6: poteau
    (250, 170, 30),   # 7: panneau de signalisation
]

# --- Chargement de l'Interprète TFLite ---
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Modèle TFLite chargé avec succès.")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle TFLite : {e}")
    interpreter = None

# --- Fonctions Utilitaires ---
def preprocess_image(image_path):
    """Charge et pré-traite une image pour le modèle."""
    img = Image.open(image_path).convert('RGB').resize(INPUT_SHAPE[::-1]) # Inverser pour (largeur, hauteur)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def save_predicted_mask(prediction, output_path):
    """
    Sauvegarde le masque prédit en tant qu'image RGB colorée.
    """
    # ✅ Logique de coloration reprise de votre code original
    segmentation_map = np.argmax(prediction[0], axis=-1)
    h, w = segmentation_map.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Appliquer la couleur pour chaque classe
    for class_idx, color in enumerate(CITYSCAPES_COLOR_PALETTE):
        if class_idx < len(CITYSCAPES_COLOR_PALETTE):
            rgb_mask[segmentation_map == class_idx] = color

    mask_image = Image.fromarray(rgb_mask)
    mask_image.save(output_path)


# --- Routes de l'API ---
@app.route('/predict/<string:image_id>', methods=['GET'])
def predict(image_id):
    if not interpreter:
        return jsonify({'error': 'Modèle non disponible'}), 500

    image_filename = f"{image_id}.png"
    image_path = os.path.join(IMAGE_DIR, image_filename)

    if not os.path.exists(image_path):
        return jsonify({'error': 'Image non trouvée'}), 404

    try:
        processed_image = preprocess_image(image_path)
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        predicted_mask_filename = f"predicted_{image_id}.png"
        predicted_mask_path = os.path.join(IMAGE_DIR, predicted_mask_filename)

        # Utilise la nouvelle fonction pour sauvegarder le masque coloré
        save_predicted_mask(prediction, predicted_mask_path)

        mask_url = request.host_url.rstrip('/') + url_for('uploaded_file', filename=predicted_mask_filename)
        return jsonify({'predicted_mask_url': mask_url})

    except Exception as e:
        return jsonify({'error': f'Erreur lors de la prédiction : {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Sert les fichiers statiques (images et masques)."""
    return send_from_directory(IMAGE_DIR, filename)

if __name__ == '__main__':
    # Utilise un port différent de l'app de présentation
    app.run(debug=True, port=5001)