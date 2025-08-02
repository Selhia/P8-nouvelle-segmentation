from flask import Flask, render_template, request
import requests
import os

app = Flask(__name__)

# --- Configuration ---
# ❗️ Une fois l'API déployée, vous devrez changer cette URL !
API_BASE_URL = "http://127.0.0.1:5001" 
IMAGE_DIR = 'static/images/real' 
MASK_DIR = 'static/images/masks'

def get_available_image_ids():
    """Retourne une liste triée des ID d'images disponibles."""
    if not os.path.exists(IMAGE_DIR):
        return []
    ids = [os.path.splitext(f)[0] for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
    return sorted(ids)

@app.route('/', methods=['GET', 'POST'])
def index():
    image_ids = get_available_image_ids()
    selected_id = None
    real_image_url = None
    real_mask_url = None
    predicted_mask_url = None
    error = None

    if not image_ids:
        error = "Aucune image trouvée dans le dossier static/images/real. Veuillez ajouter des images."
    else:
        # Déterminer l'ID sélectionné
        selected_id = request.form.get('image_id') if request.method == 'POST' else image_ids[0]
        
        # Définir les URLs des images locales
        real_image_url = f"/static/images/real/{selected_id}.jpg"
        real_mask_url = f"/static/images/masks/{selected_id}.png"

        # Si le formulaire est soumis, appeler l'API
        if request.method == 'POST':
            try:
                # Extrait le numéro de l'ID (ex: "image_1" -> "1")
                image_number_id = selected_id.split('_')[-1]
                
                api_response = requests.get(f"{API_BASE_URL}/predict/{image_number_id}")
                api_response.raise_for_status() 
                
                data = api_response.json()
                predicted_mask_url = data.get('predicted_mask_url')
                if not predicted_mask_url:
                    error = data.get('error', 'Réponse invalide de l\'API.')

            except requests.exceptions.RequestException as e:
                error = f"Erreur de communication avec l'API : {e}. L'API est-elle bien lancée sur {API_BASE_URL} ?"
            except Exception as e:
                error = f"Une erreur inattendue est survenue : {e}"

    return render_template('index.html',
                           image_ids=image_ids,
                           selected_id=selected_id,
                           real_image_url=real_image_url,
                           real_mask_url=real_mask_url,
                           predicted_mask_url=predicted_mask_url,
                           error=error)

if __name__ == '__main__':
    app.run(debug=True, port=8080)