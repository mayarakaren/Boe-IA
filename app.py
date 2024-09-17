import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)

# Função para processar e fazer a predição da imagem
def predict_image(model, img_path):
    img = load_img(img_path, target_size=(64, 64))  # Tamanho conforme o modelo treinado
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expandir a dimensão para o modelo
    img_array /= 255.0  # Normalizar
    
    # Fazer a predição
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction[0])
    classes = ['Dermatite Nodular', 'Berne', 'Saudável']
    predicted_class = classes[predicted_class_index]
    accuracy = float(np.max(prediction[0]))  # Probabilidade associada à classe prevista

    return predicted_class, accuracy

# Rota para predição
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        file_path = "temp_image.jpg"
        file.save(file_path)
        predicted_class, accuracy = predict_image(model, file_path)
        os.remove(file_path)  # Remover o arquivo temporário após a predição
        
        return jsonify({
            "predicted_class": predicted_class,
            "accuracy": accuracy
        }), 200

if __name__ == '__main__':
    model_path = 'bovino_classification_model.h5'
    if not os.path.exists(model_path):
        print("Modelo não encontrado. Treine o modelo primeiro.")
    else:
        model = load_model(model_path)
        app.run(debug=True)
