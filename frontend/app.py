import streamlit as st
import requests
import numpy as np
from PIL import Image
import json

# Titel und Beschreibung der Anwendung
st.title("Plant Classifier with MobileNetV2")
st.write("Upload an image, and the model will classify the plant!")

# Bild hochladen
uploaded_file = st.file_uploader("Choose a plant image...", type="jpg")

if uploaded_file is not None:
    # Bild anzeigen
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")

    # Bild für das Modell vorbereiten
    image = image.resize((224, 224))  # MobileNetV2 erwartet 224x224 Pixel
    img_array = np.array(image) / 255.0  # Normalisieren auf [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Batch-Dimension hinzufügen
    data = json.dumps({"instances": img_array.tolist()})

    # An den Modellserver senden (API-Adresse anpassen, falls erforderlich)
    response = requests.post('http://model-server:8501/v1/models/plant_classifier:predict', data=data)
    prediction = np.array(response.json()['predictions'][0])

    # Ausgabe des Modells dekodieren
    labels = requests.get("https://path-to-your-plant-labels/plant_class_index.json").json()
    decoded_label = labels[str(np.argmax(prediction))][1]
    st.write(f"Prediction: {decoded_label}")
