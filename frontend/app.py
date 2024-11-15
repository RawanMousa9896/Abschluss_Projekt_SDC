import streamlit as st
import requests
import base64
import os

# PlantNet API-Schlüssel hier einfügen
API_KEY = 'BuWptEWokk2rDa04HvCSLyWupnE5hiRYV5QFuajtVaNWMi2YkC'

# PlantNet API-Schlüssel aus Umgebungsvariable laden
API_KEY = os.getenv('API_KEY')
API_URL = 'https://my-api.plantnet.org/v2/identify/all'

# Funktion zur Bildkodierung in Base64
def encode_image(image):
    return base64.b64encode(image.read()).decode('utf-8')

# Funktion zur Pflanzenidentifikation über PlantNet API
def identify_plant(encoded_image, organ='leaf'):
    headers = {'Content-Type': 'application/json'}
    payload = {
        'images': [encoded_image],
        'organs': [organ],
        'include-related-images': False,
        'lang': 'en'
    }
    params = {'api-key': API_KEY}
    response = requests.post(API_URL, headers=headers, json=payload, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Fehler bei der API-Anfrage: {response.status_code}")
        return None

# Streamlit-Frontend
st.title("Plant Classifier with PlantNet API")
st.write("Lade ein Bild einer Pflanze hoch, um ihre Art zu bestimmen.")

# Bild hochladen
uploaded_file = st.file_uploader("Wähle ein Bild aus...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Hochgeladenes Bild', use_column_width=True)

    organ = st.selectbox("Wähle das Organ der Pflanze aus:", ["leaf", "flower", "fruit", "bark", "habit"])
    
    st.write("Klassifikation läuft...")
    encoded_image = encode_image(uploaded_file)
    data = identify_plant(encoded_image, organ=organ)

    if data:
        st.write("Ergebnisse der Pflanzenklassifikation:")
        for suggestion in data['results']:
            st.write(f"Name: {suggestion['species']['scientificNameWithoutAuthor']}, Wahrscheinlichkeit: {suggestion['score'] * 100:.2f}%")
            if 'images' in suggestion:
                st.image(suggestion['images'][0]['url']['m'], caption="Ähnliches Bild", use_column_width=True)
