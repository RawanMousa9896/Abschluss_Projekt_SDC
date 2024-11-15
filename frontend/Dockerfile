# Verwende Python-Image
FROM python:3.9

# Arbeitsverzeichnis
WORKDIR /app

# Anforderungen installieren
COPY requirements.txt .
RUN pip install -r requirements.txt

# Kopiere das Streamlit-Skript
COPY app.py .

# Port für Streamlit
EXPOSE 8501

# Starte Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
