import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os

# Anzahl der Pflanzenarten (Klassen) im Datensatz
num_classes = 5  # Passen Sie diesen Wert an die Anzahl der Pflanzenarten an

# Laden des vortrainierten MobileNetV2-Modells ohne Klassifikationskopf
base_model = MobileNetV2(weights='imagenet', include_top=False)

# Neuen Klassifikationskopf hinzufügen
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Nur den neuen Klassifikationskopf trainieren
for layer in base_model.layers:
    layer.trainable = False

# Kompilieren des Modells
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Daten vorbereiten (Pfad zu deinem Pflanzen-Datensatz anpassen)
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'path_to_plant_images/train',  # Pfad zum Trainingsdatensatz
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Modell trainieren
model.fit(train_generator, epochs=10)

# Speicherpfad für das Modell
model_save_path = "models/plant_classifier/1"

# Speicherverzeichnis erstellen, falls es nicht existiert
os.makedirs(model_save_path, exist_ok=True)

# Modell im SavedModel-Format speichern
model.save(model_save_path)
print(f"Modell wurde erfolgreich im Ordner '{model_save_path}' gespeichert.")
