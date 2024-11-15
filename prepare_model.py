import os
import tarfile
import urllib.request
import scipy.io
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Schritt 1: Datensatz herunterladen und entpacken
data_dir = 'oxford_flowers'
os.makedirs(data_dir, exist_ok=True)

# Download-URLs
images_url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
labels_url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat'

# Herunterladen der Bilder
print("Lade Bilder herunter...")
urllib.request.urlretrieve(images_url, os.path.join(data_dir, '102flowers.tgz'))

# Herunterladen der Labels
print("Lade Labels herunter...")
urllib.request.urlretrieve(labels_url, os.path.join(data_dir, 'imagelabels.mat'))

# Entpacke die Bilder
print("Entpacke Bilder...")
with tarfile.open(os.path.join(data_dir, '102flowers.tgz'), 'r:gz') as tar:
    tar.extractall(path=data_dir)

print("Download und Entpacken abgeschlossen.")

# Schritt 2: Daten vorbereiten und Labelstruktur erstellen
images_folder = os.path.join(data_dir, 'jpg')
labels_path = os.path.join(data_dir, 'imagelabels.mat')
train_dir = os.path.join(data_dir, 'train')
os.makedirs(train_dir, exist_ok=True)

# Labels laden
print("Ordne Bilder in Klassenordner...")
labels = scipy.io.loadmat(labels_path)['labels'][0]

# Anzahl der Bilder pro Klasse, die verwendet werden sollen
num_images_per_class = 5
class_image_count = {}

# Erstelle Klassenordner und verschiebe Bilder
for idx, label in enumerate(labels):
    class_folder = os.path.join(train_dir, f'class_{label}')
    os.makedirs(class_folder, exist_ok=True)
    
    # Überprüfe, ob das Limit an Bildern pro Klasse erreicht wurde
    if class_image_count.get(label, 0) >= num_images_per_class:
        continue
    
    # Verschiebe das Bild in den entsprechenden Klassenordner
    img_name = f'image_{idx + 1:05d}.jpg'
    shutil.move(os.path.join(images_folder, img_name), os.path.join(class_folder, img_name))
    
    # Aktualisiere den Zähler für die aktuelle Klasse
    class_image_count[label] = class_image_count.get(label, 0) + 1

print("Bilder sind sortiert und bereit für das Training.")

# Schritt 3: Modell trainieren und speichern
num_classes = len(class_image_count)  # Anzahl der tatsächlich verwendeten Klassen

# MobileNetV2-Modell ohne Klassifikationskopf laden
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

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

# Daten vorbereiten für das Training
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Modell trainieren
model.fit(train_generator, validation_data=validation_generator, epochs=10)

# Speichern des Modells
model_save_path = "models/oxford_flowers_classifier/1"
os.makedirs(model_save_path, exist_ok=True)
model.save(model_save_path)
print(f"Modell wurde erfolgreich im Ordner '{model_save_path}' gespeichert.")
