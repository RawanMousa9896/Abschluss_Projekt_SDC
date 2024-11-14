import tensorflow as tf

# Laden des vortrainierten MobileNetV2-Modells mit "ImageNet"-Gewichten
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Speichern des Modells im SavedModel-Format
tf.saved_model.save(model, "models/plant_classifier/1")  # Speichert die erste Version des Modells im SavedModel-Format

