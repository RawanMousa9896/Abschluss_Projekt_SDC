import tensorflow as tf

# Laden des vortrainierten MobileNetV2-Modells
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Speichern des Modells im SavedModel-Format
model.save("models/animal_classifier/1")  # Speichert die erste Version des Modells
