import tensorflow as tf
import numpy as np
import cv2
import os

# ==========================
# Load Model
# ==========================
model = tf.keras.models.load_model("gender_model.keras")

# ==========================
# Gambar yang ingin dites
# ==========================
image_path = "dataset/test/male/1_0_0_20161219190621290.jpg.chip.jpg"   # Ganti dengan nama gambar

if not os.path.exists(image_path):
    print("File tidak ditemukan!")
    exit()

# ==========================
# Baca gambar
# ==========================
img = cv2.imread(image_path)

img = cv2.resize(img, (128,128))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)

# ==========================
# Prediksi
# ==========================
prediction = model.predict(img, verbose=0)[0][0]

if prediction >= 0.5:
    gender = "Male"
    confidence = prediction * 100
else:
    gender = "Female"
    confidence = (1 - prediction) * 100

print("="*40)
print("Hasil Prediksi")
print("="*40)
print("Jenis Kelamin :", gender)
print("Confidence     : {:.2f}%".format(confidence))
print("="*40)