import cv2
import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model("gender_model.keras")

# Ganti nama gambar nanti
img = cv2.imread("test.png")

# Resize
img = cv2.resize(img, (128,128))

# BGR ke RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Normalisasi
img = img / 255.0

# Tambah dimensi
img = np.expand_dims(img, axis=0)

# Prediksi
hasil = model.predict(img, verbose=0)[0][0]

print("Nilai Prediksi :", hasil)

if hasil >= 0.5:
    print("Prediksi : Laki-laki")
else:
    print("Prediksi : Perempuan")