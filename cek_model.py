import tensorflow as tf

model = tf.keras.models.load_model("gender_model.keras")

print("Model berhasil dibuka!")

model.summary()