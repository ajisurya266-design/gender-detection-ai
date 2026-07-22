import tensorflow as tf

model = tf.keras.models.load_model(
    "gender_model.keras",
    compile=False
)

model.save("gender_model_fixed.keras")

print("Berhasil convert")