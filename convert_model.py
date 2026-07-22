import tensorflow as tf

print("Loading model...")

model = tf.keras.models.load_model(
    "gender_model.keras",
    compile=False
)

print("Saving model...")

model.save(
    "gender_model.h5"
)

print("Selesai!")