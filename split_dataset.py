import os
import shutil
import random

# ==========================
# Lokasi Dataset
# ==========================
SOURCE_DIR = "dataset/raw"

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/validation"
TEST_DIR = "dataset/test"

# ==========================
# Buat Folder
# ==========================
for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    os.makedirs(os.path.join(folder, "male"), exist_ok=True)
    os.makedirs(os.path.join(folder, "female"), exist_ok=True)

# ==========================
# Ambil Semua File Gambar
# ==========================
images = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".jpg")]

random.shuffle(images)

# ==========================
# Kelompokkan Berdasarkan Gender
# ==========================
male = []
female = []

for img in images:
    try:
        gender = img.split("_")[1]

        if gender == "0":
            male.append(img)
        elif gender == "1":
            female.append(img)

    except:
        continue


# ==========================
# Fungsi Membagi Data
# ==========================
def split_data(data, label):

    total = len(data)

    train = int(total * 0.8)
    val = int(total * 0.1)

    train_data = data[:train]
    val_data = data[train:train + val]
    test_data = data[train + val:]

    for file in train_data:
        shutil.copy(
            os.path.join(SOURCE_DIR, file),
            os.path.join(TRAIN_DIR, label, file)
        )

    for file in val_data:
        shutil.copy(
            os.path.join(SOURCE_DIR, file),
            os.path.join(VAL_DIR, label, file)
        )

    for file in test_data:
        shutil.copy(
            os.path.join(SOURCE_DIR, file),
            os.path.join(TEST_DIR, label, file)
        )


# ==========================
# Jalankan
# ==========================
split_data(male, "male")
split_data(female, "female")

print("===================================")
print("Dataset berhasil dipisahkan!")
print(f"Male   : {len(male)} gambar")
print(f"Female : {len(female)} gambar")
print("===================================")