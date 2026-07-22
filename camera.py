import cv2
from utils.model import predict_gender


# ==========================================
# Load Haar Cascade
# ==========================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)



def detect_gender(frame):
    """
    Mendeteksi semua wajah pada frame,
    memprediksi gender setiap wajah,
    memberi warna kotak berdasarkan gender.
    """


    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80,80)
    )



    total_faces = len(faces)


    results = []



    for (x,y,w,h) in faces:


        face = frame[
            y:y+h,
            x:x+w
        ]


        if face.size == 0:
            continue



        # ==========================
        # Prediksi CNN
        # ==========================

        gender, confidence = predict_gender(face)



        results.append(
            {
                "gender": gender,
                "confidence": confidence
            }
        )



        # ==========================
        # Warna berdasarkan gender
        # ==========================

        if gender == "Perempuan":

            color = (
                0,
                0,
                255
            ) 
            # merah BGR


        else:

            color = (
                255,
                0,
                0
            )
            # biru BGR



        # ==========================
        # Kotak wajah
        # ==========================

        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            color,
            3
        )



        # ==========================
        # Background label
        # ==========================

        cv2.rectangle(
            frame,
            (x,y-40),
            (x+250,y),
            color,
            -1
        )



        # ==========================
        # Tulisan
        # ==========================

        cv2.putText(
            frame,
            f"{gender} {confidence:.1f}%",
            (x+5,y-12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )



    return (
        frame,
        results,
        total_faces
    )