import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av


from camera import detect_gender


# ===========================
# KONFIGURASI HALAMAN
# ===========================

st.set_page_config(
    page_title="AI Gender Detection",
    page_icon="🤖",
    layout="wide"
)


# ===========================
# LOAD CSS
# ===========================

def load_css():

    with open("style.css") as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )


load_css()

class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        img, results, total_faces = detect_gender(img)

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )



# ===========================
# SIDEBAR
# ===========================

with st.sidebar:

    st.image(
        "https://img.icons8.com/fluency/240/artificial-intelligence.png",
        width=120
    )

    st.markdown(
        "## 🤖 AI Gender Detection"
    )

    st.success(
        "🟢 Model Ready"
    )

    st.write("Model")
    st.info("CNN TensorFlow")

    st.write("Versi")
    st.info("1.0")

    st.write("Developer")
    st.info("Aji, Bagus, Fayed, Kayla")



# ===========================
# HEADER
# ===========================

st.markdown(
    """
    <div class="title">
        🤖 AI Gender Detection
    </div>

    <div class="subtitle">
        Deteksi Jenis Kelamin Menggunakan Convolutional Neural Network (CNN)
    </div>
    """,
    unsafe_allow_html=True
)


st.write(
    "Deteksi gender berdasarkan fitur wajah menggunakan Deep Learning."
)



# ==================================================
# UPLOAD GAMBAR
# ==================================================

st.subheader(
    "📂 Upload Gambar"
)


uploaded_file = st.file_uploader(
    "Masukkan foto wajah",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)



if uploaded_file:


    bytes_data = uploaded_file.read()


    img = cv2.imdecode(
        np.frombuffer(
            bytes_data,
            np.uint8
        ),
        cv2.IMREAD_COLOR
    )


    frame, results, total_faces = detect_gender(img)



    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    st.image(
        frame,
        width=600
    )


    st.metric(
        "Jumlah Wajah",
        total_faces
    )


    st.subheader(
        "Hasil Deteksi"
    )



    for i, data in enumerate(results):

        gender = data["gender"]

        confidence = data["confidence"]



        if gender == "Perempuan":

            st.markdown(
                f"""
                <div style="
                background:#e60026;
                color:white;
                padding:15px;
                border-radius:15px;
                margin-bottom:10px;">

                👩 Wajah {i+1}<br>
                Gender : <b>{gender}</b><br>
                Confidence : <b>{confidence:.2f}%</b>

                </div>
                """,
                unsafe_allow_html=True
            )


        else:

            st.markdown(
                f"""
                <div style="
                background:#0066ff;
                color:white;
                padding:15px;
                border-radius:15px;
                margin-bottom:10px;">

                👨 Wajah {i+1}<br>
                Gender : <b>{gender}</b><br>
                Confidence : <b>{confidence:.2f}%</b>

                </div>
                """,
                unsafe_allow_html=True
            )



st.divider()



# ==================================================
# KAMERA LIVE
# ==================================================

st.divider()

st.subheader("📷 Kamera Live")

st.info("Klik START lalu izinkan akses kamera.")

webrtc_streamer(
    key="gender-detection",
    video_processor_factory=VideoProcessor,
    rtc_configuration={
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    },
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)

