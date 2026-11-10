import io
from pathlib import Path

import hashlib
import json
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
import matplotlib.cm as cm

from huggingface_hub import hf_hub_download

# ============================================================
# Konfigurasi aplikasi
# ============================================================
st.set_page_config(
    page_title="Klasifikasi Histopatologi Kanker Kolorektal",
    page_icon="🔬",
    layout="wide"
)

MODEL_PATH = Path("models/resnet50_oversampling_aug_30epoch_fold_3.pth")
CLASS_NAMES = ["Grade I", "Grade II", "Grade III"]
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

HASH_MANIFEST_PATH = Path("dataset_hashes.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ============================================================
# Validasi SHA-256 gambar dataset
# ============================================================

@st.cache_data
def load_dataset_hashes():
    if not HASH_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "File dataset_hashes.json tidak ditemukan. "
            "Jalankan generate_hashes.py terlebih dahulu."
        )

    with open(HASH_MANIFEST_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data.get("hashes", {})


def calculate_uploaded_sha256(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()


def validate_dataset_image(file_bytes):
    hashes = load_dataset_hashes()

    uploaded_hash = calculate_uploaded_sha256(file_bytes)

    matched_file = hashes.get(uploaded_hash)

    if matched_file is None:
        return False, None

    return True, matched_file



# ============================================================
# Memuat model dari Hugging Face
# ============================================================
MODEL_REPO_ID = "Miikaelathen/resnet50-crc-oversampling"
MODEL_FILENAME = "resnet50_oversampling_aug_30epoch_fold_3.pth"

@st.cache_resource
def load_model():

    # Mengunduh checkpoint dari Hugging Face
    model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME,
        repo_type="model"
    )

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    # Memuat checkpoint
    try:
        checkpoint = torch.load(
            model_path, map_location=DEVICE, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(model_path, map_location=DEVICE)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Menghapus prefix DataParallel jika ada
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "", 1) if key.startswith("module.") else key
        cleaned_state_dict[new_key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# Grad-CAM untuk ResNet-50 pada layer4[-1]
# ============================================================
def generate_gradcam(model, input_tensor, target_class):
    activations = []
    gradients = []
    target_layer = model.layer4[-1]

    def forward_hook(module, inputs, output):
        activations.append(output)
        # Simpan gradien langsung dari tensor output untuk menghindari
        # konflik full_backward_hook dengan operasi inplace di ResNet.
        output.register_hook(lambda grad: gradients.append(grad))

    forward_handle = target_layer.register_forward_hook(forward_hook)

    try:
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor)
        score = logits[:, target_class].sum()
        score.backward()

        feature_maps = activations[0]
        grads = gradients[0]

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * feature_maps).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=IMAGE_SIZE,
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().detach().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam
    finally:
        forward_handle.remove()


def create_gradcam_overlay(original_image, cam):
    original = np.asarray(original_image.resize(IMAGE_SIZE)).astype(np.float32) / 255.0
    heatmap = cm.jet(cam)[..., :3].astype(np.float32)
    overlay = 0.45 * heatmap + 0.55 * original
    overlay = np.clip(overlay, 0, 1)
    return (heatmap * 255).astype(np.uint8), (overlay * 255).astype(np.uint8)


# ============================================================
# Antarmuka Streamlit
# ============================================================
st.title("🔬 Klasifikasi Tingkat Kanker Kolorektal Berdasarkan Citra Histopatologi")
st.write(
    "Klasifikasi citra histopatologi menggunakan **ResNet-50** "
    "dengan skenario **Oversampling (Augmentasi Data)**."
)

st.info(
    "Aplikasi ini merupakan prototipe penelitian untuk klasifikasi citra. "
    "Hasil prediksi dan Grad-CAM bukan diagnosis medis dan tidak menggantikan "
    "penilaian dokter atau ahli patologi."
)

with st.sidebar:
    st.header("Informasi Model")
    st.write("**Arsitektur:** ResNet-50")
    st.write("**Skenario:** Oversampling (Augmentasi Data)")
    st.write("**Model:** Fold 3")
    st.write("**Ukuran input:** 224 × 224 piksel")
    st.write("**Target Grad-CAM:** `layer4[-1]`")
    st.divider()
    st.caption("Model menggunakan checkpoint hasil pelatihan penelitian.")


# ============================================================
# Pilihan input: Sampel Dataset atau Upload Manual
# ============================================================

SAMPLES = {
    "Grade I - Sampel 1": Path("samples/01_CRC_G1_16130_4x_1.jpg"),
    "Grade II - Sampel 1": Path("samples/01_CRC_G2_15895_4x_1.jpg"),
    "Grade III - Sampel 1": Path("samples/01_CRC_G3_15894_4x_1.jpg"),
}


# ============================================================
# Pemilihan metode input
# ============================================================

st.subheader("Pilih Metode Input Citra")

input_method = st.radio(
    "Metode input:",
    ["Sampel Dataset", "Unggah Gambar Sendiri"],
    horizontal=True
)

input_bytes = None
image = None

if input_method == "Sampel Dataset":

    st.write("### Pilih sampel berdasarkan kelas")

    grade_options = {
        "Grade I": "grade_1",
        "Grade II": "grade_2",
        "Grade III": "grade_3"
    }

    selected_grade = st.selectbox(
        "Pilih kelas:",
        list(grade_options.keys())
    )

    grade_folder = grade_options[selected_grade]
    sample_dir = Path("samples") / grade_folder

    sample_options = [
        f"Sampel {i}" for i in range(1, 5)
    ]

    selected_sample = st.selectbox(
        "Pilih sampel:",
        sample_options
    )

    sample_number = int(selected_sample.split()[-1])

    sample_path = sample_dir / f"sample_{sample_number}.jpg"

    if sample_path.exists():
        input_bytes = sample_path.read_bytes()
        image = Image.open(io.BytesIO(input_bytes)).convert("RGB")

        st.caption(
            f"Input: {selected_grade} - {selected_sample}"
        )
    else:
        st.error(
            f"File sampel tidak ditemukan: {sample_path}"
        )

elif input_method == "Unggah Gambar Sendiri":

    uploaded_file = st.file_uploader(
        "Unggah citra histopatologi",
        type=["jpg", "jpeg", "png"],
        help="Gunakan gambar asli dari dataset CRC-HGD-v1."
    )

    if uploaded_file is not None:
        input_bytes = uploaded_file.getvalue()

        try:
            image = Image.open(
                io.BytesIO(input_bytes)
            ).convert("RGB")
        except Exception:
            st.error("File tidak dapat dibaca sebagai citra.")
            input_bytes = None
            image = None


# ============================================================
# Menampilkan citra input
# ============================================================

if image is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Citra Input")
        st.image(image, width="stretch")

    if st.button(
        "Klasifikasikan Citra",
        type="primary",
        width="stretch"
    ):
        try:
            # Validasi SHA-256 tetap berlaku untuk kedua metode input
            is_valid, matched_file = validate_dataset_image(
                input_bytes
            )

            if not is_valid:
                st.error(
                    "Gambar ditolak! Citra tidak ditemukan "
                    "dalam dataset CRC-HGD-v1."
                )
                st.stop()

            st.success(
                f"Validasi dataset berhasil: {matched_file}"
            )

            # Lanjutkan dengan kode klasifikasi ResNet-50
            # dan Grad-CAM yang sudah ada.

            model = load_model()
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)

            with torch.enable_grad():
                logits = model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)[0]

                predicted_index = int(
                    torch.argmax(probabilities).item()
                )

                confidence = float(
                    probabilities[predicted_index].item()
                )

                cam = generate_gradcam(
                    model, input_tensor, predicted_index
                )

            heatmap, overlay = create_gradcam_overlay(
                image, cam
            )

            with col2:
                st.subheader("Hasil Klasifikasi")
                st.success(
                    f"Prediksi: **{CLASS_NAMES[predicted_index]}**"
                )

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

                st.write("**Probabilitas setiap kelas**")

                for idx, class_name in enumerate(CLASS_NAMES):
                    prob = float(probabilities[idx].item())
                    st.write(
                        f"{class_name}: {prob * 100:.2f}%"
                    )
                    st.progress(prob)

            st.divider()
            st.subheader("Visualisasi Grad-CAM")

            g1, g2 = st.columns(2)

            with g1:
                st.write("**Heatmap**")
                st.image(heatmap, width="stretch")

            with g2:
                st.write("**Overlay**")
                st.image(overlay, width="stretch")

            st.caption(
                "Grad-CAM menggambarkan area citra yang berkontribusi "
                "terhadap prediksi model, bukan bukti diagnosis."
            )

        except FileNotFoundError as e:
            st.error(str(e))

        except RuntimeError as e:
            st.error(f"Kesalahan model: {e}")

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

else:
    st.warning(
        "Silakan pilih sampel dataset atau unggah gambar terlebih dahulu."
    )


st.divider()
st.caption(
    "Penelitian: Klasifikasi Tingkat Kanker Kolorektal "
    "Berdasarkan Citra Histopatologi Menggunakan ResNet-50 dan Grad-CAM."
)
