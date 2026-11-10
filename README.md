# Streamlit - Klasifikasi Histopatologi Kanker Kolorektal

Aplikasi ini menjalankan model ResNet-50 dari skenario **Oversampling (Augmentasi Data), Fold 3** untuk mengklasifikasikan citra histopatologi ke dalam Grade I, Grade II, dan Grade III, serta menampilkan visualisasi Grad-CAM.

## 1. Struktur folder

```text
streamlit_crc_oversampling/
├── app.py
├── requirements.txt
├── README.md
└── models/
    └── resnet50_oversampling_aug_30epoch_fold_3.pth
```

**Checkpoint model tidak disertakan** dalam paket ini. Salin file hasil training Fold 3 ke folder `models/` dengan nama persis seperti di atas.

## 2. Menjalankan secara lokal

Disarankan menggunakan Python 3.10 atau 3.11.

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

Install dependensi:
```bash
pip install -r requirements.txt
```

Jalankan:
```bash
streamlit run app.py
```

## 3. Upload ke GitHub

1. Buat repository GitHub baru.
2. Upload `app.py`, `requirements.txt`, `README.md`, dan folder `models/` beserta checkpoint.
3. Pastikan file model tidak melebihi batas ukuran GitHub. File besar perlu disimpan menggunakan Git LFS atau penyimpanan model eksternal yang dapat diakses aplikasi.

## 4. Hosting dengan Streamlit Community Cloud

1. Buka https://share.streamlit.io/
2. Login menggunakan akun GitHub.
3. Pilih **Create app**.
4. Pilih repository, branch, dan file utama `app.py`.
5. Klik **Deploy**.

## 5. Catatan teknis penting

- Model dibangun sebagai `torchvision.models.resnet50` dengan fully connected layer untuk tiga kelas.
- Urutan label: indeks 0 = Grade I, indeks 1 = Grade II, indeks 2 = Grade III.
- Preprocessing: resize 224 × 224, tensor, normalisasi ImageNet.
- Grad-CAM menggunakan `model.layer4[-1]`.
- Aplikasi memuat satu checkpoint Fold 3, bukan ensemble lima fold.
- Bila pemuatan model gagal dengan pesan `Missing key(s)` atau `Unexpected key(s)`, struktur checkpoint atau definisi arsitektur saat training harus disesuaikan dengan `app.py`.
- Hasil aplikasi merupakan keluaran model penelitian, bukan diagnosis medis.

## 6. Privasi data

Jangan mengunggah citra pasien yang mengandung identitas pribadi ke aplikasi publik. Gunakan citra penelitian yang telah dianonimkan dan pastikan Anda memiliki izin penggunaan serta publikasinya.
