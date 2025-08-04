# 🧠 Brain Tumor Object Detection

Proyek ini adalah implementasi aplikasi web deep learning untuk mendeteksi dan mengklasifikasikan tumor otak pada citra atau video MRI menggunakan model object detection YOLOv8. Aplikasi ini menampilkan bounding box pada area tumor, mendukung input gambar, video, serta analisis webcam secara real-time.

## 📂 Project Structure

- `samples/` — Berisi contoh gambar dan video MRI dari masing-masing kelas.
- `.gitignore` — File untuk mengabaikan folder atau file tertentu saat push ke Git.
- `Brain Tumor (Baseline).ipynb` — Notebook baseline training model deteksi.
- `Brain Tumor (Fine Tune).ipynb` — Notebook fine-tuning YOLOv8 pada dataset tumor otak.
- `app.py` — Aplikasi utama Streamlit untuk deployment.
- `best.pt` — File model YOLOv8 hasil pelatihan akhir.
- `packages.txt` — daftar sistem dependensi untuk fitur webcam.
- `requirements.txt` — Daftar dependensi Python yang diperlukan untuk menjalankan proyek.
- `restructuring.py` — Python script untuk merapikan data sebelum proses training.

## 🚀 Cara Run Aplikasi

### 🔹 1. Jalankan Secara Lokal
### Clone Repository
```bash
git clone https://github.com/RichardDeanTan/Brain-Tumor-Object-Detection
cd Brain-Tumor-Object-Detection
```
### Install Dependensi
```bash
pip install -r requirements.txt
```
### Jalankan Aplikasi Streamlit
```bash
streamlit run app.py
```

### 🔹 2. Jalankan Secara Online (Tidak Perlu Install)
Klik link berikut untuk langsung membuka aplikasi web:
#### 👉 [Streamlit - Brain Tumor Object Detection](https://brain-tumor-object-detection-richardtanjaya.streamlit.app/)

## 💡 Fitur
- ✅ **Object Detection** — Mendeteksi dan mengklasifikasikan tumor: Glioma, Meningioma, Pituitary, atau No Tumor.
- ✅ **YOLOv8 Backbone** — Menggunakan model YOLOv8 yang kuat untuk deteksi tumor otak dengan recall tinggi.
- ✅ **Image Input** — Pengguna dapat mengunggah gambar MRI atau memilih contoh gambar yang tersedia.
- ✅ **Video Input** — Mendukung analisis video MRI dengan durasi sampai dengan 30 detik.
- ✅ **Real-time Webcam** — Aplikasi dapat melakukan deteksi secara langsung menggunakan kamera pengguna.
- ✅ **Downloadable Output** — Hasil deteksi dapat diunduh dalam bentuk gambar beranotasi (.jpg) dan file CSV.
- ✅ **Detection Statistics** — Menampilkan jumlah deteksi per kelas beserta tingkat confidencenya.

## ⚙️ Tech Stack
- **Model Architecture**: YOLOv8 (Ultralytics)
- **Web Framework**: Streamlit
- **Image & Video Processing**: OpenCV, Pillow (PIL)
- **Data Handling & Analysis**: NumPy, Pandas
- **Deployment Platform**: Streamlit Cloud
- **Model Format**: PyTorch `.pt` (best.pt)

## 🧠 Model Details
- **Model**: YOLOv8 — custom trained on brain tumor MRI dataset with bounding boxes.
- **mAP@0.5 (Mean Average Precision)**: 95.8%
- **mAP@0.5:0.95**: 79.2%
- **Recall**: 91.9%
- **Dataset:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes)

## ⭐ Deployment
Aplikasi ini di-deploy menggunakan:
- Streamlit Cloud
- GitHub

## 👨‍💻 Pembuat
Richard Dean Tanjaya

## 📝 License
Proyek ini bersifat open-source dan bebas digunakan untuk keperluan edukasi dan penelitian.