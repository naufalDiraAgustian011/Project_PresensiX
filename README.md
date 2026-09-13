# Sistem Absensi Otomatis dengan Pengenalan Wajah 📸🎓

Sistem absensi cerdas berbasis web yang mengombinasikan *Object Detection* dan *Face Recognition* secara *real-time*. Aplikasi ini dibangun menggunakan **FastAPI** untuk backend, **YOLOv8** untuk deteksi area wajah/objek, dan **FaceNet (InceptionResnetV1)** untuk ekstraksi dan pencocokan identitas (*face embedding*).

Aplikasi ini dirancang untuk merekam kehadiran mahasiswa secara otomatis melalui *webcam* dan mencatatnya langsung ke dalam *database* (SQLAlchemy).

## 🛠️ Persyaratan Sistem (Prerequisites)

Untuk menjalankan aplikasi ini tanpa kendala instalasi *library* C++, sangat disarankan menggunakan spesifikasi berikut:
*   **Sistem Operasi:** Windows / Linux / macOS
*   **Python:** **Versi 3.10 atau 3.11** (Sangat direkomendasikan. Penggunaan Python 3.12/3.13 ke atas mungkin akan memicu *error* saat menginstal library Machine Learning).
*   **Perangkat Keras:** Memiliki *webcam* aktif. (Bisa menggunakan CPU biasa, atau GPU NVIDIA untuk deteksi yang lebih cepat).

---

## 🚀 Cara Cepat Menjalankan Aplikasi (Instalasi Sekali Jalan)

Pastikan kamu sudah menginstal **Python 3.10 atau 3.11** di komputermu. Buka terminal (Command Prompt/PowerShell) dan langsung *copy-paste* kumpulan perintah di bawah ini baris demi baris:

### Untuk Pengguna Windows:
```cmd
git clone [https://github.com/naufalDiraAgustian011/Project_PresensiX.git](https://github.com/naufalDiraAgustian011/Project_PresensiX.git)
cd Project_PresensiX
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python install.py
