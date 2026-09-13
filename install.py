# install.py (VERSI FINAL DENGAN PYMYSQL)

import subprocess
import sys
import platform

# --- KONFIGURASI INSTALASI ---

# GANTI VARIABEL INI:
# Jika Anda punya GPU NVIDIA: Ganti 'cpu' menjadi 'cu118' atau 'cu121'
# Jika Anda HANYA punya CPU: Biarkan sebagai 'cpu'
PYTORCH_CUDA_VERSION = 'cu118' # Pilihan default untuk GPU
PYTORCH_INDEX_URL = f"https://download.pytorch.org/whl/{PYTORCH_CUDA_VERSION}"

# Daftar paket non-PyTorch (AI, Web, DB)
REQUIRED_PACKAGES = [
    "ultralytics",
    "facenet-pytorch",
    "scikit-learn",
    "numpy",
    "pandas",
    "opencv-python",
    "sqlalchemy",
    "pymysql",       # <-- Wajib untuk koneksi MySQL
    "Pillow",        # <-- Wajib untuk operasi gambar Base64/PIL
    "fastapi",
    "uvicorn[standard]",
    "jinja2",
    "python-multipart",
]

def run_pip_install(command):
    """Menjalankan perintah pip install dan mencetak hasilnya."""
    print(f"\n>>> Menjalankan: pip {' '.join(command)}")
    try:
        # sys.executable memastikan kita menggunakan python.exe di dalam virtual environment yang aktif
        result = subprocess.run([sys.executable, "-m", "pip", "install"] + command, 
                                check=True, 
                                capture_output=True, 
                                text=True)
        print("INSTALASI BERHASIL.")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"ERROR INSTALASI GAGAL (Kode Keluar {e.returncode}):")
        print(f"Perintah: {' '.join(e.cmd)}")
        print("\n--- OUTPUT ERROR ---")
        print(e.stderr)
        print("\n--- CATATAN PENTING ---")
        print("Pastikan Anda berada di virtual environment yang menggunakan Python 3.10 atau 3.11.")
        print("Jika error terjadi pada NumPy, segera ganti ke versi Python yang stabil.")
        sys.exit(1)

def main():
    print("--- MEMULAI INSTALASI DEPENDENSI PROYEK ABSENSI OTOMATIS ---")
    
    # 1. Instalasi PyTorch (Wajib)
    pytorch_command = ["torch", "torchvision", "torchaudio", "--index-url", PYTORCH_INDEX_URL]
    run_pip_install(pytorch_command)
    
    # 2. Instalasi Paket Lainnya (Termasuk NumPy, MySQL, dan FastAPI)
    run_pip_install(REQUIRED_PACKAGES)

    print("\n--- SEMUA INSTALASI SELESAI ---")
    print("Anda sekarang bisa menjalankan: uvicorn app_fastapi:app --reload")

if __name__ == "__main__":
    main()