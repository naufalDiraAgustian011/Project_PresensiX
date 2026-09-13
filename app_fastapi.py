# app_fastapi.py (FINAL MODIFIED VERSION - FIXED TEMPLATE RESPONSE)

import os
import cv2
import torch
import pickle
import numpy as np
from datetime import datetime, timedelta
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from sklearn.neighbors import NearestNeighbors
from PIL import Image as PILImage 
import torchvision.transforms as transforms
from fastapi import FastAPI, Request, Depends, Form, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
from typing import List 
import base64
import io
import time 

# Import dari database.py
from database import get_db, Mahasiswa, Kelas, SesiAbsensi, LogAbsensi

# --- GLOBAL STATE ---

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

yolo_model = None
resnet = None
nn = None
names = None
cap = None
ACTIVE_SESI_ID = None
PREVIEW_CLASS_ID = None

# Ambang Batas Pengenalan Wajah (FaceNet Distance)
THRESHOLD = 0.50 
# Ambang Batas Kepercayaan Deteksi YOLO (Untuk memfilter benda non-wajah)
YOLO_CONF_THRESHOLD = 0.75 

ABSENSI_END_TIME = None 
IS_PAUSED = False 

# --- RELOAD LOGIC ---
def reload_models(db: Session):
    global nn, names
    mahasiswas = db.query(Mahasiswa).filter(Mahasiswa.embedding.isnot(None)).all()
    if mahasiswas:
        names = [m.nim for m in mahasiswas]
        emb_matrix = np.vstack([m.embedding for m in mahasiswas])
        nn = NearestNeighbors(n_neighbors=1, metric='cosine').fit(emb_matrix)
        return True
    return False

# --- FASTAPI LIFESPAN ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global yolo_model, resnet, nn, names, cap
    
    db = next(get_db())
    try:
        yolo_model = YOLO("runs/detect/train12/weights/best.pt").to(device)
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        reload_models(db) 
    finally:
        db.close()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
             print("ERROR: Gagal membuka kamera. Video stream tidak akan berfungsi.")
    
    print("Inisialisasi selesai. Server siap.")
    yield 
    
    if cap is not None and cap.isOpened():
        cap.release()

# --- FASTAPI SETUP & STATIC FILES ---
app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- FUNGSI UTILITIES ---

def get_face_embedding(face_img):
    global resnet, device
    try:
        img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        pil = PILImage.fromarray(img_rgb)
        tf = transforms.Compose([
            transforms.Resize((160, 160)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        t = tf(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = resnet(t).cpu().numpy()
        return emb
    except Exception:
        return None

# --- GENERATOR VIDEO STREAM ---

def generate_frames(db: Session):
    global ACTIVE_SESI_ID, PREVIEW_CLASS_ID, IS_PAUSED, ABSENSI_END_TIME
    is_preview = ACTIVE_SESI_ID is None
    
    while True:
        if IS_PAUSED:
            time.sleep(0.5)
            continue
            
        if cap is None or not cap.isOpened(): break
        success, frame = cap.read()
        if not success: break
        
        # Cek apakah waktu sesi resmi sudah habis
        if ACTIVE_SESI_ID is not None and ABSENSI_END_TIME and datetime.now() > ABSENSI_END_TIME:
            sesi = db.query(SesiAbsensi).filter(SesiAbsensi.id == ACTIVE_SESI_ID).first()
            if sesi and sesi.status_aktif:
                db.query(LogAbsensi).filter(LogAbsensi.sesi_id == ACTIVE_SESI_ID, LogAbsensi.status == "ALPHA").update({"status": "ALPHA_FINAL"}, synchronize_session=False)
                sesi.status_aktif = False
                sesi.waktu_selesai = datetime.now()
                db.commit()
                ACTIVE_SESI_ID = None
                PREVIEW_CLASS_ID = None
                print("Sesi berakhir otomatis.")
                
        # --- LOGIKA DETEKSI & PENGENALAN ---
        results = yolo_model(frame, conf=YOLO_CONF_THRESHOLD, verbose=False)
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                yolo_conf = float(box.conf[0])
                face = frame[y1:y2, x1:x2]
                if face.size == 0: continue

                emb = get_face_embedding(face)

                if emb is not None and nn is not None and len(names) > 0:
                    dist_array, idx_array = nn.kneighbors(emb, 1, return_distance=True)
                    dist = float(dist_array[0][0])
                    nim = names[int(idx_array[0][0])]
                    
                    if dist <= THRESHOLD and ACTIVE_SESI_ID is not None: 
                        log_entry = db.query(LogAbsensi).filter(LogAbsensi.nim_mahasiswa == nim, LogAbsensi.sesi_id == ACTIVE_SESI_ID).first()

                        if log_entry and log_entry.status == "ALPHA":
                            log_entry.status = "HADIR"
                            log_entry.waktu_absensi = datetime.now()
                            db.commit()
                            label = f"✅ HADIR: {nim}"
                            color = (0, 255, 0)
                        elif log_entry and log_entry.status == "HADIR":
                            label = f"✅ HADIR: {nim} (TERCATAT)"
                            color = (0, 150, 0)
                        else:
                            label = f"Mahasiswa ini di kelas lain"
                            color = (200, 200, 0)
                    elif is_preview:
                         # Mode Preview
                        label = f"PREVIEW: {nim} (F: {dist:.2f})"
                        color = (0, 150, 255) 
                    else:
                        label = f"❓ Unknown (Y: {yolo_conf:.2f})"
                        color = (0, 0, 255)
                else:
                    label = f"⚙️ Processing... (Y: {yolo_conf:.2f})"
                    color = (255, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    db.close() 

# --- ENDPOINT UTAMA & STREAM ---

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    global ACTIVE_SESI_ID, PREVIEW_CLASS_ID
    if PREVIEW_CLASS_ID is not None and ACTIVE_SESI_ID is None:
        return RedirectResponse(url=f"/absensi/preview/{PREVIEW_CLASS_ID}", status_code=303)
    # DIPERBARUI
    return templates.TemplateResponse(request=request, name="index.html", context={"active_sesi": ACTIVE_SESI_ID})

@app.get("/video_feed")
async def video_feed(db: Session = Depends(get_db)):
    return StreamingResponse(generate_frames(db), media_type="multipart/x-mixed-replace; boundary=frame")


# --- ENDPOINT TIMER CONTROL & STATUS ---

@app.post("/control/pause")
def control_pause():
    global IS_PAUSED
    IS_PAUSED = not IS_PAUSED
    return JSONResponse(content={"paused": IS_PAUSED})

@app.post("/control/stop_session")
def control_stop_session(db: Session = Depends(get_db)):
    global ACTIVE_SESI_ID, PREVIEW_CLASS_ID, ABSENSI_END_TIME
    
    if ACTIVE_SESI_ID is None:
        return JSONResponse(content={"stopped": False, "message": "Tidak ada sesi aktif"}, status_code=200)

    sesi = db.query(SesiAbsensi).filter(SesiAbsensi.id == ACTIVE_SESI_ID).first()
    
    if sesi and sesi.status_aktif:
        db.query(LogAbsensi).filter(LogAbsensi.sesi_id == ACTIVE_SESI_ID, LogAbsensi.status == "ALPHA").update({"status": "ALPHA_FINAL"}, synchronize_session=False)
        sesi.status_aktif = False
        sesi.waktu_selesai = datetime.now() 
        db.commit()
        
        ACTIVE_SESI_ID = None
        PREVIEW_CLASS_ID = None
        ABSENSI_END_TIME = None
        
        return JSONResponse(content={"stopped": True, "message": "Sesi berhasil dihentikan"}, status_code=200)
        
    return JSONResponse(content={"stopped": False, "message": "Sesi tidak ditemukan atau sudah selesai"}, status_code=200)

@app.get("/status/timer")
def status_timer():
    global ABSENSI_END_TIME, IS_PAUSED, ACTIVE_SESI_ID
    if ACTIVE_SESI_ID is None or ABSENSI_END_TIME is None:
        return JSONResponse(content={"active": False})
    
    remaining_time = ABSENSI_END_TIME - datetime.now()
    if remaining_time.total_seconds() < 0:
        return JSONResponse(content={"active": False, "remaining": 0})
        
    return JSONResponse(content={
        "active": True,
        "remaining": int(remaining_time.total_seconds()),
        "paused": IS_PAUSED
    })


# --- ENDPOINT MANAJEMEN KELAS & SESI ---

@app.get("/kelas/manage", response_class=HTMLResponse)
async def manage_kelas_page(request: Request, db: Session = Depends(get_db)):
    kelas_list = db.query(Kelas).all()
    # DIPERBARUI
    return templates.TemplateResponse(request=request, name="manage_kelas.html", context={"kelas_list": kelas_list})

@app.post("/kelas/create")
async def create_new_class(
    nama_kelas: str = Form(...), db: Session = Depends(get_db)
):
    if db.query(Kelas).filter(Kelas.nama_kelas == nama_kelas).first():
        raise HTTPException(status_code=400, detail="Nama kelas sudah ada.")
    new_kelas = Kelas(nama_kelas=nama_kelas)
    db.add(new_kelas)
    db.commit()
    return RedirectResponse(url="/kelas/manage", status_code=303)

@app.get("/absensi/start", response_class=HTMLResponse)
async def select_class_form(request: Request, db: Session = Depends(get_db)):
    kelas_list = db.query(Kelas).all()
    # DIPERBARUI
    return templates.TemplateResponse(request=request, name="select_class.html", context={"kelas_list": kelas_list})

@app.get("/absensi/preview/{class_id}", response_class=HTMLResponse)
async def live_preview_page(class_id: int, request: Request, db: Session = Depends(get_db)):
    global ACTIVE_SESI_ID, PREVIEW_CLASS_ID
    ACTIVE_SESI_ID = None
    PREVIEW_CLASS_ID = class_id 
    kelas = db.query(Kelas).filter(Kelas.id == class_id).first()
    mahasiswas = db.query(Mahasiswa).filter(Mahasiswa.kelas_id == class_id).all()
    if not kelas: raise HTTPException(status_code=404, detail="Kelas tidak ditemukan")
    # DIPERBARUI
    return templates.TemplateResponse(request=request, name="live_preview.html", context={"kelas": kelas, "mahasiswas": mahasiswas, "threshold": THRESHOLD})


@app.post("/absensi/start_official")
async def start_official_session(
    class_id: int = Form(...), duration: int = Form(...), db: Session = Depends(get_db)
):
    global ACTIVE_SESI_ID, PREVIEW_CLASS_ID, ABSENSI_END_TIME
    db.query(SesiAbsensi).filter(SesiAbsensi.status_aktif == True).update({SesiAbsensi.status_aktif: False})

    waktu_selesai = datetime.now() + timedelta(minutes=duration)
    ABSENSI_END_TIME = waktu_selesai
    
    new_sesi = SesiAbsensi(kelas_id=class_id, waktu_mulai=datetime.now(), waktu_selesai=waktu_selesai, status_aktif=True)
    db.add(new_sesi)
    db.flush()
    
    ACTIVE_SESI_ID = new_sesi.id
    PREVIEW_CLASS_ID = None
    mahasiswas = db.query(Mahasiswa).filter(Mahasiswa.kelas_id == class_id).all()
    for m in mahasiswas:
        db.add(LogAbsensi(nim_mahasiswa=m.nim, sesi_id=ACTIVE_SESI_ID, status="ALPHA"))
    
    db.commit()
    return RedirectResponse(url="/", status_code=303)


# --- ENDPOINT REKAP & HISTORY ---

@app.get("/rekap", response_class=HTMLResponse)
async def rekap_absensi(request: Request, db: Session = Depends(get_db)):
    sesi_history = db.query(SesiAbsensi).order_by(SesiAbsensi.waktu_mulai.desc()).all()
    rekap_data = []
    for sesi in sesi_history:
        hadir_count = db.query(LogAbsensi).filter(LogAbsensi.sesi_id == sesi.id, LogAbsensi.status == 'HADIR').count()
        alpha_count = db.query(LogAbsensi).filter(LogAbsensi.sesi_id == sesi.id, LogAbsensi.status.in_(['ALPHA', 'ALPHA_FINAL'])).count()
        kelas = db.query(Kelas).filter(Kelas.id == sesi.kelas_id).first()
        rekap_data.append({"sesi_id": sesi.id, "kelas_nama": kelas.nama_kelas if kelas else "N/A", "waktu_mulai": sesi.waktu_mulai.strftime("%Y-%m-%d %H:%M"), "hadir_total": hadir_count, "alpha_total": alpha_count, "status_aktif": sesi.status_aktif})
        
    # DIPERBARUI
    return templates.TemplateResponse(request=request, name="history.html", context={"rekap_data": rekap_data})

@app.get("/history/{sesi_id}", response_class=HTMLResponse)
async def history_detail(sesi_id: int, request: Request, db: Session = Depends(get_db)):
    sesi = db.query(SesiAbsensi).filter(SesiAbsensi.id == sesi_id).first()
    log_details = db.query(LogAbsensi, Mahasiswa.nama, Mahasiswa.nim).join(Mahasiswa, LogAbsensi.nim_mahasiswa == Mahasiswa.nim).filter(LogAbsensi.sesi_id == sesi_id).all()
    details = [{"nim": log.nim_mahasiswa, "nama": nama, "status": log.status, "waktu_absensi": log.waktu_absensi.strftime("%H:%M:%S") if log.waktu_absensi else 'N/A'} for log, nama, nim in log_details]
    kelas_nama = db.query(Kelas).filter(Kelas.id == sesi.kelas_id).first().nama_kelas
    # DIPERBARUI
    return templates.TemplateResponse(request=request, name="history_detail.html", context={"sesi": sesi, "kelas_nama": kelas_nama, "details": details})


# --- ENDPOINT PENDAFTARAN MAHASISWA (ENROLL) ---

@app.get("/enroll", response_class=HTMLResponse)
async def enroll_mahasiswa(request: Request, db: Session = Depends(get_db)):
    kelas_list = db.query(Kelas).all()
    success_message = request.query_params.get("success") == "true"
    # DIPERBARUI
    return templates.TemplateResponse(request=request, name="enroll_form.html", context={
        "kelas_list": kelas_list,
        "success_message": success_message
    })

@app.post("/enroll/submit")
async def process_enrollment(
    nim: str = Form(...),
    nama: str = Form(...),
    kelas_id: int = Form(...),
    sample_photos: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    if len(sample_photos) < 5:
        raise HTTPException(status_code=400, detail="Harus mengunggah minimal 5 foto.")

    if db.query(Mahasiswa).filter(Mahasiswa.nim == nim).first():
        raise HTTPException(status_code=400, detail=f"NIM {nim} sudah terdaftar.")
    if not db.query(Kelas).filter(Kelas.id == kelas_id).first():
        raise HTTPException(status_code=400, detail="ID Kelas tidak valid.")

    files_to_process = sample_photos[:20]
    embeddings = []
    
    for i, uploaded_file in enumerate(files_to_process):
        try:
            file_bytes = await uploaded_file.read()
            np_arr = np.frombuffer(file_bytes, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if cv_image is None: continue
                 
            emb = get_face_embedding(cv_image) 
            
            if emb is not None:
                embeddings.append(emb)
            else:
                print(f"Peringatan: Wajah tidak terdeteksi di file {uploaded_file.filename}.")
        except Exception as e:
            print(f"Error memproses file {uploaded_file.filename}: {e}")
            continue

    if len(embeddings) < 5:
        raise HTTPException(status_code=400, detail=f"Gagal mendapatkan minimal 5 embedding wajah yang jelas. Hanya {len(embeddings)} embedding yang berhasil.")
        
    avg_embedding = np.mean(embeddings, axis=0)

    new_mahasiswa = Mahasiswa(
        nim=nim,
        nama=nama,
        kelas_id=kelas_id,
        embedding=avg_embedding 
    )
    db.add(new_mahasiswa)
    db.commit()
    
    db = next(get_db())
    if reload_models(db):
        print("Model embeddings berhasil di-reload dengan data baru!")
        
    return RedirectResponse(url="/enroll?success=true", status_code=303)