import os, cv2, torch, pickle, numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

ENROLL_DIR = r"D:\pwebif\CAPSTONE_ABSENSI\enroll_images"
# SOLUSI: Tambahkan nama file '.pkl' di akhir path untuk menyimpan output.
EMB_DB_PATH = r"D:\pwebif\CAPSTONE_ABSENSI\embeddings_db.pkl"

def get_embedding(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face = mtcnn(img_rgb)
    if face is None:
        return None
    with torch.no_grad():
        emb = resnet(face.unsqueeze(0).to(device))
    return emb.cpu().numpy()[0]

db = {}
for student in os.listdir(ENROLL_DIR):
    sdir = os.path.join(ENROLL_DIR, student)
    if not os.path.isdir(sdir):
        continue
    print(f"Processing {student} ...")
    embs = []
    for fname in os.listdir(sdir):
        if fname.lower().endswith(('.jpg','.png','.jpeg')):
            emb = get_embedding(os.path.join(sdir, fname))
            if emb is not None:
                embs.append(emb)
    if len(embs) > 0:
        db[student] = np.mean(embs, axis=0)

# Baris 38 sekarang akan membuat file 'embeddings_db.pkl'
with open(EMB_DB_PATH, 'wb') as f:
    pickle.dump(db, f)

print("✅ Database embedding mahasiswa selesai dibuat!")