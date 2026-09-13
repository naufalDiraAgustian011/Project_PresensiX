# database.py

from sqlalchemy import create_engine, Column, Integer, String, DateTime, PickleType, ForeignKey, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session 
from datetime import datetime
from sqlalchemy.engine.url import URL
import pymysql 

# --- KONFIGURASI MYSQL LARAGON ---
DB_CONFIG = {
    "drivername": "mysql+pymysql",
    "username": "root",    
    "password": "",        
    "host": "localhost",
    "port": 3306,
    "database": "absensi_mahasiswa" 
}

# Konfigurasi Database Engine
ENGINE = create_engine(
    URL.create(**DB_CONFIG),
    pool_recycle=3600, 
    pool_pre_ping=True
)
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=ENGINE)

# --- MODELS ---

class Kelas(Base):
    __tablename__ = "kelas"
    id = Column(Integer, primary_key=True, index=True)
    nama_kelas = Column(String(255), unique=True, index=True, nullable=False)
    mahasiswa = relationship("Mahasiswa", back_populates="kelas")
    sesi = relationship("SesiAbsensi", back_populates="kelas")

class Mahasiswa(Base):
    __tablename__ = "mahasiswa"
    nim = Column(String(50), primary_key=True, index=True) 
    nama = Column(String(255), nullable=False)
    kelas_id = Column(Integer, ForeignKey("kelas.id"))
    embedding = Column(PickleType, nullable=True) 
    
    kelas = relationship("Kelas", back_populates="mahasiswa")
    log_absensi = relationship("LogAbsensi", back_populates="mahasiswa")

class SesiAbsensi(Base):
    __tablename__ = "sesi_absensi"
    id = Column(Integer, primary_key=True, index=True)
    kelas_id = Column(Integer, ForeignKey("kelas.id"))
    waktu_mulai = Column(DateTime, default=datetime.now)
    waktu_selesai = Column(DateTime, nullable=True)
    status_aktif = Column(Boolean, default=True)
    kelas = relationship("Kelas", back_populates="sesi")
    log_absensi = relationship("LogAbsensi", back_populates="sesi")

class LogAbsensi(Base):
    __tablename__ = "log_absensi"
    id = Column(Integer, primary_key=True, index=True)
    nim_mahasiswa = Column(String(50), ForeignKey("mahasiswa.nim"))
    waktu_absensi = Column(DateTime, nullable=True)
    sesi_id = Column(Integer, ForeignKey("sesi_absensi.id"))
    
    status = Column(String(50), nullable=False) 
    mahasiswa = relationship("Mahasiswa", back_populates="log_absensi")
    sesi = relationship("SesiAbsensi", back_populates="log_absensi")

# --- FUNGSI UTAMA DATABASE ---

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()