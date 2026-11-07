
from pathlib import Path
import time

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT / "results" / "training" / "viela_yolo"
    / "weights" / "best.pt"
)

VIDEO_PATH = ROOT / "videos" / "entrada.mp4"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Modelo nao encontrado: {MODEL_PATH}")

if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"Video nao encontrado: {VIDEO_PATH}")

device = 0 if torch.cuda.is_available() else "cpu"

print("===================================")
print("VIELA - DIAGNOSTICO DE DESEMPENHO")
print("===================================")
print(f"CUDA disponivel: {torch.cuda.is_available()}")
print(f"Dispositivo: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

model = YOLO(str(MODEL_PATH))
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Nao foi possivel abrir o video.")

# Obtem um quadro real do video
ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError("Nao foi possivel ler o primeiro quadro.")

# Aquecimento: evita contar a inicializacao da GPU
print("\nAquecendo o modelo...")

for _ in range(10):
    model.predict(
        source=frame,
        conf=0.10,
        imgsz=960,
        device=device,
        verbose=False,
    )

if torch.cuda.is_available():
    torch.cuda.synchronize()

# Mede apenas a inferencia, sem salvar ou exibir video
repeticoes = 50

print(f"Executando {repeticoes} inferencias...")

inicio = time.perf_counter()

for _ in range(repeticoes):
    model.predict(
        source=frame,
        conf=0.10,
        imgsz=960,
        device=device,
        verbose=False,
    )

if torch.cuda.is_available():
    torch.cuda.synchronize()

tempo_total = time.perf_counter() - inicio
tempo_medio_ms = (tempo_total / repeticoes) * 1000
fps_estimado = repeticoes / tempo_total

print("\n===================================")
print("RESULTADO DO BENCHMARK")
print("===================================")
print(f"Tempo medio por inferencia: {tempo_medio_ms:.2f} ms")
print(f"FPS estimado da inferencia: {fps_estimado:.2f}")
print("===================================")
