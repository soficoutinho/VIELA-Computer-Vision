
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT / "results" / "training" / "viela_yolo"
    / "weights" / "best.pt"
)

IMAGES_PATH = ROOT / "dataset" / "test" / "images"

OUTPUT_PATH = ROOT / "results" / "images"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

model = YOLO(str(MODEL_PATH))

# Seleciona até 10 imagens do conjunto de teste
extensions = {".jpg", ".jpeg", ".png", ".webp"}
images = sorted(
    file for file in IMAGES_PATH.iterdir()
    if file.suffix.lower() in extensions
)[:10]

if not images:
    raise FileNotFoundError("Nenhuma imagem encontrada para teste.")

for image_path in images:
    results = model.predict(
        source=str(image_path),
        conf=0.25,
        imgsz=640,
        device=0,
        save=True,
        project=str(OUTPUT_PATH),
        name="deteccoes",
        exist_ok=True,
        verbose=False,
    )

    quantidade = len(results[0].boxes)
    print(f"{image_path.name}: {quantidade} luminaria(s) detectada(s)")

print("\nDeteccoes concluidas!")
print(f"Imagens salvas em: {OUTPUT_PATH / 'deteccoes'}")
