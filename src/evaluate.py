
from pathlib import Path
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"

MODEL_PATH = (
    ROOT / "results" / "training" / "viela_yolo"
    / "weights" / "best.pt"
)

config = {
    "path": str(DATASET.resolve()),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 1,
    "names": {0: "luminaria_publica"},
}

config_path = DATASET / "data_local.yaml"

with open(config_path, "w", encoding="utf-8") as file:
    yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)

model = YOLO(str(MODEL_PATH))

metrics = model.val(
    data=str(config_path),
    split="test",
    imgsz=640,
    batch=8,
    device=0,
    workers=0,
    plots=True,
    project=str(ROOT / "results" / "evaluation"),
    name="viela_test",
    exist_ok=True,
)

print("\n===== RESULTADOS DO TESTE =====")
print(f"Precisao: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
print(f"mAP@50: {metrics.box.map50:.4f}")
print(f"mAP@50-95: {metrics.box.map:.4f}")
print("================================")
