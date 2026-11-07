
from pathlib import Path
import yaml
from ultralytics import YOLO

# Localiza automaticamente a pasta principal do projeto
ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"

# Configuração portátil do dataset
config = {
    "path": str(DATASET.resolve()),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 1,
    "names": {0: "luminaria_publica"},
}

# Cria um YAML temporário com os caminhos corretos
config_path = DATASET / "data_local.yaml"

with open(config_path, "w", encoding="utf-8") as file:
    yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)

# Carrega um modelo YOLO pré-treinado
model = YOLO("yolo11n.pt")

# Treinamento personalizado para identificar luminárias públicas
model.train(
    data=str(config_path),
    epochs=40,
    imgsz=640,
    batch=8,
    device=0,
    workers=0,
    project=str(ROOT / "results" / "training"),
    name="viela_yolo",
    exist_ok=True,
    patience=10,
    plots=True,
    seed=42,
)

print("\nTreinamento concluído!")
print("Modelo salvo em results/training/viela_yolo/weights/best.pt")
