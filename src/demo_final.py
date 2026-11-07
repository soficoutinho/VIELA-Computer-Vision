
from pathlib import Path
from queue import Queue, Empty, Full
from threading import Thread, Event
import time

import cv2
import torch
from ultralytics import YOLO


# ==========================================
# CONFIGURACOES
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT / "results" / "training"
    / "viela_yolo" / "weights" / "best.pt"
)

VIDEO_PATH = ROOT / "videos" / "entrada.mp4"

DEVICE = 0 if torch.cuda.is_available() else "cpu"

IMGSZ = 960
CONF = 0.10

WINDOW = "VIELA - Deteccao de Luminarias"

MAX_DISPLAY_WIDTH = 1600
MAX_DISPLAY_HEIGHT = 900

QUEUE_SIZE = 4


# ==========================================
# LEITURA DO VIDEO EM THREAD
# ==========================================

class VideoReader:

    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(str(video_path))

        if not self.cap.isOpened():
            raise RuntimeError(
                "Nao foi possivel abrir o video."
            )

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        if self.fps <= 0:
            self.fps = 30.0

        self.total_frames = int(
            self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        self.queue = Queue(maxsize=QUEUE_SIZE)
        self.stop_event = Event()

        self.thread = Thread(
            target=self._read_frames,
            daemon=True
        )

    def _read_frames(self):

        try:
            while not self.stop_event.is_set():

                ok, frame = self.cap.read()

                if not ok:
                    break

                while not self.stop_event.is_set():
                    try:
                        self.queue.put(
                            frame,
                            timeout=0.1
                        )
                        break
                    except Full:
                        continue

        finally:
            self.cap.release()

            if not self.stop_event.is_set():
                while not self.stop_event.is_set():
                    try:
                        self.queue.put(
                            None,
                            timeout=0.1
                        )
                        break
                    except Full:
                        continue

    def start(self):
        self.thread.start()

    def read(self):

        while True:
            if self.stop_event.is_set():
                return None

            try:
                return self.queue.get(timeout=0.1)
            except Empty:
                if not self.thread.is_alive():
                    return None

    def stop(self):
        self.stop_event.set()

    def join(self):
        self.thread.join(timeout=3.0)


# ==========================================
# VERIFICACAO DOS ARQUIVOS
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Modelo nao encontrado: {MODEL_PATH}"
    )

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video nao encontrado: {VIDEO_PATH}"
    )


# ==========================================
# CARREGAMENTO E AQUECIMENTO DO YOLO
# ==========================================

print("\n===================================")
print("VIELA - DEMONSTRACAO OTIMIZADA")
print("===================================")
print(f"Dispositivo: {DEVICE}")
print(f"Resolucao de inferencia: {IMGSZ}")
print(f"Confianca minima: {CONF}")

model = YOLO(str(MODEL_PATH))

cap_warmup = cv2.VideoCapture(str(VIDEO_PATH))
ok, primeiro_frame = cap_warmup.read()
cap_warmup.release()

if not ok:
    raise RuntimeError(
        "O video nao possui quadros validos."
    )

print("Aquecendo o modelo...")

for _ in range(10):
    model.predict(
        source=primeiro_frame,
        conf=CONF,
        imgsz=IMGSZ,
        device=DEVICE,
        verbose=False,
    )

if torch.cuda.is_available():
    torch.cuda.synchronize()


# ==========================================
# INICIALIZACAO
# ==========================================

reader = VideoReader(VIDEO_PATH)

print(f"FPS original: {reader.fps:.2f}")
print(f"Total de quadros: {reader.total_frames}")
print("Pressione Q para sair.")
print("===================================\n")

cv2.namedWindow(
    WINDOW,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW,
    MAX_DISPLAY_WIDTH,
    MAX_DISPLAY_HEIGHT
)

frames_processados = 0
total_deteccoes = 0
maior_quantidade = 0

fps_atual = 0.0
frames_intervalo = 0

inicio_total = time.perf_counter()
inicio_intervalo = inicio_total

interrompido = False

reader.start()


# ==========================================
# PROCESSAMENTO
# ==========================================

try:
    while True:

        frame = reader.read()

        if frame is None:
            break

        # ----------------------------------
        # INFERENCIA NO QUADRO ORIGINAL
        # ----------------------------------

        resultado = model.predict(
            source=frame,
            conf=CONF,
            imgsz=IMGSZ,
            device=DEVICE,
            verbose=False,
        )[0]

        quantidade = len(resultado.boxes)

        total_deteccoes += quantidade
        maior_quantidade = max(
            maior_quantidade,
            quantidade
        )

        frames_processados += 1
        frames_intervalo += 1

        # ----------------------------------
        # REDIMENSIONAMENTO APOS INFERENCIA
        # ----------------------------------

        altura_original, largura_original = frame.shape[:2]

        escala = min(
            MAX_DISPLAY_WIDTH / largura_original,
            MAX_DISPLAY_HEIGHT / altura_original,
            1.0
        )

        nova_largura = max(
            1,
            int(largura_original * escala)
        )

        nova_altura = max(
            1,
            int(altura_original * escala)
        )

        if escala < 1.0:
            exibicao = cv2.resize(
                frame,
                (nova_largura, nova_altura),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            exibicao = frame.copy()

        # Escalas exatas apos arredondamento
        escala_x = nova_largura / largura_original
        escala_y = nova_altura / altura_original

        # ----------------------------------
        # DESENHO DAS CAIXAS NA IMAGEM MENOR
        # ----------------------------------

        if quantidade > 0:

            coordenadas = (
                resultado.boxes.xyxy
                .cpu()
                .numpy()
            )

            confiancas = (
                resultado.boxes.conf
                .cpu()
                .numpy()
            )

            for coordenada, confianca in zip(
                coordenadas,
                confiancas
            ):

                x1, y1, x2, y2 = coordenada

                x1 = int(x1 * escala_x)
                y1 = int(y1 * escala_y)
                x2 = int(x2 * escala_x)
                y2 = int(y2 * escala_y)

                cv2.rectangle(
                    exibicao,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    exibicao,
                    f"Luminaria {confianca:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

        # ----------------------------------
        # CALCULO DO FPS
        # ----------------------------------

        agora = time.perf_counter()
        intervalo = agora - inicio_intervalo

        if intervalo >= 0.5:
            fps_atual = (
                frames_intervalo / intervalo
            )
            frames_intervalo = 0
            inicio_intervalo = agora

        # ----------------------------------
        # INFORMACOES NA TELA
        # ----------------------------------

        cv2.putText(
            exibicao,
            f"VIELA | Luminarias: {quantidade}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            exibicao,
            f"FPS: {fps_atual:.1f}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------
        # EXIBICAO DE TODOS OS QUADROS
        # ----------------------------------

        cv2.imshow(WINDOW, exibicao)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord("q"):
            interrompido = True
            break

finally:

    if interrompido:
        reader.stop()

    reader.join()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    tempo_total = time.perf_counter() - inicio_total

    cv2.destroyAllWindows()


# ==========================================
# RESULTADOS FINAIS
# ==========================================

fps_medio = (
    frames_processados / tempo_total
    if tempo_total > 0
    else 0.0
)

media_deteccoes = (
    total_deteccoes / frames_processados
    if frames_processados > 0
    else 0.0
)

print("\n===================================")
print("VIELA - RESULTADO FINAL")
print("===================================")

print(f"Frames processados: {frames_processados}")
print(f"FPS original: {reader.fps:.2f}")
print(f"FPS medio: {fps_medio:.2f}")
print(f"Tempo total: {tempo_total:.2f} segundos")
print(f"Maior numero de luminarias: {maior_quantidade}")
print(f"Media de deteccoes por quadro: {media_deteccoes:.2f}")

if interrompido:
    print("Resultado: demonstracao interrompida.")
elif fps_medio >= reader.fps:
    print("Resultado: acompanha o FPS original.")
else:
    print("Resultado: abaixo do FPS original.")

print("===================================")
