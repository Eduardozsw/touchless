"""Loop principal: webcam → MediaPipe → gesto → mouse."""

import argparse
import os
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from touchless.gestos import CONEXOES, MEDIO_BASE, detectar_gesto, mapear_para_tela, para_array
from touchless.metricas import RegistroMetricas

MODELO_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODELO = Path("hand_landmarker.task")

TIMEOUT_ULTIMA_DETECCAO = 0.3
JANELA_SUAVIZACAO = 7
ZONA_SCROLL_CIMA, ZONA_SCROLL_BAIXO = 0.33, 0.66
VELOCIDADE_SCROLL = 20


class MouseNulo:
    """Substitui o pyautogui no --dry-run: nada mexe no mouse de verdade."""

    def size(self):
        return 1920, 1080

    def __getattr__(self, _nome):
        return lambda *a, **k: None


def criar_detector() -> vision.HandLandmarker:
    if not MODELO.exists():
        print("Baixando modelo...")
        urllib.request.urlretrieve(MODELO_URL, MODELO)
    opcoes = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODELO)),
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    return vision.HandLandmarker.create_from_options(opcoes)


def desenhar(frame: np.ndarray, lm: np.ndarray | None, gesto: str, fps: int) -> None:
    h, w = frame.shape[:2]
    if lm is not None:
        pontos = (lm * (w, h)).astype(int)
        for a, b in CONEXOES:
            cv2.line(frame, tuple(pontos[a]), tuple(pontos[b]), (200, 200, 200), 1)
        for p in pontos:
            cv2.circle(frame, tuple(p), 4, (0, 255, 100), -1)
    cv2.rectangle(frame, (0, 0), (320, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"Gesto: {gesto}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 100), 2)
    cv2.putText(frame, f"FPS: {fps}", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)


def rodar(dry_run: bool, metricas: Path | None, duracao: float | None, inferir_cada: int, camera: int) -> None:
    if dry_run:
        mouse = MouseNulo()
    else:
        import pyautogui as mouse

        mouse.FAILSAFE = False
        mouse.PAUSE = 0
    tela_w, tela_h = mouse.size()

    detector = criar_detector()
    registro = RegistroMetricas(metricas) if metricas else None

    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)

    posicoes = deque(maxlen=JANELA_SUAVIZACAO)
    clicando_esq = clicando_dir = arrastando = False
    ultimo_lm, ultimo_lm_t = None, 0.0
    ultimo_resultado = None
    quadro = 0
    fps_contador, fps_exibido, fps_t = 0, 0, time.time()
    inicio = time.time()

    print("Motor de Gestos ativo. ESC para sair." + (" (dry-run: mouse desligado)" if dry_run else ""))
    try:
        while duracao is None or time.time() - inicio < duracao:
            t_quadro = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            quadro += 1
            inferiu = quadro % inferir_cada == 0
            inferencia_ms = None
            if inferiu:
                t0 = time.perf_counter()
                ultimo_resultado = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                inferencia_ms = (time.perf_counter() - t0) * 1000

            fps_contador += 1
            if time.time() - fps_t >= 1.0:
                fps_exibido, fps_contador, fps_t = fps_contador, 0, time.time()

            # Última detecção válida com timeout
            if ultimo_resultado and ultimo_resultado.hand_landmarks:
                lm = para_array(ultimo_resultado.hand_landmarks[0])
                ultimo_lm, ultimo_lm_t = lm, time.time()
            elif ultimo_lm is not None and time.time() - ultimo_lm_t < TIMEOUT_ULTIMA_DETECCAO:
                lm = ultimo_lm
            else:
                lm = None

            rotulo = "---"
            if lm is not None:
                gesto = rotulo = detectar_gesto(lm)

                if gesto in ("MOVE", "RIGHT_CLICK", "DRAG"):
                    if gesto == "DRAG" and not arrastando:
                        arrastando = True
                        mouse.mouseDown(button="left")
                    posicoes.append(mapear_para_tela(lm, tela_w, tela_h))
                    sx, sy = np.mean(posicoes, axis=0).astype(int)
                    if gesto == "DRAG" or not arrastando:
                        mouse.moveTo(sx, sy)

                if gesto == "PINCH":
                    clicando_esq = True
                elif arrastando and gesto != "DRAG":
                    mouse.mouseUp(button="left")
                    arrastando = False
                elif clicando_esq:
                    mouse.click(button="left")
                    clicando_esq = False

                if gesto == "RIGHT_CLICK" and not clicando_dir:
                    mouse.click(button="right")
                    clicando_dir = True
                elif gesto != "RIGHT_CLICK":
                    clicando_dir = False

                if gesto == "SCROLL":
                    mao_y = lm[MEDIO_BASE, 1]
                    if mao_y < ZONA_SCROLL_CIMA:
                        mouse.scroll(VELOCIDADE_SCROLL)
                        rotulo = "SCROLL CIMA"
                    elif mao_y > ZONA_SCROLL_BAIXO:
                        mouse.scroll(-VELOCIDADE_SCROLL)
                        rotulo = "SCROLL BAIXO"
                    else:
                        rotulo = "SCROLL NEUTRO"

            desenhar(frame, lm, rotulo, fps_exibido)
            cv2.imshow("Touchless", frame)
            if cv2.waitKey(1) == 27:
                break

            if registro:
                registro.registrar(
                    t=round(time.time() - inicio, 4),
                    quadro=quadro,
                    inferiu=int(inferiu),
                    inferencia_ms=None if inferencia_ms is None else round(inferencia_ms, 3),
                    quadro_ms=round((time.perf_counter() - t_quadro) * 1000, 3),
                    mao=int(lm is not None),
                    gesto=rotulo,
                )
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if registro:
            registro.fechar()
            print(f"Métricas salvas em {metricas}")


def main() -> None:
    p = argparse.ArgumentParser(description="Controle do mouse por gestos da mão.")
    p.add_argument("--dry-run", action="store_true", help="Não mexe no mouse (para testar e medir).")
    p.add_argument("--metricas", type=Path, help="Grava métricas por quadro neste CSV.")
    p.add_argument("--duracao", type=float, help="Encerra depois de N segundos.")
    p.add_argument("--inferir-cada", type=int, default=2, help="Roda o modelo a cada N quadros (padrão: 2).")
    p.add_argument("--camera", type=int, default=int(os.environ.get("TOUCHLESS_CAMERA", 0)))
    a = p.parse_args()
    rodar(a.dry_run, a.metricas, a.duracao, a.inferir_cada, a.camera)


if __name__ == "__main__":
    main()
