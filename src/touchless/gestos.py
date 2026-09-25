"""Classificação de gestos a partir dos 21 landmarks da mão.

Os landmarks chegam como um array NumPy (21, 2) com x, y normalizados em [0, 1]
(origem no canto superior esquerdo: y cresce para baixo). Tudo aqui é puro e
testável sem câmera.
"""

import numpy as np

# Índices dos landmarks do MediaPipe Hand Landmarker
PULSO = 0
POLEGAR_PONTA = 4
INDICADOR_PONTA = 8
MEDIO_BASE = 9
PONTAS = np.array([8, 12, 16, 20])      # indicador, médio, anelar, mínimo
ARTICULACOES = np.array([6, 10, 14, 18])  # PIP de cada um desses dedos

LIMIAR_PINCA = 0.05
MARGEM_INDICADOR = 0.04

CONEXOES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def para_array(landmarks) -> np.ndarray:
    """Converte a lista de landmarks do MediaPipe em array (21, 2)."""
    return np.array([(p.x, p.y) for p in landmarks], dtype=np.float32)


def dedos_levantados(lm: np.ndarray) -> np.ndarray:
    """Máscara booleana [indicador, médio, anelar, mínimo]: ponta acima da articulação."""
    return lm[PONTAS, 1] < lm[ARTICULACOES, 1]


def distancia_pinca(lm: np.ndarray) -> float:
    return float(np.linalg.norm(lm[POLEGAR_PONTA] - lm[INDICADOR_PONTA]))


def indicador_apontando(lm: np.ndarray) -> bool:
    outros = lm[PONTAS[1:], 1].mean()
    return bool(lm[INDICADOR_PONTA, 1] < outros - MARGEM_INDICADOR)


def detectar_gesto(lm: np.ndarray) -> str:
    indicador, medio, anelar, minimo = dedos_levantados(lm)
    pinca = distancia_pinca(lm) < LIMIAR_PINCA

    # Scroll: punho fechado
    if not indicador_apontando(lm) and not (medio or anelar or minimo):
        return "SCROLL"
    # Drag: só o mínimo levantado
    if minimo and not (indicador or medio or anelar):
        return "DRAG"
    # V (indicador + médio)
    if indicador and medio and not (anelar or minimo):
        return "RIGHT_CLICK" if pinca else "MOVE"
    # Pinça: clique esquerdo
    if pinca and not medio:
        return "PINCH"
    return "NONE"


def mapear_para_tela(lm: np.ndarray, largura: int, altura: int, escala: float = 2.0) -> tuple[int, int]:
    """Ponta do indicador → coordenada de tela, ampliando a área central da câmera."""
    xy = np.clip((lm[INDICADOR_PONTA] - 0.5) * escala + 0.5, 0.0, 1.0)
    return int(xy[0] * largura), int(xy[1] * altura)
