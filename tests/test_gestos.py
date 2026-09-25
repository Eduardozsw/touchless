import numpy as np
import pytest

from touchless.gestos import ARTICULACOES, INDICADOR_PONTA, PONTAS, POLEGAR_PONTA, detectar_gesto, mapear_para_tela


def mao(levantados=(False, False, False, False), pinca=False) -> np.ndarray:
    """Mão sintética: todos os pontos em y=0.6; dedo levantado tem a ponta em y=0.3."""
    lm = np.full((21, 2), 0.6, dtype=np.float32)
    lm[:, 0] = np.linspace(0.3, 0.7, 21)
    lm[ARTICULACOES, 1] = 0.5
    for ponta, up in zip(PONTAS, levantados):
        lm[ponta, 1] = 0.3 if up else 0.55
    lm[POLEGAR_PONTA] = lm[INDICADOR_PONTA] + (0.01 if pinca else 0.2)
    return lm


@pytest.mark.parametrize(
    "levantados, pinca, esperado",
    [
        ((False, False, False, False), False, "SCROLL"),
        ((False, False, False, True), False, "DRAG"),
        ((True, True, False, False), False, "MOVE"),
        ((True, True, False, False), True, "RIGHT_CLICK"),
        ((True, False, False, False), True, "PINCH"),
        ((True, False, False, False), False, "NONE"),
        ((True, True, True, True), False, "NONE"),
    ],
)
def test_detectar_gesto(levantados, pinca, esperado):
    assert detectar_gesto(mao(levantados, pinca)) == esperado


def test_mapear_para_tela_amplia_centro_e_limita_bordas():
    lm = np.zeros((21, 2), dtype=np.float32)
    lm[INDICADOR_PONTA] = (0.5, 0.5)
    assert mapear_para_tela(lm, 1920, 1080) == (960, 540)
    lm[INDICADOR_PONTA] = (0.9, 0.05)
    assert mapear_para_tela(lm, 1920, 1080) == (1920, 0)
