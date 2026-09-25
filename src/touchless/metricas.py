"""Registro de métricas por quadro em CSV, para análise posterior (ver analise.py)."""

import csv
from pathlib import Path

CAMPOS = ["t", "quadro", "inferiu", "inferencia_ms", "quadro_ms", "mao", "gesto"]


class RegistroMetricas:
    def __init__(self, caminho: Path):
        caminho.parent.mkdir(parents=True, exist_ok=True)
        self._arquivo = caminho.open("w", newline="")
        self._csv = csv.DictWriter(self._arquivo, fieldnames=CAMPOS)
        self._csv.writeheader()

    def registrar(self, **linha) -> None:
        self._csv.writerow(linha)

    def fechar(self) -> None:
        self._arquivo.close()
