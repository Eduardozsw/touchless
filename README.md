# Touchless

Controle do mouse por gestos da mão, em tempo real, só com a webcam.

Pipeline: captura com **OpenCV** → detecção dos 21 landmarks da mão com o **MediaPipe Hand Landmarker** →
classificação do gesto por regras geométricas (**NumPy**, vetorizado) → ação no mouse (**PyAutoGUI**).

## Gestos

| Gesto | Ação |
|---|---|
| Indicador + médio levantados (V) | Mover o cursor |
| Pinça polegar–indicador | Clique esquerdo (ao soltar) |
| V + pinça | Clique direito |
| Só o mínimo levantado | Arrastar |
| Punho fechado, mão no terço de cima / de baixo | Scroll para cima / para baixo |

## Rodando

Requer [uv](https://docs.astral.sh/uv/) e uma webcam. O modelo (`hand_landmarker.task`) é baixado na primeira execução
se não existir.

```bash
uv sync
uv run touchless             # controla o mouse de verdade; ESC para sair
uv run touchless --dry-run   # mostra a detecção sem mexer no mouse
```

Opções: `--inferir-cada N` (roda o modelo a cada N quadros, padrão 2), `--camera N`, `--duracao S`.

## Medindo desempenho

```bash
uv run touchless --dry-run --duracao 60 --metricas metricas/sessao.csv
uv run --extra analise touchless-analise metricas/sessao.csv
```

O primeiro comando grava por quadro: tempo, latência de inferência, tempo total do quadro, se havia mão e o gesto.
O segundo usa **Pandas** para resumir (FPS médio, p50/p95 de latência, % de quadros com mão) e **Matplotlib** para
gerar `sessao.png` (FPS ao longo do tempo, histograma de latência, quadros por gesto).

## Decisões

- **Inferência a cada 2 quadros:** o quadro intermediário reaproveita o último resultado. Troca um pouco de
  responsividade por menos CPU; `--inferir-cada 1` desliga isso para comparar.
- **Suavização:** média móvel das últimas 7 posições do cursor, contra o tremor dos landmarks.
- **Tolerância a perda:** se a mão some por até 0,3 s, a última detecção válida continua valendo.
- **Área ampliada:** o centro da câmera é ampliado 2× para alcançar a tela inteira sem esticar o braço.

## Estrutura

```
src/touchless/
  gestos.py    # regras de gesto e mapeamento câmera→tela (puro, testado)
  app.py       # loop da câmera, mouse e HUD
  metricas.py  # CSV por quadro
  analise.py   # relatório com Pandas + Matplotlib
tests/         # pytest com mãos sintéticas
```

```bash
uv run pytest
```

## Limitações

- Usa o modelo pré-treinado do MediaPipe: não há treino nem dataset próprio.
- Limiares (pinça, zonas de scroll, escala) foram ajustados na mão, sem avaliação sistemática.
