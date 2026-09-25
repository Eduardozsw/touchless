"""Relatório de desempenho a partir do CSV gravado com --metricas.

    uv run --extra analise touchless-analise metricas/sessao.csv
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def resumir(df: pd.DataFrame) -> pd.Series:
    duracao = df["t"].iloc[-1] - df["t"].iloc[0]
    inf = df["inferencia_ms"].dropna()
    return pd.Series(
        {
            "quadros": len(df),
            "duracao_s": round(duracao, 1),
            "fps_medio": round((len(df) - 1) / duracao, 1) if duracao > 0 else float("nan"),
            "quadro_ms_p50": df["quadro_ms"].median(),
            "quadro_ms_p95": df["quadro_ms"].quantile(0.95),
            "inferencias": len(inf),
            "inferencia_ms_p50": inf.median(),
            "inferencia_ms_p95": inf.quantile(0.95),
            "mao_detectada_pct": round(100 * df["mao"].mean(), 1),
        }
    ).round(2)


def graficos(df: pd.DataFrame, destino: Path) -> None:
    fig, (a, b, c) = plt.subplots(3, 1, figsize=(9, 9), constrained_layout=True)

    fps = df.set_index(pd.to_timedelta(df["t"], unit="s"))["quadro"].resample("1s").count()
    a.plot(fps.index.total_seconds(), fps.values)
    a.set(title="FPS por segundo", xlabel="tempo (s)", ylabel="quadros")

    b.hist(df["inferencia_ms"].dropna(), bins=40)
    b.set(title="Latência de inferência (MediaPipe)", xlabel="ms", ylabel="inferências")

    df.loc[df["mao"] == 1, "gesto"].value_counts().plot.barh(ax=c)
    c.set(title="Quadros por gesto (com mão detectada)", xlabel="quadros")

    fig.savefig(destino, dpi=120)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", type=Path)
    a = p.parse_args()

    df = pd.read_csv(a.csv)
    resumo = resumir(df)
    print(resumo.to_string())

    png = a.csv.with_suffix(".png")
    graficos(df, png)
    resumo.to_json(a.csv.with_suffix(".json"), indent=2)
    print(f"\nGráficos: {png}\nResumo:   {a.csv.with_suffix('.json')}")


if __name__ == "__main__":
    main()
