"""
scripts/04_complexidade_estrutural/structural_analysis.py
============================================
Análise de Complexidade Estrutural.
Cruza os resultados de desvio (resumo_mdp.csv) com os dados estruturais (mazes.json)
para avaliar o impacto de características físicas do labirinto no comportamento humano[cite: 9].
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from utils.plots import (plotar_js_vs_complexidade, plotar_heatmap_correlacoes, 
                         plotar_desvio_vs_moves, plotar_ranking_desvio, plotar_histogramas_desvio)

def compilar_analise_estrutural(arq_resumo: Path, arq_mazes: Path) -> pd.DataFrame:
    """Calcula tortuosidade e densidade e cruza com a política do MDP[cite: 9]."""
    df_mdp = pd.read_csv(arq_resumo)
    with open(arq_mazes, "r", encoding="utf-8") as f:
        df_mazes = pd.DataFrame(json.load(f))

    df_mazes["tortuosidade"] = (df_mazes["avg_moves"] / (df_mazes["grid_width"] + df_mazes["grid_height"])).round(4)
    df_mazes["area_grid"] = df_mazes["grid_width"] * df_mazes["grid_height"]
    df_mazes["densidade"] = (df_mazes["avg_moves"] / df_mazes["area_grid"]).round(4)

    df_final = df_mdp.merge(df_mazes, on="maze_name", how="inner")
    df_final.to_csv("../../data/processed/analise_complexidade.csv", index=False)
    return df_final

def extrair_correlacoes(df: pd.DataFrame) -> pd.DataFrame:
    """Extrai matriz de correlação de Spearman para as variáveis chave[cite: 9]."""
    metricas = {
        "difficult_level": "Nível de dificuldade (1–5)", "avg_score_difficult": "Score médio de dificuldade",
        "avg_time_to_finish": "Tempo médio de conclusão (s)", "avg_moves": "Média de movimentos por partida",
        "tortuosidade": "Tortuosidade (moves/grid)", "densidade": "Densidade (moves/área)",
        "grid_width": "Largura do grid", "grid_height": "Altura do grid", "total_games": "Total de partidas"
    }

    resultados = []
    for col, label in metricas.items():
        sub = df[[col, "js_divergencia", "desvio_pct"]].dropna()
        if len(sub) < 5: continue

        r_js, p_js = stats.spearmanr(sub[col], sub["js_divergencia"])
        r_desv, p_desv = stats.spearmanr(sub[col], sub["desvio_pct"])

        resultados.append({
            "métrica": label, "coluna": col,
            "r_JS": round(r_js, 3), "p_JS": round(p_js, 4),
            "r_desvio": round(r_desv, 3), "p_desvio": round(p_desv, 4),
            "sig_JS": "✓" if p_js < 0.05 else "", "sig_desvio": "✓" if p_desv < 0.05 else ""
        })

    return pd.DataFrame(resultados).sort_values("r_JS", ascending=False)

if __name__ == "__main__":
    PASTA_DADOS = Path("../../data")
    arq_resumo = PASTA_DADOS / "processed/resumo_mdp.csv"
    arq_mazes = PASTA_DADOS / "mazes.json"
    
    df_estrutural = compilar_analise_estrutural(arq_resumo, arq_mazes)
    df_corr = extrair_correlacoes(df_estrutural)
    
    plotar_histogramas_desvio(df_estrutural)
    plotar_js_vs_complexidade(df_estrutural)
    plotar_heatmap_correlacoes(df_corr)
    plotar_desvio_vs_moves(df_estrutural)
    plotar_ranking_desvio(df_estrutural)
