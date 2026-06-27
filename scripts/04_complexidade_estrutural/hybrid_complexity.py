"""
scripts/04_complexidade_estrutural/hybrid_complexity.py
==========================================
Cálculo da Complexidade Híbrida de Terada × Desvio Humano do MDP.
Extrai métricas granulares de partida (reflexões, voltas, streaks) e calcula
o Score Híbrido, cruzando com os desvios da política empírica[cite: 8].
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from utils.plots import plotar_dispersao_ch_desvio

# ── Pesos da fórmula de Terada (2025) ──
W_REFLEXOES = 0.50
W_VOLTAS    = 0.10
W_TEMPO     = 0.20
W_RISCO     = 0.20
W_FLUIDEZ   = 0.20

OPOSTOS = {'UP': 'DOWN', 'DOWN': 'UP', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}

def processar_metricas_movimentacao(arq_movimentos: Path, arq_partidas: Path, arq_mazes: Path, min_partidas: int = 30) -> pd.DataFrame:
    """Extrai voltas, reflexões e streaks a partir das trajetórias brutas[cite: 8]."""
    with open(arq_MAZES, "r", encoding="utf-8") as f:
        mazes = json.load(f)
        mazes_validos = {m["maze_name"] for m in mazes if m["total_games"] >= min_partidas}

    df_partidas = pd.DataFrame(json.load(open(arq_partidas, "r", encoding="utf-8")))
    df_partidas = df_partidas[(df_partidas["maze_finish_time"] < 600) & (df_partidas["maze_name"].isin(mazes_validos))]
    ids_sucesso = set(df_partidas["id"])

    df_mov = pd.read_json(arq_movimentos)
    df_mov = df_mov[df_mov["game_id"].isin(ids_sucesso) & df_mov["maze_name"].isin(mazes_validos)]
    df_mov = df_mov[~df_mov["direction"].isin(["START", "STAY", "UNKNOWN"])]
    df_mov = df_mov.sort_values(["game_id", "move_idx"])

    registros = []
    for game_id, grupo in df_mov.groupby("game_id"):
        maze = grupo.iloc[0]["maze_name"]
        dirs = grupo["direction"].tolist()
        n_mov = len(dirs)
        if n_mov == 0: continue

        n_voltas = n_reflexoes = n_streaks = 0
        for i in range(1, n_mov):
            ant, atual = dirs[i - 1], dirs[i]
            if ant == atual: continue
            elif OPOSTOS[ant] == atual: n_reflexoes += 1
            else: n_voltas += 1

        i = 0
        while i < n_mov:
            j = i + 1
            while j < n_mov and dirs[j] == dirs[i]: j += 1
            if j - i >= 2: n_streaks += 1
            i = j

        registros.append({
            "game_id": game_id, "maze_name": maze,
            "n_movimentos": n_mov, "n_reflexoes": n_reflexoes,
            "n_voltas": n_voltas, "n_streaks": n_streaks,
        })

    df_metr = pd.DataFrame(registros).merge(
        df_partidas[["id", "maze_finish_time"]].rename(columns={"id": "game_id"}), on="game_id", how="left"
    )
    return df_metr

def calcular_score_terada(df_metr: pd.DataFrame) -> pd.DataFrame:
    """Agrega métricas por labirinto e calcula o Score de Dificuldade Híbrido[cite: 8]."""
    def minmax(serie):
        mn, mx = serie.min(), serie.max()
        return pd.Series(np.zeros(len(serie)), index=serie.index) if mx == mn else (serie - mn) / (mx - mn)

    df_agg = df_metr.groupby("maze_name").agg(
        media_reflexoes=("n_reflexoes", "mean"), media_voltas=("n_voltas", "mean"),
        media_movimentos=("n_movimentos", "mean"), mediana_tempo=("maze_finish_time", "median"),
        p90_backtracks=("n_voltas", lambda x: np.percentile(x, 90)),
        media_streaks=("n_streaks", "mean"), n_partidas=("game_id", "count")
    ).reset_index()

    df_agg["sinuosidade"] = df_agg["media_voltas"] / df_agg["media_movimentos"]
    df_agg["fluidez"] = df_agg["media_streaks"] / df_agg["media_movimentos"]

    df_agg["norm_reflexoes"] = minmax(df_agg["media_reflexoes"])
    df_agg["norm_voltas"] = minmax(df_agg["media_voltas"])
    df_agg["norm_tempo"] = minmax(df_agg["mediana_tempo"])
    df_agg["norm_risco"] = minmax(df_agg["p90_backtracks"])
    df_agg["norm_fluidez"] = minmax(df_agg["fluidez"])

    df_agg["score_base"] = (df_agg["norm_reflexoes"] * W_REFLEXOES + df_agg["norm_voltas"] * W_VOLTAS + 
                            df_agg["norm_tempo"] * W_TEMPO + df_agg["norm_risco"] * W_RISCO)
    df_agg["score_final_raw"] = df_agg["score_base"] - df_agg["norm_fluidez"] * W_FLUIDEZ
    df_agg["complexidade_hibrida"] = (minmax(df_agg["score_final_raw"]) * 100).round(2)
    return df_agg

def cruzar_e_exportar(df_agg: pd.DataFrame, arq_resumo_mdp: Path) -> pd.DataFrame:
    """Cruza o score com os desvios empíricos e salva o resultado final[cite: 8]."""
    df_mdp = pd.read_csv(arq_resumo_mdp)
    df = df_mdp.merge(df_agg, on="maze_name", how="inner")
    
    cols = ["maze_name", "complexidade_hibrida", "taxa_acordo", "js_divergencia", "desvio_pct", 
            "n_estados", "n_partidas", "media_reflexoes", "media_voltas", "mediana_tempo", 
            "p90_backtracks", "fluidez", "sinuosidade", "norm_reflexoes", "norm_voltas", 
            "norm_tempo", "norm_risco", "norm_fluidez", "score_base", "score_final_raw"]
    
    df[cols].to_csv("../../data/processed/complexidade_hibrida.csv", index=False)
    return df

if __name__ == "__main__":
    PASTA_DADOS = Path("../../data")
    arq_mov = PASTA_DADOS / "movimentos.json"
    arq_part = PASTA_DADOS / "partidas.json"
    arq_mazes = PASTA_DADOS / "mazes.json"
    arq_resumo = PASTA_DADOS / "processed/resumo_mdp.csv"
    
    df_metr = processar_metricas_movimentacao(arq_mov, arq_part, arq_mazes)
    df_agg = calcular_score_terada(df_metr)
    df_final = cruzar_e_exportar(df_agg, arq_resumo)
    
    plotar_dispersao_ch_desvio(df_final)
