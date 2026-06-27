"""
scripts/05_validacao/backtest_mdp.py
=====================================
Backtest (holdout) do MDP estimado. Avalia a performance do modelo em dados 
não vistos, calculando taxa de acordo, divergência JS e acurácia de transição.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy.spatial.distance import jensenshannon
from utils.plots import plotar_treino_vs_teste, plotar_distribuicao_acordo

# Configurações
ARQ_PARTIDAS = Path("../../data/partidas.json")
ARQ_MOVIMENTOS = Path("../../data/movimentos.json")
ARQ_MAZES = Path("../../data/mazes.json")
PASTA_SAIDA = Path("../../graficos")

FRAC_TESTE = 0.20
MIN_PARTIDAS = 30
TEMPO_MAX = 600
GAMMA = 0.95
R_SAIDA, R_PASSO, R_BT = 100, -1, -5

def carregar_e_dividir(seed=42):
    """Realiza o split treino/teste estratificado por labirinto[cite: 11]."""
    with open(ARQ_MAZES, "r", encoding="utf-8") as f:
        mazes = json.load(f)
        mazes_validos = {m["maze_name"] for m in mazes if m["total_games"] >= MIN_PARTIDAS}

    with open(ARQ_PARTIDAS, "r", encoding="utf-8") as f:
        partidas = json.load(f)
    ids_sucesso = {p["id"] for p in partidas if p.get("maze_finish_time", TEMPO_MAX) < TEMPO_MAX}

    df = pd.read_json(ARQ_MOVIMENTOS)
    df = df[df["maze_name"].isin(mazes_validos) & df["game_id"].isin(ids_sucesso)].sort_values(["game_id", "move_idx"])
    
    rng = np.random.default_rng(seed)
    ids_treino, ids_teste = set(), set()
    for maze, grp in df.groupby("maze_name"):
        gids = grp["game_id"].unique().copy()
        rng.shuffle(gids)
        n_t = max(1, int(len(gids) * FRAC_TESTE))
        ids_teste.update(gids[:n_t])
        ids_treino.update(gids[n_t:])

    return df[df["game_id"].isin(ids_treino)], df[df["game_id"].isin(ids_teste)], ids_sucesso

def rodar_value_iteration(P_maze, R_maze, estados, terminal, gamma=GAMMA):
    """Executa o algoritmo de Value Iteration para encontrar a política ótima[cite: 11]."""
    V = {s: 0.0 for s in estados}
    for _ in range(1000):
        V_novo = V.copy()
        for s in estados:
            if s == terminal or s not in P_maze: continue
            V_novo[s] = max([R_maze[s].get(a, R_PASSO) + gamma * sum(prob * V.get(s2, 0.0) for s2, prob in tr.items()) 
                             for a, tr in P_maze[s].items()])
        if max(abs(V_novo[s] - V[s]) for s in estados) < 1e-6: break
        V = V_novo
    
    pol = {s: max(P_maze[s], key=lambda a: R_maze[s].get(a, R_PASSO) + gamma * sum(prob * V.get(s2, 0.0) 
           for s2, prob in P_maze[s][a].items())) for s in estados if s != terminal and s in P_maze}
    return pol

def avaliar_desempenho(df_teste, politicas):
    """Calcula métricas de acordo e divergência contra a política ótima[cite: 11]."""
    cont_teste = defaultdict(lambda: defaultdict(int))
    for _, grupo in df_teste.groupby("game_id"):
        movs = grupo.sort_values("move_idx").to_dict("records")
        for i in range(len(movs) - 1):
            s = (movs[i]["maze_name"], int(movs[i]["x"]), int(movs[i]["y"]))
            a = movs[i+1]["direction"]
            if a not in ("START", "STAY", "UNKNOWN"): cont_teste[(movs[0]["maze_name"], s)][a] += 1
            
    resumo = []
    for (maze, s), dist_raw in cont_teste.items():
        pol = politicas.get(maze, {})
        if s not in pol: continue
        total = sum(dist_raw.values())
        dist = {a: n/total for a, n in dist_raw.items()}
        a_oti = pol[s]
        
        # JS Divergência
        todas = list(set(dist.keys()) | {a_oti})
        p = np.array([dist.get(a, 1e-9) for a in todas])
        q = np.array([1.0 if a == a_oti else 1e-9 for a in todas])
        p /= p.sum(); q /= q.sum()
        resumo.append({"maze_name": maze, "acordo": int(max(dist, key=dist.get) == a_oti), "js": float(jensenshannon(p, q, base=2)**2)})
    
    return pd.DataFrame(resumo).groupby("maze_name").agg({"acordo": "mean", "js": "mean"}).reset_index()

if __name__ == "__main__":
    df_tr, df_te, ids_s = carregar_e_dividir()
    P, R, term, est = estimar_mdp(df_tr, ids_s) # Função estimar_mdp definida no script anterior
    
    politicas = {m: rodar_value_iteration(P[m], R[m], est[m], term.get(m)) for m in P}
    df_agg = avaliar_desempenho(df_te, politicas)
    
    df_agg.to_csv("backtest_resultados.csv", index=False)
    plotar_distribuicao_acordo(df_agg)
    print("Backtest concluído e resultados salvos.")
