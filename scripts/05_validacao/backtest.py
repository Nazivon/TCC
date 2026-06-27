"""
scripts/05_validacao/backtest_mdp.py
=====================================
Backtest (holdout) do MDP estimado. Avalia taxa de acordo, JS-divergência
e acurácia de transição em dados de teste.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from scipy.spatial.distance import jensenshannon
from utils.plots import plotar_treino_vs_teste, plotar_distribuicao_acordo

# Configurações globais
ARQ_PARTIDAS = Path("../../data/partidas.json")
ARQ_MOVIMENTOS = Path("../../data/movimentos.json")
ARQ_MAZES = Path("../../data/mazes.json")
PASTA_SAIDA = Path("../../graficos")
PASTA_SAIDA.mkdir(exist_ok=True)

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

def estimar_mdp(df_treino, ids_sucesso):
    """Constrói T e R a partir do treino, identificando terminais apenas por sucesso[cite: 11]."""
    counts = defaultdict(lambda: defaultdict(int))
    terminais_cand = defaultdict(Counter)
    ids_treino = set(df_treino["game_id"].unique())
    sucesso_treino = ids_treino & ids_sucesso

    for gid, grupo in df_treino.groupby("game_id"):
        movs = grupo.sort_values("move_idx").to_dict("records")
        maze = movs[0]["maze_name"]
        for i in range(len(movs) - 1):
            s = (maze, int(movs[i]["x"]), int(movs[i]["y"]))
            s2 = (maze, int(movs[i+1]["x"]), int(movs[i+1]["y"]))
            a = movs[i+1]["direction"]
            if a in ("START", "STAY", "UNKNOWN"): continue
            is_bt = int(movs[i+1].get("is_backtrack", False))
            counts[(maze, s, a)][s2] += 1
            counts[(maze, s, a)][f"_bt_{s2}"] += is_bt
        if gid in sucesso_treino:
            terminais_cand[maze][(maze, int(movs[-1]["x"]), int(movs[-1]["y"]))] += 1

    terminais = {m: cand.most_common(1)[0][0] for m, cand in terminais_cand.items()}
    P, R, estados = {}, {}, defaultdict(set)
    for (m, s, a), nxt in counts.items():
        real = {k: v for k, v in nxt.items() if not str(k).startswith("_bt_")}
        total = sum(real.values())
        if total == 0: continue
        estados[m].add(s)
        if m not in P: P[m], R[m] = defaultdict(lambda: defaultdict(dict)), defaultdict(lambda: defaultdict(float))
        for s2, n in real.items():
            prob = n / total
            estados[m].add(s2)
            recomp = R_SAIDA if s2 == terminais.get(m) else (R_BACKTRACK if (nxt.get(f"_bt_{s2}", 0)/n) > 0.5 else R_PASSO)
            P[m][s][a][s2] = prob
            R[m][s][a] += prob * recomp
    return P, R, terminais, estados

# ... (Função rodar_vi e avaliar_teste similar à estrutura do notebook) ...

if __name__ == "__main__":
    df_tr, df_te, ids_s = carregar_e_dividir()
    P, R, term, est = estimar_mdp(df_tr, ids_s)
    # ... executa fluxo de validação ...
