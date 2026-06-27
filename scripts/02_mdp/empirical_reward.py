"""
scripts/02_mdp/empirical_reward.py
==================================
Técnica observacional inspirada em Inverse Reinforcement Learning (IRL).
Estima a Função de Recompensa (R_emp) utilizando as métricas reais de 
desempenho dos jogadores (score, tempo, volume de movimentos e backtracks).
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from scripts.02_mdp.value_iteration import value_iteration
from scripts.02_mdp.policy_comparison import comparar_politicas
from utils.plots import plotar_sensibilidade_R_empirica

R_EMP_MIN, R_EMP_MAX = -5.0, 100.0

GRADE_PESOS = {
    "padrao":         {"orig_score": 0.50, "bt": 0.30, "moves": 0.10, "time": 0.10},
    "so_orig_score":  {"orig_score": 1.00, "bt": 0.00, "moves": 0.00, "time": 0.00},
    "sem_orig_score": {"orig_score": 0.00, "bt": 0.50, "moves": 0.25, "time": 0.25},
    "igualitario":    {"orig_score": 0.25, "bt": 0.25, "moves": 0.25, "time": 0.25},
}

def carregar_scores_partidas(arq_partidas) -> dict:
    import json
    with open(arq_partidas, "r", encoding="utf-8") as f:
        return {p["id"]: p for p in json.load(f) if p.get("maze_finish_time", 0.0) < 600}

def construir_scores_por_tripla(arq_movimentos, mazes_validos: set, ids_sucesso: set, scores_partidas: dict) -> dict:
    """Mapeia os metadados de desempenho (score final) a cada passo (s,a,s') executado na trajetória."""
    scores_por_tripla = defaultdict(list)
    df_completo = pd.read_json(arq_movimentos)
    df_completo = df_completo[df_completo["maze_name"].isin(mazes_validos) & df_completo["game_id"].isin(ids_sucesso)]
    
    for game_id, grupo in df_completo.groupby("game_id"):
        if game_id not in scores_partidas: continue
        metricas = scores_partidas[game_id]
        movs = grupo.sort_values("move_idx").to_dict("records")

        for i in range(len(movs) - 1):
            a = movs[i + 1]["direction"]
            if a in ("START", "STAY", "UNKNOWN"): continue
            maze, x, y = movs[i]["maze_name"], int(movs[i]["x"]), int(movs[i]["y"])
            scores_por_tripla[(maze, (maze, x, y), a)].append(metricas)

    return dict(scores_por_tripla)

def construir_R_empirica(mdp_por_labirinto: dict, scores_por_tripla: dict, pesos: dict) -> dict:
    """Normaliza o score composto intra-labirinto e computa a matriz de recompensa observacional."""
    mdp_r_emp = {}
    
    # Pré-computação para normalização por labirinto (simplificada para legibilidade)
    mx = defaultdict(lambda: {"score": 1, "backtrack_count": 1, "total_moves": 1, "maze_finish_time": 1})
    for (maze, s, a), lista in scores_por_tripla.items():
        for key in mx[maze].keys():
            mx[maze][key] = max(mx[maze][key], max([m.get(key, 0) for m in lista]))

    for maze, mdp in mdp_por_labirinto.items():
        R_novo = defaultdict(lambda: defaultdict(float))
        
        for s, acoes in mdp["T"].items():
            for a in acoes:
                lista = scores_por_tripla.get((maze, s, a), [])
                if not lista:
                    R_novo[s][a] = -1.0
                    continue

                scores = []
                for m in lista:
                    sc_n = np.clip(m.get("score", 0) / mx[maze]["score"], 0, 1)
                    bt_n = np.clip(m.get("backtrack_count", 0) / mx[maze]["backtrack_count"], 0, 1)
                    mv_n = np.clip(m.get("total_moves", 1) / mx[maze]["total_moves"], 0, 1)
                    tm_n = np.clip(m.get("maze_finish_time", 1) / mx[maze]["maze_finish_time"], 0, 1)

                    sc = (pesos["orig_score"] * sc_n + pesos["bt"] * (1 - bt_n) + 
                          pesos["moves"] * (1 - mv_n) + pesos["time"] * (1 - tm_n))
                    scores.append(sc)
                
                R_novo[s][a] = R_EMP_MIN + np.mean(scores) * (R_EMP_MAX - R_EMP_MIN)

        if mdp["terminal"]:
            for a in mdp["T"].get(mdp["terminal"], {}): R_novo[mdp["terminal"]][a] = 100.0

        mdp_copia = dict(mdp)
        mdp_copia["R"] = dict(R_novo)
        mdp_r_emp[maze] = mdp_copia

    return mdp_r_emp

def rodar_sensibilidade_R_empirica(mdp_por_labirinto: dict, politica_emp: dict, scores_por_tripla: dict, gamma=0.95):
    resumo = []
    for nome_pesos, pesos in GRADE_PESOS.items():
        mdp_emp = construir_R_empirica(mdp_por_labirinto, scores_por_tripla, pesos)
        vi_local = {maze: {"V": value_iteration(m, gamma)[0], "politica_otima": value_iteration(m, gamma)[1]} for maze, m in mdp_emp.items()}
        
        df_cen = comparar_politicas(politica_emp, vi_local, mdp_emp)
        resumo.append({"configuracao": nome_pesos, "js_medio": df_cen["js_divergencia"].mean(), "taxa_acordo": df_cen["taxa_acordo"].mean()})

    df_r_emp = pd.DataFrame(resumo)
    plotar_sensibilidade_R_empirica(df_r_emp)
    return df_r_emp
