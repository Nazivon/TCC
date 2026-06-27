"""
scripts/03_aprendizado/interplayer_learning.py
===========================================
Análise de Aprendizado Inter-Jogadores (Cross-sectional).
Investiga se jogadores mais experientes se aproximam mais da política ótima 
do que novatos.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from collections import defaultdict
from utils.plots import plotar_aprendizado_grupos, plotar_correlacao_continua

def definir_grupos_experiencia(df_jogadores: pd.DataFrame, metodo: str = "tercis", 
                               corte_novato: int = 3, corte_veterano: int = 15) -> pd.DataFrame:
    """Divide a base de jogadores em Novatos, Intermediários e Veteranos."""
    df_jog = df_jogadores.copy()
    
    if metodo == "tercis":
        t1 = df_jog["total_games"].quantile(0.33)
        t2 = df_jog["total_games"].quantile(0.67)
        def classificar_grupo(n):
            if n <= t1: return "novato"
            if n <= t2: return "intermediario"
            return "veterano"
    else:
        def classificar_grupo(n):
            if n <= corte_novato: return "novato"
            if n >= corte_veterano: return "veterano"
            return "intermediario"

    df_jog["grupo"] = df_jog["total_games"].apply(classificar_grupo)
    return df_jog

def reconstruir_politicas_por_grupo(df_mov: pd.DataFrame, df_jog: pd.DataFrame, mazes_validos: set) -> dict:
    """Agrega a distribuição de ações nos estados mapeados, isolando por grupo de experiência."""
    uid_para_grupo = dict(zip(df_jog["uid"], df_jog["grupo"]))
    contagens = defaultdict(lambda: defaultdict(int))

    df = df_mov[df_mov["maze_name"].isin(mazes_validos)]
    for game_id, grupo_df in df.groupby("game_id"):
        uid = grupo_df.iloc[0]["uid"]
        grp = uid_para_grupo.get(uid)
        if grp is None: continue
        
        maze = grupo_df.iloc[0]["maze_name"]
        movs = grupo_df.sort_values("move_idx").to_dict("records")
        
        for i in range(len(movs) - 1):
            s = (maze, int(movs[i]["x"]), int(movs[i]["y"]))
            a = movs[i + 1]["direction"]
            if a not in ("START", "STAY", "UNKNOWN"):
                contagens[(grp, maze, s)][a] += 1

    politicas = {}
    for (grp, maze, s), ac in contagens.items():
        total = sum(ac.values())
        politicas[(grp, maze, s)] = {a: n / total for a, n in ac.items()}

    return politicas

def calcular_metricas_inter_jogadores(politicas_grupo: dict, resultados_vi: dict) -> pd.DataFrame:
    """Calcula a Divergência JS e a Taxa de Acordo por grupo populacional."""
    grupos = ["novato", "intermediario", "veterano"]
    resultados = []

    for grp in grupos:
        n_total = n_acordo = 0
        js_vals = []

        for (g, maze, s), dist in politicas_grupo.items():
            if g != grp: continue
            
            pol_oti = resultados_vi.get(maze, {}).get("politica_otima", {})
            if s not in pol_oti: continue

            a_emp = max(dist, key=dist.get)
            a_oti = pol_oti[s]

            n_total += 1
            n_acordo += int(a_emp == a_oti)

            todas = list(set(dist.keys()) | {a_oti})
            p = np.clip(np.array([dist.get(a, 0.0) for a in todas]), 1e-9, None)
            q = np.clip(np.array([1.0 if a == a_oti else 0.0 for a in todas]), 1e-9, None)
            p /= p.sum(); q /= q.sum()
            js_vals.append(float(jensenshannon(p, q, base=2) ** 2))

        ta = n_acordo / n_total if n_total else np.nan
        js = float(np.mean(js_vals)) if js_vals else np.nan

        resultados.append({
            "grupo": grp, "n_estados": n_total,
            "taxa_acordo": round(ta, 4) if not np.isnan(ta) else np.nan,
            "js_divergencia": round(js, 4) if not np.isnan(js) else np.nan
        })

    df = pd.DataFrame(resultados)
    df["_ordem"] = df["grupo"].map({"novato": 0, "intermediario": 1, "veterano": 2})
    return df.sort_values("_ordem").drop(columns="_ordem").reset_index(drop=True)

def analisar_correlacao_continua(df_mov: pd.DataFrame, df_jog: pd.DataFrame, resultados_vi: dict) -> pd.DataFrame:
    """Gera métricas isoladas por UID para correlação estatística de Spearman."""
    uid_para_games = dict(zip(df_jog["uid"], df_jog["total_games"]))
    df = df_mov[df_mov["maze_name"].isin(set(resultados_vi.keys()))]
    cont_uid = defaultdict(lambda: defaultdict(int))
    
    for game_id, grupo_df in df.groupby("game_id"):
        uid = grupo_df.iloc[0]["uid"]
        maze = grupo_df.iloc[0]["maze_name"]
        movs = grupo_df.sort_values("move_idx").to_dict("records")
        for i in range(len(movs) - 1):
            s = (maze, int(movs[i]["x"]), int(movs[i]["y"]))
            a = movs[i + 1]["direction"]
            if a not in ("START", "STAY", "UNKNOWN"):
                cont_uid[(uid, maze, s)][a] += 1

    rows = []
    uid_metricas = defaultdict(lambda: {"n": 0, "acordo": 0, "js": []})
    for (uid, maze, s), ac in cont_uid.items():
        pol_oti = resultados_vi.get(maze, {}).get("politica_otima", {})
        if s not in pol_oti: continue
        
        total = sum(ac.values())
        dist = {a: n / total for a, n in ac.items()}
        a_emp = max(dist, key=dist.get)
        a_oti = pol_oti[s]

        uid_metricas[uid]["n"] += 1
        uid_metricas[uid]["acordo"] += int(a_emp == a_oti)

        todas = list(set(dist.keys()) | {a_oti})
        p = np.clip(np.array([dist.get(a, 0.0) for a in todas]), 1e-9, None)
        q = np.clip(np.array([1.0 if a == a_oti else 0.0 for a in todas]), 1e-9, None)
        p /= p.sum(); q /= q.sum()
        uid_metricas[uid]["js"].append(float(jensenshannon(p, q, base=2) ** 2))

    for uid, m in uid_metricas.items():
        if m["n"] >= 5:
            rows.append({
                "uid": uid, "total_games": uid_para_games.get(uid, np.nan),
                "taxa_acordo": m["acordo"] / m["n"], "js_medio": float(np.mean(m["js"]))
            })

    df_uid = pd.DataFrame(rows).dropna()
    r_js, p_js = stats.spearmanr(df_uid["total_games"], df_uid["js_medio"])
    r_ta, p_ta = stats.spearmanr(df_uid["total_games"], df_uid["taxa_acordo"])
    
    plotar_correlacao_continua(df_uid, r_js, p_js, r_ta, p_ta)
    return df_uid
