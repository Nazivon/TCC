"""
scripts/03_aprendizado/intraplayer_learning.py
===========================================
Análise de Aprendizado Intra-Jogador (Longitudinal).
Divide as trajetórias dos jogadores ao longo do tempo (tercis) e avalia o 
aumento da aderência à política ótima, com correção baseada na topologia 
de dificuldade do mapa.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from collections import defaultdict
from utils.plots import plotar_intraplayer_tercis, plotar_intraplayer_delta

def atribuir_tercos_cronologicos(df_part: pd.DataFrame, min_partidas: int = 5) -> pd.DataFrame:
    """Fatia cronologicamente as partidas de um jogador em 3 períodos de maturação[cite: 5]."""
    contagem = df_part.groupby("uid")["id"].count()
    uids_validos = set(contagem[contagem >= min_partidas].index)
    df_part = df_part[df_part["uid"].isin(uids_validos)].copy()
    
    def processar_terco(grupo):
        n = len(grupo)
        grupo["tercio"] = pd.cut(range(n), bins=3, labels=[1, 2, 3]).astype(int)
        return grupo

    df_part = df_part.groupby("uid", group_keys=False).apply(processar_terco)
    return df_part

def calcular_metricas_longitudinais(df_mov: pd.DataFrame, mapa_tercio: dict, 
                                    resultados_vi: dict, dificuldade: dict, 
                                    min_estados: int = 10) -> pd.DataFrame:
    """Extrai JS-divergência e Acordo segmentados por terço cronológico do indivíduo[cite: 5]."""
    contagens = defaultdict(lambda: defaultdict(int))
    for game_id, grupo in df_mov.groupby("game_id"):
        info = mapa_tercio.get(game_id)
        if not info: continue
        
        uid, tercio = info["uid"], info["tercio"]
        maze = grupo.iloc[0]["maze_name"]
        movs = grupo.sort_values("move_idx").to_dict("records")
        
        for i in range(len(movs) - 1):
            s = (maze, int(movs[i]["x"]), int(movs[i]["y"]))
            a = movs[i + 1]["direction"]
            if a not in ("START", "STAY", "UNKNOWN"):
                contagens[(uid, tercio, maze, s)][a] += 1

    uid_tercio = defaultdict(lambda: {"n": 0, "acordo": 0, "js": [], "js_pond": [], "pesos": []})
    for (uid, tercio, maze, s), ac in contagens.items():
        pol_oti = resultados_vi.get(maze, {}).get("politica_otima", {})
        if s not in pol_oti: continue

        total = sum(ac.values())
        dist = {a: n / total for a, n in ac.items()}
        a_emp = max(dist, key=dist.get)
        a_oti = pol_oti[s]

        todas = list(set(dist.keys()) | {a_oti})
        p = np.clip(np.array([dist.get(a, 0.0) for a in todas]), 1e-9, None)
        q = np.clip(np.array([1.0 if a == a_oti else 0.0 for a in todas]), 1e-9, None)
        p /= p.sum(); q /= q.sum()
        js = float(jensenshannon(p, q, base=2) ** 2)
        
        peso = dificuldade.get(maze, 0.5)
        m = uid_tercio[(uid, tercio)]
        m["n"] += 1
        m["acordo"] += int(a_emp == a_oti)
        m["js"].append(js)
        m["js_pond"].append(js * peso)
        m["pesos"].append(peso)

    rows = []
    for (uid, tercio), m in uid_tercio.items():
        if m["n"] < min_estados: continue
        soma_pesos = sum(m["pesos"])
        js_pond = sum(m["js_pond"]) / soma_pesos if soma_pesos > 0 else float(np.mean(m["js"]))
        rows.append({
            "uid": uid, "tercio": tercio, "n_estados": m["n"],
            "taxa_acordo": m["acordo"] / m["n"], "js_medio": float(np.mean(m["js"])),
            "js_pond": js_pond
        })

    df_raw = pd.DataFrame(rows)
    uids_completos = df_raw.groupby("uid")["tercio"].nunique().pipe(lambda s: s[s == 3].index)
    return df_raw[df_raw["uid"].isin(uids_completos)].copy()

def testar_hipotese_wilcoxon(df_ind: pd.DataFrame) -> None:
    """Aplica o teste pareado de Wilcoxon para validar significância entre Terço 1 e Terço 3[cite: 5]."""
    t1 = df_ind[df_ind["tercio"] == 1].set_index("uid")
    t3 = df_ind[df_ind["tercio"] == 3].set_index("uid")
    comum = t1.index.intersection(t3.index)

    def wilcoxon_eval(a, b, direction):
        w, p = stats.wilcoxon(a, b, alternative=direction)
        return w, p

    w_js, p_js = wilcoxon_eval(t1.loc[comum, "js_medio"], t3.loc[comum, "js_medio"], "greater")
    w_jsp, p_jsp = wilcoxon_eval(t1.loc[comum, "js_pond"], t3.loc[comum, "js_pond"], "greater")
    w_ta, p_ta = wilcoxon_eval(t1.loc[comum, "taxa_acordo"], t3.loc[comum, "taxa_acordo"], "less")

    plotar_intraplayer_tercis(df_ind, len(comum), p_js, p_jsp, p_ta)
    plotar_intraplayer_delta(t1.loc[comum, "js_medio"], t3.loc[comum, "js_medio"], 
                             t1.loc[comum, "js_pond"], t3.loc[comum, "js_pond"], 
                             t1.loc[comum, "taxa_acordo"], t3.loc[comum, "taxa_acordo"])
