"""
scripts/02_mdp/reward_sensitivity.py
====================================
Avalia a estabilidade da política extraída frente a variações nos 
parâmetros da Função de Recompensa (R_SAIDA, R_PASSO, R_BACKTRACK).
"""

import pandas as pd
from collections import defaultdict
from scripts.02_mdp.value_iteration import value_iteration
from scripts.02_mdp.policy_comparison import comparar_politicas
from utils.plots import plotar_sensibilidade_recompensas

GRADE_SENSIBILIDADE = {
    "base":          (100,  -1,  -5),
    "penalidade_2x": (100,  -2, -10),
    "penalidade_0x": (100,  -1,  -1),
    "saida_menor":   ( 10,  -1,  -5),
    "saida_maior":   (500,  -1,  -5),
}

def _recalcular_R(mdp_por_labirinto: dict, counts: dict, r_saida: float, r_passo: float, r_bt: float) -> dict:
    """Reconstrói a matriz R baseando-se nas contagens de transição empíricas absolutas."""
    mdp_mod = {}
    for maze, mdp in mdp_por_labirinto.items():
        T, terminal = mdp["T"], mdp["terminal"]
        R_novo = defaultdict(lambda: defaultdict(float))

        for s, acoes in T.items():
            for a, transicoes in acoes.items():
                raw = counts.get((maze, s, a), {})
                
                for s2, prob in transicoes.items():
                    n_s2 = raw.get(s2, 0)
                    frac_bt = (raw.get("_bt_" + str(s2), 0) / n_s2) if n_s2 > 0 else 0.0

                    if s2 == terminal: recomp = r_saida
                    elif frac_bt > 0.5: recomp = r_bt
                    else: recomp = r_passo

                    R_novo[s][a] += prob * recomp

        mdp_copia = dict(mdp)
        mdp_copia["R"] = dict(R_novo)
        mdp_mod[maze] = mdp_copia
        
    return mdp_mod

def rodar_sensibilidade_recompensa(mdp_por_labirinto: dict, politica_emp: dict, counts: dict, gamma: float = 0.95) -> pd.DataFrame:
    """Executa o pipeline de simulação para o grid de hiperparâmetros de recompensa."""
    resultados_sens = {}

    for nome, (r_saida, r_passo, r_bt) in GRADE_SENSIBILIDADE.items():
        mdp_mod = _recalcular_R(mdp_por_labirinto, counts, r_saida, r_passo, r_bt)
        
        vi_local = {}
        for maze, mdp in mdp_mod.items():
            V, pol = value_iteration(mdp, gamma=gamma)
            vi_local[maze] = {"V": V, "politica_otima": pol}

        df_cen = comparar_politicas(politica_emp, vi_local, mdp_mod)
        resultados_sens[nome] = df_cen[["maze_name", "js_divergencia"]].copy().rename(columns={"js_divergencia": f"js_{nome}"})

    df_sens = resultados_sens["base"].copy()
    for nome, df_cen in resultados_sens.items():
        if nome != "base":
            df_sens = df_sens.merge(df_cen, on="maze_name", how="outer")

    plotar_sensibilidade_recompensas(df_sens)
    df_sens.to_csv("sensibilidade_recompensas.csv", index=False)
    print("[Sensibilidade] Análise de Recompensas de Design concluída.")
    return df_sens
