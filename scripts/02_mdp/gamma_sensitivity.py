"""
scripts/02_mdp/gamma_sensitivity.py
===================================
Estuda o impacto do Fator de Desconto Temporal (γ) na convergência 
da política ótima e no nível de acordo com as escolhas empíricas.
"""

import pandas as pd
from scripts.02_mdp.value_iteration import value_iteration
from scripts.02_mdp.policy_comparison import comparar_politicas
from utils.plots import plotar_sensibilidade_gamma

GRADE_GAMMA = [0.50, 0.70, 0.90, 0.95, 0.99]

def rodar_sensibilidade_gamma(mdp_por_labirinto: dict, politica_emp: dict) -> pd.DataFrame:
    """Mede a robustez do acordo entre humanos e o modelo sob diferentes horizontes de tempo."""
    resumo_gamma = []
 
    for gamma in GRADE_GAMMA:
        vi_local = {}
        for maze, mdp in mdp_por_labirinto.items():
            V, pol = value_iteration(mdp, gamma=gamma)
            vi_local[maze] = {"V": V, "politica_otima": pol}
 
        df_cen = comparar_politicas(politica_emp, vi_local, mdp_por_labirinto)
 
        resumo_gamma.append({
            "gamma": gamma,
            "js_medio": round(df_cen["js_divergencia"].mean(), 4),
            "taxa_acordo": round(df_cen["taxa_acordo"].mean(), 4),
            "n_labirintos": len(df_cen),
        })
 
    df_gamma = pd.DataFrame(resumo_gamma)
    plotar_sensibilidade_gamma(df_gamma)
    df_gamma.to_csv("sensibilidade_gamma.csv", index=False)
    print("[Sensibilidade] Análise de Fator de Desconto (Gamma) concluída.")
    
    return df_gamma
