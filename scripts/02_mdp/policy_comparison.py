"""
scripts/02_mdp/policy_comparison.py
===================================
Módulo de comparação entre a política comportamental (empírica) dos 
jogadores e a política ótima (Markoviana) extraída via Value Iteration.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

def comparar_politicas(politica_emp: dict, resultados_vi: dict, mdp_por_labirinto: dict) -> pd.DataFrame:
    """
    Mede o desvio entre a política humana e a ótima utilizando a Divergência de Jensen-Shannon.
    JS ∈ [0, 1] (base 2), onde 0 indica distribuições idênticas e 1 divergência máxima.
    """
    resumo = []
 
    for maze, vi in resultados_vi.items():
        pol_oti = vi["politica_otima"]
        pares = [(maze, s) for (m, s) in politica_emp if m == maze]
        
        n_total = n_acordo = 0
        js_vals = []
 
        for (m, s) in pares:
            if s not in pol_oti:
                continue
 
            dist = politica_emp[(m, s)]
            a_emp = max(dist, key=dist.get)
            a_oti = pol_oti[s]
 
            n_total += 1
            n_acordo += int(a_emp == a_oti)
 
            todas_acoes = list(set(dist.keys()) | {a_oti})
 
            # Distribuição empírica vs ótima (one-hot)
            p = np.clip(np.array([dist.get(a, 0.0) for a in todas_acoes], dtype=float), 1e-9, None)
            p /= p.sum()
            q = np.clip(np.array([1.0 if a == a_oti else 0.0 for a in todas_acoes], dtype=float), 1e-9, None)
            q /= q.sum()
 
            # Jensen-Shannon (quadrado da distância JS)
            js = jensenshannon(p, q, base=2) ** 2
            js_vals.append(float(js))
 
        ta = n_acordo / n_total if n_total else np.nan
        js_medio = float(np.mean(js_vals)) if js_vals else np.nan
 
        resumo.append({
            "maze_name": maze,
            "n_estados": mdp_por_labirinto[maze]["n_estados"],
            "n_comparados": n_total,
            "taxa_acordo": round(ta, 4) if not np.isnan(ta) else np.nan,
            "desvio_pct": round(1 - ta, 4) if not np.isnan(ta) else np.nan,
            "js_divergencia": round(js_medio, 4) if not np.isnan(js_medio) else np.nan,
            "terminal": str(mdp_por_labirinto[maze]["terminal"]),
        })
 
    df = pd.DataFrame(resumo).sort_values("desvio_pct", ascending=False)
    print(f"[Análise JS] Divergência média: {df['js_divergencia'].mean():.3f} | Acordo médio: {df['taxa_acordo'].mean():.1%}")
    return df
