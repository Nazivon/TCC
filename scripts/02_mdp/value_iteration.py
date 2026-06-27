"""
scripts/02_mdp/value_iteration.py
=================================
Implementação do algoritmo Value Iteration para resolução do MDP.
Extrai a função de valor V(s) ótima e a política estável associada.
"""

def value_iteration(mdp: dict, gamma: float = 0.95, theta: float = 1e-6, max_iter: int = 1000) -> tuple:
    """
    Itera sobre a Equação de Otimalidade de Bellman até a convergência.
    
    Parâmetros
    ----------
    mdp : dict
        Estrutura contendo S, A, T (transições) e R (recompensas).
    gamma : float
        Fator de desconto temporal.
    theta : float
        Critério de parada para convergência.
    
    Retorna
    -------
    tuple (V, pol)
        Função de valor mapeada por estado e política determinística ótima.
    """
    T, R = mdp["T"], mdp["R"]
    estados, terminal = mdp["estados"], mdp["terminal"]
    V = {s: 0.0 for s in estados}

    for it in range(max_iter):
        delta = 0.0
        V_novo = V.copy()

        for s in estados:
            if s == terminal or s not in T:
                continue
            
            melhor = max(
                R.get(s, {}).get(a, -1.0) + 
                gamma * sum(prob * V.get(s2, 0.0) for s2, prob in trans.items())
                for a, trans in T[s].items()
            )
            delta = max(delta, abs(melhor - V[s]))
            V_novo[s] = melhor

        V = V_novo
        if delta < theta:
            break

    pol = {}
    for s in estados:
        if s == terminal or s not in T:
            continue
        pol[s] = max(
            T[s],
            key=lambda a: (
                R.get(s, {}).get(a, -1.0) + 
                gamma * sum(prob * V.get(s2, 0.0) for s2, prob in T[s][a].items())
            )
        )
        
    return V, pol

def rodar_vi_todos(mdp_por_labirinto: dict, gamma: float = 0.95) -> dict:
    """Orquestra a resolução do Value Iteration para todos os ambientes de teste."""
    resultados = {}
    for maze, mdp in mdp_por_labirinto.items():
        V, pol = value_iteration(mdp, gamma=gamma)
        resultados[maze] = {"V": V, "politica_otima": pol}
    print(f"[VI] Convergência alcançada para {len(resultados)} labirintos.")
    return resultados
