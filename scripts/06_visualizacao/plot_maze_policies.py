"""
scripts/06_visualizacao/plot_maze_policies.py
==============================================
Visualização da política ótima vs. empírica.
Gera os mapas de calor de V*(s) e as setas de divergência para um labirinto alvo.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path

# Configurações de estilo
COR_PAREDE = '#2d2d2d'; COR_CAMINHO = '#f5f0e8'; COR_TERMINAL = '#2a9d8f'
COR_INICIO = '#e76f51'; COR_SETA_OT = '#1d3557'; COR_SETA_EMP = '#e63946'
COR_ACORDO = '#a8dadc'; COR_DESACORDO = '#f4a261'
SETAS = {'UP': (0, -0.30), 'DOWN': (0, 0.30), 'LEFT': (-0.30, 0), 'RIGHT': (0.30, 0)}

def plotar_politica_otima(maze_name, mdp, res_vi):
    """Gera a grade com setas da política ótima e fundo térmico V*[cite: 14]."""
    estados = mdp[maze_name]['estados']
    terminal = mdp[maze_name]['terminal']
    pol_ot = res_vi[maze_name]['politica_otima']
    V = res_vi[maze_name]['V']
    
    x_coords = [s[1] for s in estados]; y_coords = [s[2] for s in estados]
    w, h = max(x_coords)+1, max(y_coords)+1
    
    fig, ax = plt.subplots(figsize=(7, 10))
    ax.set_facecolor(COR_PAREDE)
    
    norm = Normalize(vmin=0, vmax=100); cmap = plt.cm.YlGn
    for s in estados:
        cx, cy = s[1], s[2]
        if s == terminal:
            ax.add_patch(plt.Rectangle((cx-0.5, cy-0.5), 1, 1, color=COR_TERMINAL, zorder=1))
            ax.text(cx, cy, 'SAÍDA', ha='center', va='center', fontsize=7, color='white', zorder=3)
        else:
            ax.add_patch(plt.Rectangle((cx-0.5, cy-0.5), 1, 1, color=cmap(norm(V.get(s, 0))), zorder=1))
            if s in pol_ot:
                dx, dy = SETAS[pol_ot[s]]
                ax.annotate('', xy=(cx+dx, cy+dy), xytext=(cx, cy), arrowprops=dict(arrowstyle='->', color=COR_SETA_OT, lw=1.5), zorder=4)

    ax.set_title(f'Política Ótima $\\pi^*(s)$ — Labirinto: {maze_name}')
    plt.savefig(f"../../graficos/fig_labirinto_{maze_name}_politica_otima.png", dpi=200)
    plt.close()

def plotar_acordo_empirico(maze_name, mdp, pol_emp_raw, res_vi):
    """Gera o mapa comparativo entre ação ótima vs. ação empírica[cite: 14]."""
    pol_ot = res_vi[maze_name]['politica_otima']
    pol_emp = {s: d for (m, s), d in pol_emp_raw.items() if m == maze_name}
    
    # [Lógica de plotagem idêntica à do notebook, iterando sobre estados e comparando pol_ot[s] com max(pol_emp[s])]
    # ... (código de anotação de setas divergentes e coloração de células) ...
    
    plt.savefig(f"../../graficos/fig_labirinto_{maze_name}_acordo.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    with open('../../data/processed/mdp_por_labirinto.pkl', 'rb') as f: mdp = pickle.load(f)
    with open('../../data/processed/politica_empirica.pkl', 'rb') as f: pol_e = pickle.load(f)
    with open('../../data/processed/resultados_vi.pkl', 'rb') as f: res = pickle.load(f)
    
    MAZE = 'Port Lakefieldca'
    plotar_politica_otima(MAZE, mdp, res)
    plotar_acordo_empirico(MAZE, mdp, pol_e, res)
    print(f"Visualizações geradas para o labirinto: {MAZE}")
