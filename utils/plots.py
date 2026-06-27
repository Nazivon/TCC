"""
utils/plots.py
==============
Centraliza a geração e exportação de gráficos do projeto.
Mantém os scripts analíticos focados apenas na lógica matemática.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Garante que a pasta de gráficos exista
PASTA_GRAFICOS = Path("graficos")
PASTA_GRAFICOS.mkdir(exist_ok=True)

def plotar_sensibilidade_recompensas(df_sens: pd.DataFrame) -> None:
    """Gera boxplot comparando a JS-divergência entre cenários de recompensa de design."""
    cols = [c for c in df_sens.columns if c.startswith("js_")]
    labels = [c.replace("js_", "") for c in cols]
    dados = [df_sens[c].dropna().values for c in cols]
 
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(dados, patch_artist=True, notch=False)
    cores = plt.cm.tab10(np.linspace(0, 0.6, len(cols)))
    
    for patch, cor in zip(bp["boxes"], cores):
        patch.set_facecolor(cor)
        patch.set_alpha(0.75)
 
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("JS-divergência (0=ótimo, 1=máximo desvio)")
    ax.set_title("Sensibilidade: Desvio Humano sob Diferentes Parâmetros de Recompensa", fontweight="bold")
    ax.axhline(df_sens[cols[0]].mean(), color="gray", ls="--", lw=1, label="Média (Cenário Base)")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / "fig_sensibilidade_recompensas.png", dpi=300, bbox_inches="tight")
    plt.close()

def plotar_sensibilidade_gamma(df_gamma: pd.DataFrame) -> None:
    """Gera gráfico de linhas para JS-divergência e taxa de acordo em função de gamma."""
    fig, ax1 = plt.subplots(figsize=(8, 4))
    cor_js, cor_acordo = "#e63946", "#457b9d"
 
    ax1.plot(df_gamma["gamma"], df_gamma["js_medio"], marker="o", color=cor_js, lw=2, label="JS-Divergência")
    ax1.set_xlabel("Fator de Desconto temporal (γ)")
    ax1.set_ylabel("JS-divergência Média", color=cor_js)
    ax1.tick_params(axis="y", labelcolor=cor_js)
    ax1.set_ylim(0, max(df_gamma["js_medio"]) * 1.4)
 
    ax2 = ax1.twinx()
    ax2.plot(df_gamma["gamma"], df_gamma["taxa_acordo"] * 100, marker="s", color=cor_acordo, lw=2, ls="--", label="Taxa de Acordo (%)")
    ax2.set_ylabel("Taxa de Acordo (%)", color=cor_acordo)
    ax2.tick_params(axis="y", labelcolor=cor_acordo)
    ax2.set_ylim(0, 100)
 
    linhas = ax1.get_lines() + ax2.get_lines()
    ax1.legend(linhas, [l.get_label() for l in linhas], loc="lower left")
    ax1.set_title("Sensibilidade ao Fator de Desconto γ", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / "fig_sensibilidade_gamma.png", dpi=300, bbox_inches="tight")
    plt.close()

def plotar_sensibilidade_R_empirica(df: pd.DataFrame) -> None:
    """Gráfico de barras duplas avaliando configurações de recompensa empírica (IRL)."""
    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    largura = 0.35
    cor_js, cor_acordo = "#e63946", "#457b9d"

    barras_js = ax1.bar(x - largura / 2, df["js_medio"], largura, color=cor_js, alpha=0.8, label="JS-Divergência")
    ax1.set_ylabel("JS-divergência Média", color=cor_js)
    ax1.tick_params(axis="y", labelcolor=cor_js)
    ax1.set_ylim(0, max(df["js_medio"]) * 1.5)

    ax2 = ax1.twinx()
    barras_ac = ax2.bar(x + largura / 2, df["taxa_acordo"] * 100, largura, color=cor_acordo, alpha=0.8, label="Taxa de Acordo (%)")
    ax2.set_ylabel("Taxa de Acordo (%)", color=cor_acordo)
    ax2.tick_params(axis="y", labelcolor=cor_acordo)
    ax2.set_ylim(0, 100)

    ax1.set_xticks(x)
    ax1.set_xticklabels(df["configuracao"], rotation=15, ha="right")
    ax1.set_xlabel("Configuração de Pesos (Score Composto)")
    
    linhas = [barras_js, barras_ac]
    ax1.legend(linhas, [l.get_label() for l in linhas], loc="upper right")
    ax1.set_title("Sensibilidade: Recompensa Empírica via Score Composto", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / "fig_sensibilidade_R_empirica.png", dpi=300, bbox_inches="tight")
    plt.close()
