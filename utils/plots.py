"""
utils/plots.py
==============
Centraliza a geração e exportação de gráficos do projeto.
Mantém os scripts analíticos focados apenas na lógica matemática.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

# Para as Análises de Aprendizado
def plotar_aprendizado_grupos(df_grupos: pd.DataFrame) -> None:
    grupos, js_vals, ta_vals = df_grupos["grupo"].tolist(), df_grupos["js_divergencia"].tolist(), (df_grupos["taxa_acordo"] * 100).tolist()
    x, w = np.arange(len(grupos)), 0.35
    fig, ax1 = plt.subplots(figsize=(8, 5))

    bars1 = ax1.bar(x - w/2, js_vals, w, color="#e63946", alpha=0.8, label="JS-Divergência")
    ax1.set_ylabel("JS-divergência (0=ótimo)", color="#e63946")
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + w/2, ta_vals, w, color="#457b9d", alpha=0.8, label="Taxa de Acordo (%)")
    ax2.set_ylabel("Taxa de Acordo (%)", color="#457b9d")

    ax1.set_xticks(x); ax1.set_xticklabels(["Novatos", "Intermediários", "Veteranos"])
    ax1.set_title("Desvio da Política Ótima por Nível de Experiência")
    ax1.legend([bars1, bars2], ["JS-divergência", "Taxa de acordo (%)"], loc="upper right")
    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / "fig_aprendizado_grupos.png", dpi=300)
    plt.close()

def plotar_correlacao_continua(df_uid: pd.DataFrame, r_js, p_js, r_ta, p_ta) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Relação entre Experiência e Desvio da Política")

    ax = axes[0]
    ax.scatter(df_uid["total_games"], df_uid["js_medio"], alpha=0.4, s=20, color="#e63946")
    m, b, *_ = stats.linregress(df_uid["total_games"], df_uid["js_medio"])
    xr = np.linspace(df_uid["total_games"].min(), df_uid["total_games"].max(), 100)
    ax.plot(xr, m * xr + b, color="black", lw=1.5, ls="--")
    ax.set_title(f"Spearman r = {r_js:.3f} (p = {p_js:.4f})")

    ax2 = axes[1]
    ax2.scatter(df_uid["total_games"], df_uid["taxa_acordo"] * 100, alpha=0.4, s=20, color="#457b9d")
    m2, b2, *_ = stats.linregress(df_uid["total_games"], df_uid["taxa_acordo"] * 100)
    ax2.plot(xr, m2 * xr + b2, color="black", lw=1.5, ls="--")
    ax2.set_title(f"Spearman r = {r_ta:.3f} (p = {p_ta:.4f})")
    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / "fig_aprendizado_correlacao.png", dpi=300)
    plt.close()

def plotar_intraplayer_tercis(df_ind: pd.DataFrame, n_jogadores: int, p_js: float, p_jsp: float, p_ta: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Evolução do desvio comportamental (n = {n_jogadores})")
    
    paineis = [(axes[0], "js_medio", "#e63946", "JS Bruto", p_js), 
               (axes[1], "js_pond", "#e63946", "JS Ponderado", p_jsp), 
               (axes[2], "taxa_acordo", "#457b9d", "Taxa Acordo", p_ta)]
    
    for ax, col, cor, titulo, p_val in paineis:
        dados = [df_ind[df_ind["tercio"] == t][col].values for t in [1, 2, 3]]
        bp = ax.boxplot(dados, patch_artist=True)
        for patch in bp["boxes"]: patch.set_facecolor(cor); patch.set_alpha(0.85)
        ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["Terço 1", "Terço 2", "Terço 3"])
        ax.set_title(f"{titulo} (p = {p_val:.4f})")
        
    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / "fig_intra_tercis.png", dpi=300)
    plt.close()

def plotar_intraplayer_delta(js_t1, js_t3, jsp_t1, jsp_t3, ta_t1, ta_t3) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, delta, melhora, titulo in [(axes[0], js_t3 - js_t1, js_t3 < js_t1, "Δ JS Bruto"),
                                       (axes[1], jsp_t3 - jsp_t1, jsp_t3 < jsp_t1, "Δ JS Ponderado"),
                                       (axes[2], ta_t3 - ta_t1, ta_t3 > ta_t1, "Δ Taxa Acordo")]:
        bins = np.linspace(delta.min(), delta.max(), 25)
        ax.hist(delta[melhora], bins=bins, color="#2a9d8f", alpha=0.8, label="Melhora")
        ax.hist(delta[~melhora], bins=bins, color="#e76f51", alpha=0.8, label="Piora/Estável")
        ax.axvline(0, color="black", ls="--")
        ax.set_title(titulo)
        ax.legend()
    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / "fig_intra_delta.png", dpi=300)
    plt.close()
