"""
utils/plots.py
==============
Centraliza a geração e exportação de gráficos do projeto.
Mantém os scripts analíticos focados apenas na lógica matemática.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
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

# Para as Análises de Complexidade Estrutural

# Paleta global (manter coesa com o restante do projeto)
COR_ACORDO = "#457b9d"; COR_JS = "#e63946"; COR_NEUTRA = "#6c757d"; COR_CH = "#2a9d8f"

def plotar_dispersao_ch_desvio(df: pd.DataFrame) -> None:
    """Dispersão entre Score de Terada e as métricas de divergência[cite: 8]."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, col_y, cor_y, ylabel, titulo in [(axes[0], "js_divergencia", COR_JS, "JS-divergência (empírica ∥ ótima)", "(a) Complexidade Híbrida × JS-divergência"),
                                             (axes[1], "taxa_acordo", COR_ACORDO, "Taxa de acordo (empírica == ótima)", "(b) Complexidade Híbrida × Taxa de acordo")]:
        r_val, p_val = stats.spearmanr(df["complexidade_hibrida"], df[col_y])
        ax.scatter(df["complexidade_hibrida"], df[col_y], color=cor_y, alpha=0.60, s=36, edgecolors="white", linewidths=0.4)
        
        mask = ~(np.isnan(df["complexidade_hibrida"]) | np.isnan(df[col_y]))
        m, b, *_ = stats.linregress(df["complexidade_hibrida"][mask], df[col_y][mask])
        xr = np.linspace(df["complexidade_hibrida"][mask].min(), df["complexidade_hibrida"][mask].max(), 200)
        ax.plot(xr, m * xr + b, color="black", lw=1.5, ls="--", label=f"Spearman r = {r_val:.3f} (p = {p_val:.4f})")
        
        ax.set_xlabel("Score de Complexidade Híbrida (0–100)"); ax.set_ylabel(ylabel); ax.set_title(titulo)
        ax.legend()
        if col_y == "taxa_acordo": ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / "fig_ch_scatter_ch_vs_desvio.png", dpi=200)
    plt.close()

def plotar_histogramas_desvio(df: pd.DataFrame) -> None:
    """Distribuição geral das métricas de desvio nos labirintos[cite: 9]."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ta_media = df["taxa_acordo"].mean()
    js_media = df["js_divergencia"].mean()

    ax = axes[0]
    ax.hist(df["taxa_acordo"], bins=22, color=COR_ACORDO, alpha=0.8, edgecolor="white")
    ax.axvline(ta_media, color="black", lw=1.8, ls="--", label=f"média = {ta_media:.3f} ({ta_media:.1%})")
    ax.set_xlabel("Taxa de acordo (ação empírica == ação ótima)"); ax.set_ylabel("Número de labirintos")
    ax.set_title("(a) Taxa de acordo"); ax.legend()
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    ax2 = axes[1]
    ax2.hist(df["js_divergencia"], bins=22, color=COR_JS, alpha=0.8, edgecolor="white")
    ax2.axvline(js_media, color="black", lw=1.8, ls="--", label=f"média = {js_media:.4f}")
    ax2.set_xlabel("Divergência JS (0 = idênticas, 1 = máximo)"); ax2.set_ylabel("Número de labirintos")
    ax2.set_title("(b) Divergência JS"); ax2.legend()

    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / "fig_51_histogramas.png", dpi=200)
    plt.close()

def plotar_js_vs_complexidade(df: pd.DataFrame) -> None:
    """Dispersão e Boxplot comparando Tortuosidade e Nível com a divergência JS[cite: 9]."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    sc = ax.scatter(df["tortuosidade"], df["js_divergencia"], c=df["difficult_level"], cmap="RdYlGn_r",
                    s=df["total_games"] / df["total_games"].max() * 200 + 20, alpha=0.75, edgecolors="white", linewidths=0.4)
    x, y = df["tortuosidade"].dropna(), df["js_divergencia"].dropna()
    idx = x.index.intersection(y.index)
    m, b, r, p, _ = stats.linregress(x[idx], y[idx])
    xr = np.linspace(x.min(), x.max(), 100)
    ax.plot(xr, m * xr + b, color="crimson", lw=1.5, ls="--", label=f"r={r:.2f}, p={p:.3f}")
    ax.set_xlabel("Tortuosidade (moves / grid)"); ax.set_ylabel("JS-divergência média")
    ax.set_title("Tortuosidade vs. Desvio"); ax.legend()
    plt.colorbar(sc, ax=ax, label="Nível de dificuldade")

    ax2 = axes[1]
    niveis = sorted(df["difficult_level"].dropna().unique())
    dados = [df[df["difficult_level"] == n]["js_divergencia"].dropna().values for n in niveis]
    bp = ax2.boxplot(dados, patch_artist=True, notch=False)
    cores = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(niveis)))
    for patch, cor in zip(bp["boxes"], cores): patch.set_facecolor(cor); patch.set_alpha(0.75)
    ax2.set_xticks(range(1, len(niveis) + 1)); ax2.set_xticklabels([f"Nível {n}" for n in niveis])
    ax2.set_ylabel("JS-divergência média"); ax2.set_title("Desvio por nível de dificuldade")

    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / "fig1_js_vs_complexidade.png", dpi=200)
    plt.close()

def plotar_heatmap_correlacoes(df_corr: pd.DataFrame) -> None:
    """Heatmap das correlações de Spearman (Métricas vs. Desvio)[cite: 9]."""
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [m.split("(")[0].strip() for m in df_corr["métrica"]]
    valores = df_corr[["r_JS", "r_desvio"]].values
    im = ax.imshow(valores.T, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["r JS-divergência", "r Desvio (%)"])
    ax.set_title("Correlações de Spearman: métricas estruturais vs. desvio humano")

    for i in range(len(labels)):
        for j in range(2):
            v = valores[i, j]
            sig = df_corr.iloc[i]["sig_JS" if j == 0 else "sig_desvio"]
            txt = f"{v:.2f}{'*' if sig == '✓' else ''}"
            ax.text(i, j, txt, ha="center", va="center", fontsize=8, color="white" if abs(v) > 0.5 else "black")

    plt.colorbar(im, ax=ax, label="Correlação de Spearman")
    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / "fig2_heatmap_correlacoes.png", dpi=200)
    plt.close()

def plotar_desvio_vs_moves(df: pd.DataFrame) -> None:
    """Relação entre média de movimentos e taxa de desvio (%)[cite: 9]."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(df["avg_moves"], df["desvio_pct"] * 100, c=df["js_divergencia"], cmap="plasma", s=60, alpha=0.8, edgecolors="white")
    x, y = df["avg_moves"].dropna(), (df["desvio_pct"] * 100).dropna()
    idx = x.index.intersection(y.index)
    m, b, r, *_ = stats.linregress(x[idx], y[idx])
    xr = np.linspace(x.min(), x.max(), 100)
    ax.plot(xr, m * xr + b, color="black", lw=1.5, ls="--", label=f"Spearman r≈{r:.2f}")
    
    ax.set_xlabel("Média de movimentos por partida"); ax.set_ylabel("Taxa de desvio humano (%)")
    ax.set_title("Comprimento do caminho vs. desvio da política ótima"); ax.legend()
    plt.colorbar(sc, ax=ax, label="JS-divergência")
    
    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / "fig3_desvio_vs_avg_moves.png", dpi=200)
    plt.close()

def plotar_ranking_desvio(df: pd.DataFrame) -> None:
    """Gráfico de barras dos 20 labirintos com maior divergência JS[cite: 9]."""
    top20 = df.nlargest(20, "js_divergencia").sort_values("js_divergencia")
    fig, ax = plt.subplots(figsize=(9, 7))
    cores = plt.cm.RdYlGn_r((top20["difficult_level"] - 1) / 4)
    bars = ax.barh(top20["maze_name"], top20["js_divergencia"], color=cores, edgecolor="white", linewidth=0.4)

    for bar, nivel in zip(bars, top20["difficult_level"]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2, f"Nv.{int(nivel)}", va="center", fontsize=8, color="gray")

    ax.set_xlabel("JS-divergência média (empírica ∥ ótima)")
    ax.set_title("Top 20 labirintos com maior desvio humano")
    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / "fig4_ranking_desvio.png", dpi=200)
    plt.close()

# Adicionar a utils/plots.py
def plotar_estacionariedade(agg: pd.DataFrame) -> tuple:
    """Gera gráfico da tendência temporal do tempo por movimento[cite: 12]."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    x = agg["move_idx"].values
    
    # Painel superior
    axes[0].plot(x, agg["media"], color=COR_JS, lw=1.5, label="Média")
    axes[0].plot(x, agg["mediana"], color=COR_ACORDO, lw=1.5, ls="--", label="Mediana")
    axes[0].fill_between(x, agg["media"]-agg["dp"], agg["media"]+agg["dp"], alpha=0.15, color=COR_JS)
    
    slope, intercept, r, p, _ = stats.linregress(x, agg["media"])
    axes[0].plot(x, slope * x + intercept, color="black", lw=1, ls=":", label=f"Tendência (r={r:.3f}, p={p:.4f})")
    axes[0].legend(); axes[0].set_ylabel("Tempo (s)")
    
    # Painel inferior
    axes[1].bar(x, agg["n"]/1000, color=COR_NEUTRA, alpha=0.6)
    axes[1].set_xlabel("Índice do movimento"); axes[1].set_ylabel("Observações (x1.000)")
    
    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / "fig_estacionariedade.png", dpi=150)
    plt.close()
    return slope, p

def plotar_treino_vs_teste(ta_tr, js_tr, ta_te, js_te) -> None:
    """Visualização da generalização do modelo[cite: 11]."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for i, (vals, label, cor) in enumerate([((ta_tr*100, ta_te*100), "Taxa de acordo (%)", COR_ACORDO), 
                                            ((js_tr, js_te), "JS-divergência", COR_JS)]):
        bars = ax[i].bar(["Treino", "Teste"], vals, color=cor, alpha=0.7)
        ax[i].set_ylabel(label)
        for b, v in zip(bars, vals): ax[i].text(b.get_x()+b.get_width()/2, v+max(vals)*0.02, f"{v:.3f}", ha="center")
    plt.savefig(PASTA_GRAFICOS / "fig_backtest_treino_vs_teste.png", dpi=200)
    plt.close()
