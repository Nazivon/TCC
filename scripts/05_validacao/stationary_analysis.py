"""
scripts/05_validacao/stationarity_analysis.py
==============================================
Verifica estabilidade do tempo por movimento ao longo da partida.
Se a tendência linear for nula (p >= 0.05), valida o MDP estacionário[cite: 12].
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from utils.plots import plotar_estacionariedade

ARQ_MOVIMENTOS = Path("../../data/movimentos.json")

def agregar_estacionariedade(max_idx=150):
    df = pd.read_json(ARQ_MOVIMENTOS)
    # Limpeza e filtro
    df = df[df["move_time"] <= df["move_time"].quantile(0.99)]
    df = df[df["move_idx"] <= max_idx]

    agg = df.groupby("move_idx")["move_time"].agg(["mean", "median", "std", "count"]).reset_index()
    agg.columns = ["move_idx", "media", "mediana", "dp", "n"]
    return agg[agg["n"] >= 100].reset_index(drop=True)

if __name__ == "__main__":
    dados = agregar_estacionariedade()
    slope, p = plotar_estacionariedade(dados)
    
    print("── Interpretação ──")
    print(f"Inclinação: {slope:.6f} s/passo | p-valor: {p:.4f}")
    if p >= 0.05: print("Tendência não significativa. MDP estacionário validado. ✓")
