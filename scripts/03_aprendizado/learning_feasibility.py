"""
scripts/03_aprendizado/learning_feasibility.py
===========================================
Módulo de Análise de Viabilidade de Aprendizado.
Verifica o volume de reincidência e o total de partidas por jogador 
para justificar o poder estatístico das análises longitudinais (intra-jogador).
"""

import json
import pandas as pd
from pathlib import Path

def analisar_viabilidade_amostral(arq_partidas: Path, min_partidas: int = 5, min_repeticoes: int = 3) -> None:
    """
    Avalia a distribuição de partidas e repetições por usuário (UID)[cite: 7].
    
    Parâmetros
    ----------
    arq_partidas : Path
        Caminho para o arquivo partidas.json.
    min_partidas : int
        Ponto de corte para considerar o jogador apto à análise intra-jogador.
    min_repeticoes : int
        Ponto de corte para considerar o jogador reincidente em um mesmo mapa.
    """
    with open(arq_partidas, 'r', encoding='utf-8') as f:
        partidas = json.load(f)

    df = pd.DataFrame(partidas)
    df = df[df['stars'] > 0]  # Considera apenas partidas concluídas com sucesso[cite: 7]

    rep = df.groupby(['uid', 'maze_name']).size().reset_index(name='n_repeticoes')
    jogadores_repetiram = (rep['n_repeticoes'] >= 2).sum()
    jogadores_alta_reincidencia = (rep['n_repeticoes'] >= min_repeticoes).sum()

    print("── Viabilidade de Aprendizado Específico (Mesmo Mapa) ──")
    print(f"Jogadores com >= 2 partidas no mesmo labirinto: {jogadores_repetiram}")
    print(f"Jogadores com >= {min_repeticoes} partidas no mesmo labirinto: {jogadores_alta_reincidencia}\n")

    por_jogador = df.groupby('uid').size()
    print("── Distribuição Global de Partidas por Jogador ──")
    print(por_jogador.describe().round(2))
    print(f"Jogadores com >= {min_partidas} partidas: {(por_jogador >= min_partidas).sum()}")
    print(f"Jogadores com >= 10 partidas: {(por_jogador >= 10).sum()}")

if __name__ == "__main__":
    PASTA_DADOS = Path("../../data")
    analisar_viabilidade_amostral(PASTA_DADOS / "partidas.json")
