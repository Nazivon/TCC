"""
utils/io.py
===========
Módulo de utilitários para operações de entrada e saída (I/O) do pipeline.
Lida com a ingestão de dados brutos (JSON) e persistência de estruturas (Pickle/CSV).
"""

import json
import pickle
from pathlib import Path
import pandas as pd

def carregar_mazes_validos(arq_mazes: Path, min_partidas: int = 30) -> set:
    """
    Retorna o conjunto de nomes de labirintos que possuem volume amostral suficiente.
    Labirintos com poucas partidas apresentam matrizes de transição ruidosas.
    """
    with open(arq_mazes, "r", encoding="utf-8") as f:
        mazes = json.load(f)
    validos = {m["maze_name"] for m in mazes if m["total_games"] >= min_partidas}
    print(f"[I/O] {len(validos)} labirintos com >= {min_partidas} partidas validadas.")
    return validos

def carregar_ids_sucesso(arq_partidas: Path, tempo_limite: int = 600) -> set:
    """
    Filtra e retorna os IDs das partidas que alcançaram a condição de vitória 
    (chegada à saída) dentro do limite de tempo estipulado.
    """
    with open(arq_partidas, "r", encoding="utf-8") as f:
        partidas = json.load(f)
    ids = {p["id"] for p in partidas if p.get("maze_finish_time", 0) < tempo_limite}
    print(f"[I/O] {len(ids)} partidas bem-sucedidas mapeadas.")
    return ids

def converter_para_jsonl(arq_json: Path, arq_jsonl: Path) -> None:
    """
    Converte um array JSON monolítico para o formato JSON Lines (JSONL).
    Procedimento fundamental para permitir a ingestão escalável de dados via streaming.
    """
    if arq_jsonl.exists():
        return
        
    print(f"[I/O] Convertendo {arq_json.name} para formato JSONL...")
    with open(arq_json, "r", encoding="utf-8") as f:
        dados = json.load(f)
    with open(arq_jsonl, "w", encoding="utf-8") as f:
        for obj in dados:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"[I/O] Conversão concluída.")

def salvar_resultados_mdp(pasta_saida: Path, mdp_por_labirinto: dict, 
                          politica_emp: dict, resultados_vi: dict, df_resumo: pd.DataFrame) -> None:
    """Persiste as estruturas computadas do MDP em formato binário e tabular."""
    arqs = {
        "mdp_por_labirinto.pkl": mdp_por_labirinto,
        "politica_empirica.pkl": politica_emp,
        "resultados_vi.pkl":     resultados_vi,
    }
    for nome, obj in arqs.items():
        with open(pasta_saida / nome, "wb") as f:
            pickle.dump(obj, f)

    df_resumo.to_csv(pasta_saida / "resumo_mdp.csv", index=False)
    print("[I/O] Estruturas do MDP exportadas com sucesso (Pickle e CSV).")
