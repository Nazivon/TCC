"""
scripts/02_mdp/build_empirical_mdp.py
=====================================
Responsável pela ingestão de logs de jogabilidade e modelagem empírica 
do Processo de Decisão Markoviano (S, A, P, R).
"""

import pandas as pd
from collections import defaultdict, Counter
from pathlib import Path

def processar_movimentos(mazes_validos: set, ids_sucesso: set, 
                         arq_movimentos: Path, chunk_size: int = 50_000) -> tuple:
    """
    Processa o histórico de movimentações espaço-temporais dos jogadores.
    Garante que a extração de estados se dê estritamente pelas coordenadas
    e sequenciamento de tempo, isolando a construção do modelo da definição 
    posterior de recompensas.
    
    A ingestão ocorre em blocos (chunks) para escalabilidade computacional.
    """
    counts = defaultdict(lambda: defaultdict(int))
    politica = defaultdict(lambda: defaultdict(int))
    terminais_candidatos = defaultdict(Counter)
 
    arq_jsonl = arq_movimentos.with_suffix(".jsonl")
    usa_streaming = arq_jsonl.exists()
 
    if usa_streaming:
        leitor = pd.read_json(arq_jsonl, lines=True, chunksize=chunk_size)
    else:
        df_completo = pd.read_json(arq_movimentos)
        df_completo = (df_completo.sort_values(["game_id", "move_idx"])
                                  .reset_index(drop=True))
        df_completo = df_completo[df_completo["maze_name"].isin(mazes_validos) & 
                                  df_completo["game_id"].isin(ids_sucesso)]
        leitor = (df_completo.iloc[i: i + chunk_size] 
                  for i in range(0, len(df_completo), chunk_size))
 
    for chunk in leitor:
        if usa_streaming:
            chunk = chunk[chunk["maze_name"].isin(mazes_validos)]
            chunk = chunk.sort_values(["game_id", "move_idx"])
 
        for game_id, grupo in chunk.groupby("game_id"):
            if game_id not in ids_sucesso:
                continue # Mitigação de viés de seleção: apenas partidas completas
            
            grupo = grupo.sort_values("move_idx").reset_index(drop=True)
            maze  = grupo.iloc[0]["maze_name"]
            movs  = grupo.to_dict("records")
 
            for i in range(len(movs) - 1):
                m_atual, m_prox = movs[i], movs[i + 1]
 
                s  = (maze, int(m_atual["x"]), int(m_atual["y"]))
                s2 = (maze, int(m_prox["x"]),  int(m_prox["y"]))
                a  = m_prox["direction"]
 
                if a in ("START", "STAY", "UNKNOWN"):
                    continue
 
                e_bt = bool(m_prox.get("is_backtrack", False))
                counts[(maze, s, a)][s2] += 1
                counts[(maze, s, a)]["_bt_" + str(s2)] += int(e_bt)
                politica[(maze, s)][a] += 1
 
            ult = movs[-1]
            s_term = (maze, int(ult["x"]), int(ult["y"]))
            terminais_candidatos[maze][s_term] += 1
 
    terminais = {maze: cand.most_common(1)[0][0] for maze, cand in terminais_candidatos.items() if cand}
    print(f"[MDP Build] {len(counts):,} transições (s,a)->s' modeladas.")
    return counts, politica, terminais


def montar_mdp(counts: dict, terminais: dict, mazes_validos: set, 
               r_saida: float = 100.0, r_passo: float = -1.0, r_backtrack: float = -5.0) -> dict:
    """
    Sintetiza as matrizes de probabilidade de transição P(s'|s,a) e 
    a função de recompensa empírica basal R(s,a) baseada em ineficiências de rota.
    """
    mdp_por_labirinto = {}
    labirintos = {k[0] for k in counts} & mazes_validos

    for maze in sorted(labirintos):
        T = defaultdict(lambda: defaultdict(dict))
        R = defaultdict(lambda: defaultdict(float))
        estados, acoes = set(), set()
        terminal = terminais.get(maze)

        for (m, s, a), next_counts in counts.items():
            if m != maze: continue

            estados.add(s); acoes.add(a)
            real_counts = {k: v for k, v in next_counts.items() if not str(k).startswith("_bt_")}
            bt_counts   = {k: v for k, v in next_counts.items() if str(k).startswith("_bt_")}

            total = sum(real_counts.values())
            if total == 0: continue

            for s2, n in real_counts.items():
                prob = n / total
                T[s][a][s2] = prob
                estados.add(s2)

                frac_bt = bt_counts.get("_bt_" + str(s2), 0) / n 
                
                if s2 == terminal: recomp = r_saida
                elif frac_bt > 0.5: recomp = r_backtrack
                else: recomp = r_passo

                R[s][a] += prob * recomp

        mdp_por_labirinto[maze] = {
            "T": dict(T), "R": dict(R), "estados": estados,
            "acoes": acoes, "terminal": terminal, "n_estados": len(estados)
        }

    return mdp_por_labirinto


def montar_politica_empirica(politica: dict, mazes_validos: set) -> dict:
    """Extrai a distribuição de probabilidade das ações observadas dos dados."""
    politica_emp = {}
    for (maze, s), ac in politica.items():
        if maze not in mazes_validos: continue
        total = sum(ac.values())
        politica_emp[(maze, s)] = {a: n / total for a, n in ac.items()}
    return politica_emp
