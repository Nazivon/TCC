"""
scripts/02_mdp/run_mdp_analysis.py
==================================
Script Orquestrador (Entry Point) para modelagem do MDP.
Executa a construção das matrizes, a convergência via Value Iteration 
e toda a bateria de testes de sensibilidade em lote.
"""

from pathlib import Path
from utils.io import carregar_mazes_validos, carregar_ids_sucesso, salvar_resultados_mdp
from scripts.02_mdp.build_empirical_mdp import processar_movimentos, montar_mdp, montar_politica_empirica
from scripts.02_mdp.value_iteration import rodar_vi_todos
from scripts.02_mdp.policy_comparison import comparar_politicas
from scripts.02_mdp.reward_sensitivity import rodar_sensibilidade_recompensa
from scripts.02_mdp.gamma_sensitivity import rodar_sensibilidade_gamma
from scripts.02_mdp.empirical_reward import carregar_scores_partidas, construir_scores_por_tripla, rodar_sensibilidade_R_empirica

# Configuração de Diretórios Relativos
PASTA_DADOS = Path("../../data")
ARQ_PARTIDAS = PASTA_DADOS / "partidas.json"
ARQ_MOVIMENTOS = PASTA_DADOS / "movimentos.json"
ARQ_MAZES = PASTA_DADOS / "mazes.json"
PASTA_SAIDA = Path("../../")

if __name__ == "__main__":
    print("=" * 60)
    print("  TCC Escapismo — Modelagem e Análise MDP")
    print("=" * 60)

    # 1. Ingestão e Processamento
    mazes_validos = carregar_mazes_validos(ARQ_MAZES)
    ids_sucesso = carregar_ids_sucesso(ARQ_PARTIDAS)
    counts, politica_raw, terminais = processar_movimentos(mazes_validos, ids_sucesso, ARQ_MOVIMENTOS)
    
    # 2. Estruturação do MDP e Política Empírica
    mdp_por_labirinto = montar_mdp(counts, terminais, mazes_validos)
    politica_emp = montar_politica_empirica(politica_raw, mazes_validos)

    # 3. Solução Ótima (Value Iteration)
    print("\n[Execução] Rodando Value Iteration (gamma=0.95)...")
    resultados_vi = rodar_vi_todos(mdp_por_labirinto, gamma=0.95)

    # 4. Avaliação de Divergência
    print("\n[Execução] Computando Divergência de Jensen-Shannon...")
    df_resumo = comparar_politicas(politica_emp, resultados_vi, mdp_por_labirinto)

    # 5. Pipeline de Sensibilidade (Design vs IRL)
    print("\n[Execução] Iniciando Bateria de Sensibilidade...")
    
    # 5.1 Recompensa Estática e Fator de Desconto
    df_sens_recompensa = rodar_sensibilidade_recompensa(mdp_por_labirinto, politica_emp, counts)
    df_sens_gamma = rodar_sensibilidade_gamma(mdp_por_labirinto, politica_emp)
    
    # 5.2 Recompensa Observacional (Reward Shaping)
    scores_partidas = carregar_scores_partidas(ARQ_PARTIDAS)
    scores_tripla = construir_scores_por_tripla(ARQ_MOVIMENTOS, mazes_validos, ids_sucesso, scores_partidas)
    df_sens_irl = rodar_sensibilidade_R_empirica(mdp_por_labirinto, politica_emp, scores_tripla)

    # 6. Exportação Global
    salvar_resultados_mdp(PASTA_SAIDA, mdp_por_labirinto, politica_emp, resultados_vi, df_resumo)
    print("\n✓ Pipeline MDP finalizado e estruturado.")
