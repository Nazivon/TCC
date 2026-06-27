"""
scripts/01_exploration/exploratory_analysis.py
==============================================
Módulo de Análise Exploratória e Pré-processamento dos Logs.
Realiza limpeza de dados, correção de inconsistências na contabilização 
de tempo (moves_sum vs maze_finish_time), extração de caminhos loop-free 
e cálculo de métricas descritivas (distância, curvas, perplexidade).
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuração de Caminhos
PASTA_DADOS = Path("../../data")
ARQ_MAZE_FINISH = PASTA_DADOS / "maze_finish.json"
PASTA_SAIDA = Path("../../")

# Configurações de Correção de Tempo
LOWER_PERCENTILE = 2.5
UPPER_PERCENTILE = 97.5
TOLERANCIA_DIFERENCA_SEGUNDOS = 1.5
LIMITE_MOVES_SUM_ALTO = 600

# Metadados de Mudanças de Goal (Design do Jogo)
GOALS_MUDANCA = {
    'Be': {
        'goals': [(7, 15), (12, 3)],
        'data_mudanca': pd.to_datetime('2023-08-29')
    },
    'Frostpine Fort': {
        'goals': [(32, 24), (15, 9)],
        'data_mudanca': pd.to_datetime('2023-08-29')
    },
    'Rstonenah Du Wkenarburgh': {
        'goals': [(29, 8), (15, 15)],
        'data_mudanca': pd.to_datetime('2023-08-29')
    }
}


def calcular_soma_tempos(moves_string: str) -> float:
    """Extrai e soma os valores de tempo (segundos) da string codificada de 'moves'."""
    if pd.isna(moves_string) or moves_string == "":
        return 0.0
   
    movimentos = moves_string.rstrip(';').split(';')
    tempos = []
   
    for mov in movimentos:
        if ':' in mov and mov.strip():
            resultado_acao = mov.split(':')[-1].strip()
            if resultado_acao:
                tempo_str = resultado_acao.replace(',', '.')
                try:
                    tempos.append(float(tempo_str))
                except ValueError:
                    pass
    return sum(tempos)

def remover_loops(coords: list) -> list:
    """Remove segmentos de caminho (loops) onde o jogador retornou a uma célula já visitada."""
    vistos = {}
    caminho_limpo = []

    for pos in coords:
        if pos in vistos:
            loop_start = vistos[pos]
            caminho_limpo = caminho_limpo[:loop_start + 1]
            vistos = {p: i for i, p in enumerate(caminho_limpo)}
        else:
            caminho_limpo.append(pos)
            vistos[pos] = len(caminho_limpo) - 1

    return caminho_limpo

def contar_curvas(path: list) -> int:
    """Calcula o número de mudanças de direção (curvas de 90 graus) em um caminho fornecido."""
    if len(path) < 3:
        return 0
    
    curvas = 0
    for i in range(2, len(path)):
        dr1 = path[i-1][0] - path[i-2][0]
        dc1 = path[i-1][1] - path[i-2][1]
        dr2 = path[i][0]   - path[i-1][0]
        dc2 = path[i][1]   - path[i-1][1]
        
        # Incrementa se o vetor de direção muda e não é um backtrack de 180°
        if (dr1, dc1) != (dr2, dc2) and (dr1, dc1) != (-dr2, -dc2):
            curvas += 1
            
    return curvas

def pre_processamento(arq_json: Path) -> pd.DataFrame:
    """Realiza a padronização e limpeza dos dados brutos."""
    df = pd.read_json(arq_json)

    # Padronização de Colunas
    colunas_reparo = ["score", "stars", "maze_finish_time"]
    for col in colunas_reparo:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)

    df = df.astype({
        "score": float,
        "stars": int,
        "maze_finish_time": float
    })
    
    df['trace_id'] = df.index
    df['create_time'] = pd.to_datetime(df['create_time'], format='%Y-%m-%d %H:%M:%S')

    return df

def aplicar_correcao_tempo(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica regras condicionais para corrigir inconsistências entre moves_sum e finish_time."""
    df['moves_sum'] = df['moves'].apply(calcular_soma_tempos)
    
    escala_ratio = np.where(
        (df['moves_sum'] != 0) & (df['moves_sum'].notna()) & (df['maze_finish_time'].notna()),
        df['maze_finish_time'] / df['moves_sum'],
        np.nan
    )

    lim_inf = np.percentile(escala_ratio[~np.isnan(escala_ratio)], LOWER_PERCENTILE)
    lim_sup = np.percentile(escala_ratio[~np.isnan(escala_ratio)], UPPER_PERCENTILE)

    cond_ratio = (
        (np.abs(df['maze_finish_time'] - df['moves_sum']) > TOLERANCIA_DIFERENCA_SEGUNDOS) &
        (~np.isnan(escala_ratio)) &
        ((escala_ratio < lim_inf) | (escala_ratio > lim_sup))
    )

    cond_limite = (df['moves_sum'] > LIMITE_MOVES_SUM_ALTO) | (df['moves_sum'].isna())

    df['maze_finish_time_corrigido'] = df['maze_finish_time']
    df.loc[cond_ratio, 'maze_finish_time_corrigido'] = df.loc[cond_ratio, 'moves_sum']
    df.loc[cond_limite, 'maze_finish_time_corrigido'] = df.loc[cond_limite, 'maze_finish_time']

    return df

def extrair_transicoes(df: pd.DataFrame) -> pd.DataFrame:
    """Expande o log de 'moves' para linhas individuais de transições de estado (s, a)."""
    df_exp = df.assign(moves=df['moves'].str.split(';')).explode('moves').reset_index().rename(columns={'index': 'moves_index'})
    df_exp['moves_index'] = df_exp.groupby('moves_index').cumcount()
    df_trans = df_exp[df_exp['moves'] != ''].reset_index(drop=True)

    df_trans[['coord_origem', 'tempo_str']] = df_trans['moves'].str.split(':', expand=True)
    df_trans[['state_s_row', 'state_s_col']] = df_trans['coord_origem'].str.split(',', expand=True).astype(int)
    
    df_trans['tempo_str'] = df_trans['tempo_str'].astype(str).str.replace(',', '.', regex=False).astype('float32')
    df_trans['state_s_row'] = df_trans['state_s_row'].astype('int8')
    df_trans['state_s_col'] = df_trans['state_s_col'].astype('int8')

    df_trans = df_trans.drop(['maze_finish_time', 'moves', 'moves_sum', 'coord_origem'], axis=1)
    df_trans['pos'] = list(zip(df_trans['state_s_row'], df_trans['state_s_col']))

    return df_trans

def calcular_metricas_caminho(df_trans: pd.DataFrame) -> tuple:
    """Calcula caminhos ótimos (loop-free) e caminhos empíricos para extrair métricas de eficiência."""
    df_valid = df_trans[df_trans['stars'] >= 1].copy()

    # Extração de Caminhos
    paths = (
        df_valid.sort_values(['id', 'maze_name', 'moves_index'])
        .groupby(['id', 'maze_name'])['pos']
        .apply(list).reset_index(name='path')
    )

    # Identificação do Caminho Mais Curto (Loop-Free)
    paths['loop_free_path'] = paths['path'].apply(remover_loops)
    paths['distance'] = paths['loop_free_path'].apply(lambda p: len(p) - 1)
    
    shortest_by_maze = paths.groupby('maze_name')['distance'].min().reset_index(name='shortest_distance')
    idx_shortest = paths.groupby('maze_name')['distance'].idxmin()
    shortest_paths = paths.loc[idx_shortest, ['maze_name', 'id', 'loop_free_path', 'distance']]
    
    # Adicionando curvas aos caminhos ótimos
    shortest_paths['curvas'] = shortest_paths['loop_free_path'].apply(contar_curvas)
    curvas_medias = shortest_paths.groupby('maze_name')['curvas'].mean().reset_index(name='avg_turns_optimal')

    return paths, shortest_by_maze, shortest_paths, curvas_medias

def verificar_chegada_goal(row: pd.Series, maze_goal: dict) -> bool:
    """Valida se o caminho efetivamente alcançou o tile de saída do labirinto."""
    maze = row['maze_name']
    path = row['path']
    if not path:
        return False
    
    ultima_pos = path[-1]
    goal = maze_goal.get(maze)
    if goal is None: return False

    if isinstance(goal, tuple):
        return ultima_pos == goal

    if isinstance(goal, list):
        data_mudanca = GOALS_MUDANCA[maze]['data_mudanca']
        if pd.isna(data_mudanca) or pd.isna(row.get('create_time')):
            return ultima_pos in goal
        
        antigo, novo = goal
        return ultima_pos == antigo if row['create_time'] < data_mudanca else ultima_pos == novo

    return False

def extrair_goals(df_trans: pd.DataFrame) -> dict:
    """Identifica estatisticamente (pela moda) o tile terminal (goal) de cada mapa."""
    df_goals = df_trans.sort_values(['id', 'maze_name', 'trace_id', 'moves_index'])
    
    last_positions = (
        df_goals.groupby(['id','maze_name','trace_id'])['pos']
        .apply(lambda p: p.iloc[-1]).reset_index(name='last_pos')
    )
    
    maze_goal = last_positions.groupby('maze_name')['last_pos'].agg(lambda x: x.value_counts().idxmax()).to_dict()
    
    for maze, config in GOALS_MUDANCA.items():
        maze_goal[maze] = config['goals']
        
    return maze_goal

def compilar_tabela_parametros(df_trans: pd.DataFrame, paths_valid: pd.DataFrame, curvas_medias: pd.DataFrame) -> pd.DataFrame:
    """Consolida as métricas espaciais (dimensões, obstaculos) e temporais (random walk, perplexidade)."""
    
    # Dimensões
    dims = df_trans.groupby('maze_name').agg(
        max_row=('state_s_row', 'max'), max_col=('state_s_col', 'max'),
        min_row=('state_s_row', 'min'), min_col=('state_s_col', 'min')
    ).reset_index()
    
    dims['n_rows'] = dims['max_row'] - dims['min_row'] + 1
    dims['n_cols'] = dims['max_col'] - dims['min_col'] + 1

    # Random Walk
    rw_length = paths_valid.groupby('maze_name')['real_distance'].mean().reset_index(name='avg_random_walk_length')
    
    # Perplexidade
    perplexidade = (
        paths_valid.groupby('maze_name')
        .apply(lambda g: g['real_distance'].mean() / g['shortest_distance'].iloc[0])
        .reset_index(name='perplexity')
    )

    # Obstáculos (Estimativa de densidade da matriz transitável)
    obstaculos_df = (
        df_trans.groupby('maze_name')
        .apply(lambda g: pd.Series({
            'n_rows_real': g['state_s_row'].max() - g['state_s_row'].min() + 1,
            'n_cols_real': g['state_s_col'].max() - g['state_s_col'].min() + 1,
            'visited_cells': len(set(zip(g['state_s_row'], g['state_s_col']))),
        }))
        .reset_index()
    )
    
    obstaculos_df['obstacle_ratio'] = (1 - (obstaculos_df['visited_cells'] / (obstaculos_df['n_rows_real'] * obstaculos_df['n_cols_real']))).clip(0, 1)

    # Merge Final
    df_parametros = (
        dims[['maze_name', 'n_rows', 'n_cols']]
        .merge(curvas_medias, on='maze_name')
        .merge(rw_length, on='maze_name')
        .merge(perplexidade, on='maze_name')
        .merge(obstaculos_df[['maze_name', 'obstacle_ratio']], on='maze_name')
        .round(4)
    )

    return df_parametros

if __name__ == "__main__":
    print("[EDA] Iniciando pré-processamento...")
    df_base = pre_processamento(ARQ_MAZE_FINISH)
    df_corrigido = aplicar_correcao_tempo(df_base)
    
    print("[EDA] Extraindo matrizes de transição...")
    df_transicoes = extrair_transicoes(df_corrigido)
    
    print("[EDA] Calculando caminhos e métricas de complexidade...")
    paths, shortest_by_maze, shortest_paths, curvas_medias = calcular_metricas_caminho(df_transicoes)
    maze_goals = extrair_goals(df_transicoes)
    
    # Estruturando dados de validação de chegada ao final do labirinto
    paths_agg = (
        df_transicoes.sort_values(['id', 'maze_name', 'trace_id', 'moves_index'])
        .groupby(['id', 'maze_name', 'trace_id'], as_index=False)
        .agg(path=('pos', list), create_time=('create_time', 'first'))
    )
    paths_agg['reaches_goal'] = paths_agg.apply(verificar_chegada_goal, axis=1, maze_goal=maze_goals)
    paths_valid = paths_agg[paths_agg['reaches_goal']].copy()
    
    paths_valid['real_distance'] = paths_valid['path'].apply(lambda p: len(p) - 1)
    paths_valid = paths_valid.merge(shortest_by_maze, on='maze_name', how='left')
    paths_valid['ratio'] = paths_valid['real_distance'] / paths_valid['shortest_distance']
    
    print("[EDA] Compilando e salvando tabela de parâmetros por labirinto...")
    tabela_parametros = compilar_tabela_parametros(df_transicoes, paths_valid, curvas_medias)
    
    tabela_parametros.to_csv(PASTA_SAIDA / 'parametros_labirintos.csv', index=False)
    print("[EDA] Finalizado com sucesso. Arquivo parametros_labirintos.csv gerado.")
