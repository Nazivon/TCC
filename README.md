# Modelagem de um Processo de Decisão de Markov a partir de Dados de um Jogo Digital de Estratégia

Esse repositório contém os scripts em Python desenvolvidos para meu Trabalho de Conclusão de Curso (TCC), realizando durante as disciplinas Projeto Supervisionado de Graduação I e II (ACH2017/ACH2018).

O projeto investiga o comportamento de navegação humano em um jogo de labirinto através da modelagem das decisões dos jogadores como processos de decisão de Markov (Markov Decision Processes, MDP) empíricos e comparando-os com políticas ótimas computadas por algoritmo de Iteração de Valor (Value Iteration).

## Estrutura do Projeto

* `data/`: Contém os datasets brutos (`partidas.json`, `movimentos.json`, etc.).
* `scripts/`: Scripts em Python organizados por etapa:
    * `03_aprendizado/`: Análise de aprendizado inter e intra-jogador.
    * `04_complexidade_estrutural/`: Cálculo da complexidade híbrida e análises estruturais.
    * `05_validacao/`: Backtest do MDP e verificação de estacionariedade.
* `utils/`: Módulos utilitários para geração de gráficos.
* `graficos/`: Saídas visuais das análises (figuras para a monografia).

## Como rodar

1. **Ambiente:** Este projeto foi desenvolvido em **Python 3.11.4**.
2. **Instalação:** Instale as dependências necessárias executando:
   ```bash
   pip install -r requirements.txt
