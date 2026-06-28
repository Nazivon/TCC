# ARQUIVOS DE SAÍDA DO PROJETO

**mdp_por_labirinto.pkl:** exibe as recompensas esperadas e as transições obtidas de cada estado e cada ação possível em todos os labirintos. . Além disso, ele elenca todos os estados válidos do labirinto, os quantifica, e denota o estado terminal (a célula de saída do labirinto).

**politica_empirica.pkl:** apresenta a distribuição empírica de ações observadas em cada célula válida dos labirintos, ou seja, a porcentagem de jogadores que realizaram determinada ação em cada estado. Essa política estocástica reflete decisões humanas agregadas a partir das partidas observadas.

**resultados_vi.pkl:** guarda dois objetos produzidos pelo algoritmo VI para cada labirinto – a função valor ótima V*(s), contendo um número real para cada estado, e a política ótima π*(s), uma direção para cada estado. O valor ótimo é a recompensa total esperada e descontada que um agente ótimo acumularia a partir do estado _s_ até o terminal seguindo a política ótima, que é a ação que maximiza o valor a partir desse estado. 

**Tabelas em CSV:** tabelas derivadas empregadas nas análises estatísticas e na geração das figuras do TCC.
