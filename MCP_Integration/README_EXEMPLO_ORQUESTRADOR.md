# Exemplo de Orquestração Dinâmica de Modos

Este projeto demonstra de forma prática e funcional como um orquestrador central pode gerenciar e coordenar a troca dinâmica entre diferentes modos de operação (Architect, Code, Ask, Debug) para executar uma tarefa complexa.

## 🎯 Objetivo

Ilustrar o conceito de um sistema inteligente que, ao invés de operar em um único modo fixo, pode alternar entre especialidades distintas para planejar, implementar, questionar e testar soluções de forma integrada e autônoma.

## 📁 Estrutura

- `demo_orquestrador.py`: O script principal que simula o orquestrador e a troca de modos.
- `README_EXEMPLO_ORQUESTRADOR.md`: Este arquivo de documentação.

## 🧠 Como Funciona

O script `demo_orquestrador.py` simula o processo de desenvolvimento de uma função para calcular o Índice de Força Relativa (RSI), uma métrica comum em análise técnica.

### Etapas da Orquestração

1.  **Modo Architect (`modo_architect`)**:
    *   O orquestrador inicia chamando o "modo Architect".
    *   Este modo simula o processo de planejamento: define a tarefa, cria uma especificação técnica detalhada (nome, descrição, entradas, saídas, fórmula).

2.  **Modo Ask (`modo_ask`)**:
    *   Durante ou após o planejamento, o orquestrador pode precisar de informações.
    *   Ele chama o "modo Ask" para fazer perguntas, como "Qual a melhor biblioteca para manipular listas de preços?".
    *   O modo simula a obtenção de uma resposta relevante.

3.  **Modo Code (`modo_code`)**:
    *   Com a especificação em mãos, o orquestrador chama o "modo Code".
    *   Este modo simula a geração do código-fonte da função RSI com base na especificação criada.

4.  **Modo Ask (Novamente)**:
    *   Após a implementação, o orquestrador pode buscar otimizações.
    *   Uma nova chamada ao "modo Ask" é feita com a pergunta "Como posso otimizar o cálculo do RSI para grandes conjuntos de dados?".

5.  **Modo Debug (`modo_debug`)**:
    *   Com o código gerado, o orquestrador chama o "modo Debug".
    *   Este modo simula a execução de testes com dados de exemplo para verificar se o código funciona conforme esperado.

6.  **Conclusão**:
    *   O orquestrador verifica o resultado do "modo Debug" e conclui a tarefa, informando se foi bem-sucedida ou não.

## ▶️ Como Executar

1.  Certifique-se de ter o Python 3 instalado.
2.  Navegue até o diretório `MCP_Integration`.
3.  Execute o script:
    ```bash
    python demo_orquestrador.py
    ```
4.  Observe a saída do console para ver a sequência de ações simulando a troca de modos.

## 🤖 Demonstração de Conceito

Este exemplo é uma **simulação**. Em um sistema real e mais avançado:

*   Cada "modo" poderia ser um serviço ou componente especializado.
*   A troca de modos poderia ser gerenciada por um sistema de controle central que decide qual modo chamar com base no contexto e nos objetivos.
*   As ações de cada modo (planejar, codificar, perguntar, debugar) seriam operações reais, possivelmente envolvendo LLMs, compiladores, executores de código, etc.

Este exemplo serve para ilustrar como a coordenação e a troca de contextos entre diferentes especialidades podem ser estruturadas em um sistema autônomo.

## 🖨️ Exemplo de Saída

Ao executar `python demo_orquestrador.py`, a saída no console seria semelhante a:

```
Iniciando orquestração para a tarefa: Desenvolver uma função para calcular o Índice de Força Relativa (RSI)

[ARCHITECT] Planejando a tarefa: Desenvolver uma função para calcular o Índice de Força Relativa (RSI)
[ARCHITECT] Definindo requisitos...
[ARCHITECT] Escolhendo algoritmo...
[ARCHITECT] Criando especificação técnica...
[ARCHITECT] Especificação técnica criada.

[ASK] Perguntando: Qual a melhor biblioteca para manipular listas de preços em Python?
[ASK] Resposta obtida: Para análise de dados em Python, 'pandas' e 'numpy' são excelentes escolhas.

[CODE] Implementando a especificação: Calcula RSI
[CODE] Escrevendo o código...
[CODE] Código implementado.

[ASK] Perguntando: Como posso otimizar o cálculo do RSI para grandes conjuntos de dados?
[ASK] Resposta obtida: Considere usar bibliotecas otimizadas como 'numpy' para cálculos vetoriais.

[DEBUG] Testando o código com dados: [100, 102, 101, 103, 105]...
[DEBUG] Executando testes...
[DEBUG] Testes concluídos com sucesso. Nenhum erro encontrado.

[ORQUESTRADOR] Tarefa concluída com sucesso!
[ORQUESTRADOR] A função RSI foi planejada, implementada, otimizada e testada.

--- Código Gerado ---

def calcula_rsi(preco_fechamento, periodo=14):
    """
    Calcula o Índice de Força Relativa (RSI).
    
    Args:
        preco_fechamento (list): Lista de preços de fechamento.
        periodo (int): Período para o cálculo do RSI. Padrão é 14.
        
    Returns:
        list: Lista de valores RSI. Os primeiros 'periodo' valores serão None.
    """
    if len(preco_fechamento) <= periodo:
        return [None] * len(preco_fechamento)

    deltas = [preco_fechamento[i] - preco_fechamento[i-1] for i in range(1, len(preco_fechamento))]
    
    ganhos = [delta if delta > 0 else 0 for delta in deltas]
    perdas = [-delta if delta < 0 else 0 for delta in deltas]
    
    media_ganho = sum(ganhos[:periodo]) / periodo
    media_perda = sum(perdas[:periodo]) / periodo
    
    rsi = [None] * periodo
    rs = media_ganho / media_perda if media_perda != 0 else 0
    rsi.append(100 - (100 / (1 + rs)) if rs != 0 else 0)

    for i in range(periodo + 1, len(preco_fechamento)):
        media_ganho = (media_ganho * (periodo - 1) + ganhos[i-1]) / periodo
        media_perda = (media_perda * (periodo - 1) + perdas[i-1]) / periodo
        
        rs = media_ganho / media_perda if media_perda != 0 else 0
        rsi.append(100 - (100 / (1 + rs)) if rs != 0 else 0)
        
    return rsi

```