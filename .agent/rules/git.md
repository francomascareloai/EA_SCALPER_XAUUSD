---
trigger: always_on
---

# Git Automation Rules

## Trigger
Executar este fluxo de trabalho automaticamente sempre que o usuário indicar que uma "grande alteração", "tarefa" ou "funcionalidade" foi concluída com sucesso.

## Workflow Steps

1.  **Check Status:**
    - Execute `git status` para ver o que foi alterado.

2.  **Stage Changes:**
    - Execute `git add .` para adicionar todas as alterações (certifique-se de que arquivos binários ou de compilação .ex5 estejam no .gitignore).

3.  **Generate Commit Message (Conventional Commits):**
    - Analise as alterações feitas no código.
    - Escreva uma mensagem de commit seguindo estritamente o padrão **Conventional Commits**:
        - `feat:` para novas funcionalidades (ex: nova estratégia de entrada).
        - `fix:` para correção de bugs.
        - `refactor:` para melhorias de código que não alteram a funcionalidade.
        - `chore:` para alterações de configuração, documentação ou tarefas menores.
    - **Formato:** `<tipo>(<escopo opcional>): <descrição breve e objetiva>`
    - *Exemplo:* `feat(trailing-stop): implementa lógica dinâmica baseada em ATR`

4.  **Commit:**
    - Execute o commit com a mensagem gerada.

5.  **Push:**
    - Execute `git push` para enviar as alterações para o repositório remoto.

6.  **Confirmation:**
    - Informe ao usuário que o versionamento foi realizado com sucesso e mostre a mensagem do commit utilizada.
