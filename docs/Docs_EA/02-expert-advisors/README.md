# Expert Advisors (EAs)

Documentação dos EAs do projeto (produção e desenvolvimento), com visão resumida e links.

## Estrutura sugerida por EA
- Propósito e contexto do EA
- Requisitos (versão MT5, símbolos, timeframe)
- Parâmetros (inputs) e valores recomendados
- Lógica de entrada/saída e gestão de risco
- Procedimentos de backtest e validação
- Guia de uso em conta demo/live

## Inventário (resumo)

Produção (`🚀 MAIN_EAS/PRODUCTION/`):

- EA_AUTONOMOUS_XAUUSD_ELITE_v2.0.mq5 — Elite Autônomo XAUUSD (estável)
- EA_FTMO_Scalper_Elite_v2.10_BaselineWithImprovements.mq5 — FTMO Scalper (baseline v2.10)
- EA_FTMO_Scalper_Elite.mq5 — FTMO Scalper (release)
- EA_FTMO_Scalper_Elite_1.mq5 — FTMO Scalper (variante)
- MISC_XAUUSD_M5_SUPER_SCALPER__4__v1.0_XAUUSD.mq4 — M5 Super Scalper (MQL4)

Desenvolvimento (`🚀 MAIN_EAS/DEVELOPMENT/`):

- EA_AUTONOMOUS_XAUUSD_ELITE_v2.0 5K LINHAS.mq5 — versão expandida p/ ajustes
- EA_AUTONOMOUS_XAUUSD_ELITE_v2.0_FIXED_COMPLETE.mq5 — correções consolidadas
- EA_AUTONOMOUS_XAUUSD_ELITE_v2.0_FIXED_PART1.mq5 — hotfix part 1
- EA_AUTONOMOUS_XAUUSD_ELITE_v2.0_FIXED_PART2.mq5 — hotfix part 2
- EA_AUTONOMOUS_XAUUSD_ELITE_v2.0_FIXED_PART3.mq5 — hotfix part 3
- EA_FTMO_SCALPER_ELITE_TESTE.mq5 — branch de testes
- EA_FTMO_SCALPER_ELITE_debug.mq5 — build de debug
- EA_XAUUSD_SmartMoney_v2.mq5 — abordagem SMC
- EA_XAUUSD_ULTIMATE_HYBRID_v3.0.mq5 — híbrido multi-estratégia
- QuantumFibonacci_XAUUSD_Elite_v2.0.mq5 — Fibonacci avançado
- XAUUSD_ML_Complete_EA.mq5 — integração ML completa

Observação: mantenha apenas 1 EA “oficial de produção” por estratégia.

## Onde olhar
- Código-fonte: `🚀 MAIN_EAS/` e `MAIN_EAS/`
- Produção: `🚀 MAIN_EAS/PRODUCTION/`
- Desenvolvimento: `🚀 MAIN_EAS/DEVELOPMENT/`

## Manutenção
- Sincronize com mudanças em `XAUUSD_ML_*` e arquivos `.mq5/.mq4`
- Atualize exemplos em `docs/examples/`

## Modelos rápidos
- Ficha Técnica do EA: `docs/templates/EA-Ficha-Tecnica.md`
- Playbook de Backtest: `docs/templates/Backtest-Playbook.md`
