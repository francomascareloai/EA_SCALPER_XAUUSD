# MQL5 Knowledge Base - Books & Documentation

Este diretorio contem todos os livros e documentacao para indexacao no RAG local.

## Livros Incluidos

### MQL5 Core (Obrigatorios)
| Arquivo | Descricao | Prioridade |
|---------|-----------|------------|
| `mql5.pdf` | MQL5 Language Reference - Documentacao oficial completa | ⭐⭐⭐ |
| `mql5book (1).pdf` | MQL5 Programming for Traders - Livro oficial MetaQuotes | ⭐⭐⭐ |
| `Expert_Advisor_Programming_for_MetaTrader_5.pdf` | Andrew Young - Classico de EA programming | ⭐⭐⭐ |

### Trading Strategies
| Arquivo | Descricao | Prioridade |
|---------|-----------|------------|
| `Smart Money Concept (SMC).pdf` | Smart Money Concepts completo | ⭐⭐⭐ |
| `Volume Profile The Insiders Guide to Trading.pdf` | Volume Profile avancado | ⭐⭐ |
| `Fmc Price Action.pdf` | Price Action trading | ⭐⭐ |
| `BTMM - MARKET MAKER CYCLE.pdf` | Market Maker methodology | ⭐⭐ |

### Technical Analysis
| Arquivo | Descricao | Prioridade |
|---------|-----------|------------|
| `Carolyn_Borden_Fibonacci_Trading.pdf` | Fibonacci trading completo | ⭐⭐ |
| `Maximum_Trading_Gains_with_Anchored_VWAP.pdf` | VWAP strategies | ⭐⭐ |
| `the-only-technical-analysis-book-you-will-ever-need.pdf` | TA fundamentals | ⭐ |
| `secrets-on-reversal-trading-frank-miller.pdf` | Reversal patterns | ⭐ |

### Algorithmic Trading
| Arquivo | Descricao | Prioridade |
|---------|-----------|------------|
| `Algorithmic_Trading_Methods_Applications_Using_Advanced_Statistics.pdf` | ML/Statistics para trading | ⭐⭐⭐ |
| `Forecasting_Financial_Markets_The.pdf` | Forecasting methods | ⭐⭐ |

### Business/Mindset
| Arquivo | Descricao | Prioridade |
|---------|-----------|------------|
| `Book - Six Figures From Scratch.pdf` | Trading business mindset | ⭐ |

## Status de Indexacao

- [ ] Pendente - Configurar mcp-local-rag
- [ ] Pendente - Indexar livros MQL5 core
- [ ] Pendente - Indexar livros de estrategia
- [ ] Pendente - Scrape documentacao online

## Uso com RAG

Apos indexacao, pergunte ao Claude/Cursor:
- "Como usar OnnxCreate no MQL5?"
- "Explique Order Blocks do SMC"
- "Como calcular Volume Profile?"
- "Qual a sintaxe do OnTick?"

## Estrutura Recomendada

```
DOCS/
├── BOOKS/
│   ├── mql5/           # Core MQL5 docs (move after organized)
│   ├── strategies/     # Trading strategies
│   ├── technical/      # Technical analysis
│   └── algorithmic/    # ML/Stats
├── SCRAPED/            # Online documentation (to be created)
│   ├── mql5_reference/
│   └── mql5_articles/
└── MARKDOWN/           # Converted for better chunking (optional)
```
