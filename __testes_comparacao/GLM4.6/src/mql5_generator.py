#!/usr/bin/env python3
"""
⚙️ EA Optimizer AI - MQL5 Generator
Gera automaticamente Expert Advisors MQL5 otimizados
"""

from jinja2 import Environment, FileSystemLoader, Template
from pathlib import Path
from typing import Dict, Any, List
import json
import logging
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MQL5Generator:
    """Gerador de código MQL5 para EAs otimizados"""

    def __init__(self, template_path: str = "../templates/ea_template.mq5"):
        """
        Inicializa o gerador MQL5

        Args:
            template_path: Caminho para o template MQL5
        """
        self.template_path = Path(template_path)
        self.template = None
        self.generated_code = None

        # Carregar template
        self._load_template()

    def _load_template(self) -> None:
        """Carrega o template MQL5"""
        try:
            with open(self.template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()

            self.template = Template(template_content)
            logger.info("✅ Template MQL5 carregado com sucesso")

        except Exception as e:
            logger.error(f"❌ Erro ao carregar template: {e}")
            raise

    def generate_ea(self,
                   optimized_params: Dict[str, Any],
                   output_path: str,
                   symbol: str = "XAUUSD",
                   version: str = "1.0",
                   custom_settings: Dict[str, Any] = None) -> str:
        """
        Gera EA MQL5 com parâmetros otimizados

        Args:
            optimized_params: Parâmetros otimizados
            output_path: Caminho de saída para o EA
            symbol: Símbolo de trading
            version: Versão do EA
            custom_settings: Configurações personalizadas adicionais

        Returns:
            Caminho do arquivo gerado
        """
        logger.info("🔧 Gerando EA MQL5 otimizado...")

        # Preparar parâmetros para o template
        template_params = self._prepare_template_params(
            optimized_params, symbol, version, custom_settings
        )

        # Renderizar template
        try:
            rendered_code = self.template.render(**template_params)
            self.generated_code = rendered_code

            # Validar código gerado
            if not self._validate_mql5_code(rendered_code):
                raise ValueError("Código MQL5 gerado falhou na validação")

            # Salvar arquivo
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(rendered_code)

            logger.info(f"✅ EA gerado com sucesso: {output_file}")
            return str(output_file)

        except Exception as e:
            logger.error(f"❌ Erro ao gerar EA: {e}")
            raise

    def _prepare_template_params(self,
                               optimized_params: Dict[str, Any],
                               symbol: str,
                               version: str,
                               custom_settings: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Prepara parâmetros para renderização do template

        Args:
            optimized_params: Parâmetros otimizados da otimização
            symbol: Símbolo de trading
            version: Versão do EA
            custom_settings: Configurações adicionais

        Returns:
            Parâmetros formatados para o template
        """
        # Timestamp atual
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Parâmetros padrão com valores otimizados
        template_params = {
            'VERSION': version,
            'TIMESTAMP': timestamp,
            'SYMBOL': symbol,

            # Risk Management
            'LOTS': optimized_params.get('lot_size', 0.01),
            'STOP_LOSS': optimized_params.get('stop_loss', 150),
            'TAKE_PROFIT': optimized_params.get('take_profit', 300),
            'RISK_FACTOR': optimized_params.get('risk_factor', 1.5),
            'ATR_MULTIPLIER': optimized_params.get('atr_multiplier', 1.8),
            'MAX_DRAWDOWN': optimized_params.get('max_drawdown', 15.0),

            # Technical Indicators
            'MA_PERIOD': optimized_params.get('ma_period', 20),
            'RSI_PERIOD': optimized_params.get('rsi_period', 14),
            'RSI_OVERSOLD': optimized_params.get('rsi_oversold', 30),
            'RSI_OVERBOUGHT': optimized_params.get('rsi_overbought', 70),
            'BB_STDDEV': optimized_params.get('bb_std', 2.0),

            # Trading Sessions
            'ASIAN_START': optimized_params.get('asian_session_start', 0),
            'ASIAN_END': optimized_params.get('asian_session_end', 8),
            'EU_START': optimized_params.get('european_session_start', 7),
            'EU_END': optimized_params.get('european_session_end', 16),
            'US_START': optimized_params.get('us_session_start', 13),
            'US_END': optimized_params.get('us_session_end', 22),

            # Position Management
            'MAX_POSITIONS': optimized_params.get('max_positions', 3),
            'MAGIC_NUMBER': self._generate_magic_number(symbol),
        }

        # Adicionar configurações personalizadas
        if custom_settings:
            template_params.update(custom_settings)

        # Formatar valores numéricos
        template_params = self._format_template_values(template_params)

        return template_params

    def _format_template_values(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formata valores para o template MQL5

        Args:
            params: Parâmetros brutos

        Returns:
            Parâmetros formatados
        """
        formatted_params = {}

        for key, value in params.items():
            if isinstance(value, float):
                # Formatar floats com precisão adequada
                if key in ['RISK_FACTOR', 'ATR_MULTIPLIER', 'BB_STDDEV', 'MAX_DRAWDOWN']:
                    formatted_params[key] = f"{value:.1f}"
                elif key == ['LOTS']:
                    formatted_params[key] = f"{value:.2f}"
                else:
                    formatted_params[key] = f"{value:.1f}"
            else:
                formatted_params[key] = value

        return formatted_params

    def _generate_magic_number(self, symbol: str) -> int:
        """
        Gera magic number baseado no símbolo

        Args:
            symbol: Símbolo de trading

        Returns:
            Magic number único
        """
        base_magic = 8888
        symbol_hash = sum(ord(c) for c in symbol) % 1000
        return base_magic + symbol_hash

    def _validate_mql5_code(self, code: str) -> bool:
        """
        Valida sintaxe básica do código MQL5

        Args:
            code: Código MQL5 gerado

        Returns:
            True se válido, False caso contrário
        """
        try:
            # Verificar estruturas básicas
            required_patterns = [
                r'#property\s+version',
                r'input\s+group',
                r'int\s+OnInit\(\)',
                r'void\s+OnTick\(\)',
                r'void\s+OnDeinit\(const\s+int\s+reason\)',
                r'#include\s+<Trade\\Trade\.mqh>',
            ]

            for pattern in required_patterns:
                if not re.search(pattern, code, re.IGNORECASE):
                    logger.warning(f"⚠️ Padrão não encontrado: {pattern}")
                    return False

            # Verificar balanceamento de chaves
            open_braces = code.count('{')
            close_braces = code.count('}')

            if open_braces != close_braces:
                logger.warning(f"⚠️ Desbalanceamento de chaves: {open_braces} vs {close_braces}")
                return False

            # Verificar substituição de templates
            if '{{' in code or '}}' in code:
                logger.warning("⚠️ Templates não substituídos encontrados")
                return False

            return True

        except Exception as e:
            logger.error(f"❌ Erro na validação: {e}")
            return False

    def generate_multiple_eas(self,
                            optimization_results: List[Dict[str, Any]],
                            output_dir: str,
                            top_n: int = 3) -> List[str]:
        """
        Gera múltiplos EAs a partir dos melhores resultados

        Args:
            optimization_results: Lista de resultados da otimização
            output_dir: Diretório de saída
            top_n: Número de melhores EAs para gerar

        Returns:
            Lista de caminhos dos EAs gerados
        """
        logger.info(f"🔧 Gerando top {top_n} EAs otimizados...")

        # Ordenar resultados por score
        sorted_results = sorted(
            optimization_results,
            key=lambda x: x.get('score', 0),
            reverse=True
        )[:top_n]

        generated_files = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for i, result in enumerate(sorted_results):
            params = result.get('params', {})
            score = result.get('score', 0)

            # Gerar EA
            filename = f"EA_OPTIMIZER_XAUUSD_Top{i+1}_Score{score:.2f}.mq5"
            ea_path = output_path / filename

            try:
                generated_file = self.generate_ea(
                    optimized_params=params,
                    output_path=str(ea_path),
                    version=f"1.{i+1}",
                    custom_settings={
                        'TOP_RANK': i + 1,
                        'OPTIMIZATION_SCORE': f"{score:.4f}"
                    }
                )
                generated_files.append(generated_file)

                logger.info(f"✅ EA {i+1} gerado: {filename} (Score: {score:.4f})")

            except Exception as e:
                logger.error(f"❌ Erro ao gerar EA {i+1}: {e}")

        return generated_files

    def generate_performance_summary(self,
                                   optimization_results: List[Dict[str, Any]],
                                   output_path: str) -> str:
        """
        Gera resumo de performance dos EAs gerados

        Args:
            optimization_results: Resultados da otimização
            output_path: Caminho de saída do relatório

        Returns:
            Caminho do relatório gerado
        """
        logger.info("📊 Gerando relatório de performance...")

        # Ordenar resultados
        sorted_results = sorted(
            optimization_results,
            key=lambda x: x.get('score', 0),
            reverse=True
        )

        # Gerar relatório
        report_content = self._create_performance_report(sorted_results)

        # Salvar relatório
        report_file = Path(output_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"📊 Relatório gerado: {report_file}")
        return str(report_file)

    def _create_performance_report(self, sorted_results: List[Dict[str, Any]]) -> str:
        """
        Cria conteúdo do relatório de performance

        Args:
            sorted_results: Resultados ordenados por performance

        Returns:
            Conteúdo do relatório em formato markdown
        """
        report = f"""# 📊 EA Optimizer AI - Relatório de Performance

## 🎯 Visão Geral
- **Data de Geração**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total de Configurações Testadas**: {len(sorted_results)}
- **Símbolo**: XAUUSD
- **Timeframe**: M5

## 🏆 Top 10 Configurações Otimizadas

| Rank | Score | Stop Loss | Take Profit | Risk Factor | ATR Mult | MA Period | RSI Period |
|------|-------|-----------|-------------|-------------|----------|-----------|------------|
"""

        for i, result in enumerate(sorted_results[:10]):
            params = result.get('params', {})
            score = result.get('score', 0)

            report += f"| {i+1} | {score:.4f} | {params.get('stop_loss', 'N/A')} | {params.get('take_profit', 'N/A')} | {params.get('risk_factor', 'N/A')} | {params.get('atr_multiplier', 'N/A')} | {params.get('ma_period', 'N/A')} | {params.get('rsi_period', 'N/A')} |\n"

        # Adicionar análise detalhada do top 3
        report += "\n## 📈 Análise Detalhada - Top 3\n\n"

        for i, result in enumerate(sorted_results[:3]):
            params = result.get('params', {})
            score = result.get('score', 0)

            report += f"### 🥇 {'1º' if i == 0 else '2º' if i == 1 else '3º'} Lugar - Score: {score:.4f}\n\n"
            report += f"**Risk Management:**\n"
            report += f"- Stop Loss: {params.get('stop_loss', 'N/A')} points\n"
            report += f"- Take Profit: {params.get('take_profit', 'N/A')} points\n"
            report += f"- Risk/Reward Ratio: {params.get('take_profit', 0) / params.get('stop_loss', 1):.2f}:1\n"
            report += f"- Risk Factor: {params.get('risk_factor', 'N/A')}\n"
            report += f"- ATR Multiplier: {params.get('atr_multiplier', 'N/A')}\n\n"

            report += f"**Technical Indicators:**\n"
            report += f"- MA Period: {params.get('ma_period', 'N/A')}\n"
            report += f"- RSI Period: {params.get('rsi_period', 'N/A')}\n"
            report += f"- RSI Oversold: {params.get('rsi_oversold', 'N/A')}\n"
            report += f"- RSI Overbought: {params.get('rsi_overbought', 'N/A')}\n\n"

            report += f"**Trading Sessions:**\n"
            report += f"- Asian: {params.get('asian_session_start', 'N/A')}h - {params.get('asian_session_end', 'N/A')}h\n"
            report += f"- European: {params.get('european_session_start', 'N/A')}h - {params.get('european_session_end', 'N/A')}h\n"
            report += f"- US: {params.get('us_session_start', 'N/A')}h - {params.get('us_session_end', 'N/A')}h\n\n"

        # Adicionar estatísticas gerais
        if sorted_results:
            all_scores = [r.get('score', 0) for r in sorted_results]
            all_risk_rewards = [
                r.get('params', {}).get('take_profit', 0) / max(r.get('params', {}).get('stop_loss', 1), 1)
                for r in sorted_results
            ]

            report += "## 📊 Estatísticas Gerais\n\n"
            report += f"- **Score Médio**: {np.mean(all_scores):.4f}\n"
            report += f"- **Score Máximo**: {np.max(all_scores):.4f}\n"
            report += f"- **Score Mínimo**: {np.min(all_scores):.4f}\n"
            report += f"- **Desvio Padrão**: {np.std(all_scores):.4f}\n"
            report += f"- **Risk/Reward Médio**: {np.mean(all_risk_rewards):.2f}:1\n\n"

        report += "## 🚀 Próximos Passos\n\n"
        report += "1. **Backtesting**: Testar os EAs gerados em condições de mercado realistas\n"
        report += "2. **Forward Testing**: Executar em conta demo para validar performance\n"
        report += "3. **Otimização Contínua**: Ajustar parâmetros baseado nos resultados\n"
        report += "4. **Monitoramento**: Acompanhar performance em tempo real\n\n"

        report += "---\n"
        report += "*Relatório gerado automaticamente pelo EA Optimizer AI*"

        return report

    def create_deployment_package(self,
                                ea_files: List[str],
                                output_dir: str,
                                include_docs: bool = True) -> str:
        """
        Cria pacote de deploy com EAs e documentação

        Args:
            ea_files: Lista de arquivos EA gerados
            output_dir: Diretório de saída do pacote
            include_docs: Se deve incluir documentação

        Returns:
            Caminho do pacote criado
        """
        logger.info("📦 Criando pacote de deploy...")

        package_dir = Path(output_dir) / "EA_Optimizer_Package"
        package_dir.mkdir(parents=True, exist_ok=True)

        # Copiar EAs
        eas_dir = package_dir / "Expert_Advisors"
        eas_dir.mkdir(exist_ok=True)

        for ea_file in ea_files:
            ea_path = Path(ea_file)
            target_path = eas_dir / ea_path.name
            target_path.write_text(ea_path.read_text(encoding='utf-8'), encoding='utf-8')

        # Criar documentação
        if include_docs:
            docs_dir = package_dir / "Documentation"
            docs_dir.mkdir(exist_ok=True)

            # README
            readme_content = self._create_readme(ea_files)
            (docs_dir / "README.md").write_text(readme_content, encoding='utf-8')

            # Installation Guide
            install_guide = self._create_installation_guide()
            (docs_dir / "INSTALLATION.md").write_text(install_guide, encoding='utf-8')

        logger.info(f"📦 Pacote criado: {package_dir}")
        return str(package_dir)

    def _create_readme(self, ea_files: List[str]) -> str:
        """Cria README para o pacote"""
        readme = f"""# 🤖 EA Optimizer AI - Pacote de Deploy

## 📋 Visão Geral
Este pacote contém Expert Advisors otimizados para XAUUSD, gerados automaticamente pelo EA Optimizer AI.

## 📁 Conteúdo do Pacote
- **Expert_Advisors/**: {len(ea_files)} EAs otimizados
- **Documentation/**: Guias de instalação e uso

## 🚀 EAs Incluídos
"""

        for i, ea_file in enumerate(ea_files):
            ea_name = Path(ea_file).stem
            readme += f"- {i+1}. `{ea_name}.mq5`\n"

        readme += f"""
## ⚙️ Requisitos
- MetaTrader 5 build 2600+
- Conta com permissão para trading automatizado
- Símbolo XAUUSD disponível

## 📖 Instalação
1. Copie os arquivos `.mq5` para pasta `MQL5/Experts/`
2. Compile os EAs no MetaEditor
3. Anexe ao gráfico XAUUSD M5
4. Configure parâmetros conforme necessário

## ⚠️ Aviso de Risco
Trading envolve risco de perda. Teste em conta demo antes de usar em conta real.

---
*Gerado em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return readme

    def _create_installation_guide(self) -> str:
        """Cria guia de instalação"""
        guide = """# 📋 Guia de Instalação - EA Optimizer AI

## 🔧 Passo 1: Preparação do MetaTrader 5

1. Abra o MetaTrader 5
2. Vá em **Ferramentas > Opções**
3. Na aba **Especialistas**, habilite **"Permitir negociação automatizada"**
4. Verifique se **"Permitir DLL"** está desmarcado (segurança)

## 📁 Passo 2: Instalação dos Arquivos

1. **Localize a pasta de dados do MT5:**
   - Menu: Arquivo > Abrir Pasta de Dados
   - Navegue até: `MQL5/Experts/`

2. **Copie os arquivos EA:**
   - Arraste os arquivos `.mq5` para a pasta `Experts/`
   - Alternativa: Copie e cole na pasta

## ⚙️ Passo 3: Compilação

1. Abra o **MetaEditor** (F4 ou ícone de livro amarelo)
2. Navegue até a pasta `Experts`
3. Selecione cada arquivo EA
4. Pressione **F7** ou clique em **Compilar**
5. Verifique se não há erros de compilação

## 📊 Passo 4: Configuração no Gráfico

1. Abra o gráfico **XAUUSD** no timeframe **M5**
2. Navegador (Ctrl+N) > Expert Advisors
3. Arraste o EA desejado para o gráfico
4. Configure os parâmetros:
   - Magic Number (único por EA)
   - Lot Size
   - Risk Management
   - Trading Sessions

## ✅ Passo 5: Ativação

1. Na janela de configurações do EA:
   - Aba **Comum**: Marque **"Permitir negociação automatizada"**
   - Clique em **OK**

2. Verifique se o EA está ativo:
   - Ícone sorridente no canto superior direito do gráfico
   - Mensagem no log: "EA Optimizer XAUUSD inicializado com sucesso"

## 📈 Passo 6: Monitoramento

1. **Aba Especialistas:** Monitor mensagens e operações
2. **Resultados de Negociação:** Acompanhe performance
3. **Log do EA:** Verifique diagnósticos com `GetDiagnosticInfo()`

## 🔧 Solução de Problemas

### EA não opera:
- Verifique se "Permitir negociação automatizada" está ativo
- Confira se o mercado está aberto
- Verifique sessões de trading configuradas

### Erros de compilação:
- Instale a versão mais recente do MetaTrader 5
- Verifique se todos os includes estão disponíveis

### Performance ruim:
- Ajuste Risk Factor
- Verifique se símbolo e timeframe estão corretos
- Considere reotimizar parâmetros

## 📞 Suporte
Para dúvidas, consulte a documentação completa ou logs do EA.
"""
        return guide

if __name__ == "__main__":
    # Teste do gerador
    from optimizer import EAOptimizer

    # Criar otimizador e carregar resultados de exemplo
    optimizer = EAOptimizer("../data/input/sample_backtest.csv")

    # Simular resultados de otimização
    sample_params = {
        'stop_loss': 120,
        'take_profit': 240,
        'risk_factor': 1.8,
        'atr_multiplier': 1.6,
        'lot_size': 0.02,
        'ma_period': 20,
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'bb_std': 2.0,
        'max_positions': 3,
        'asian_session_start': 0,
        'asian_session_end': 8,
        'european_session_start': 7,
        'european_session_end': 16,
        'us_session_start': 13,
        'us_session_end': 22
    }

    # Gerar EA
    generator = MQL5Generator()
    ea_path = generator.generate_ea(
        optimized_params=sample_params,
        output_path="../output/EA_OPTIMIZER_XAUUSD_TEST.mq5"
    )

    print(f"✅ EA gerado: {ea_path}")