#!/usr/bin/env python3
"""
Exemplo 01: Conexão Básica com MetaTrader 5
============================================

Este exemplo demonstra como estabelecer conexão com MetaTrader 5,
obter informações da conta e dados básicos de mercado.

Pré-requisitos:
- Python 3.8+
- MetaTrader 5 instalado
- Conta RoboForex configurada
"""

import asyncio
import sys
import os
from pathlib import Path

# Adicionar diretório raiz ao path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
from ea_scalper_sdk import MT5Client
from ea_scalper_sdk.exceptions import MT5ConnectionError, MT5Error

# Carregar variáveis de ambiente
load_dotenv()

async def main():
    """Função principal de teste de conexão"""

    print("🚀 Testando Conexão com MetaTrader 5")
    print("=" * 50)

    # Validar configuração
    required_vars = ['MT5_LOGIN', 'MT5_PASSWORD', 'MT5_SERVER']
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"❌ Variáveis de ambiente obrigatórias não configuradas: {missing_vars}")
        print("💡 Configure as variáveis no arquivo .env")
        return

    try:
        # Inicializar cliente MT5
        print("📡 Inicializando cliente MT5...")
        client = MT5Client()

        # Tentar conexão
        print("🔌 Conectando ao servidor...")
        login = int(os.getenv('MT5_LOGIN'))
        password = os.getenv('MT5_PASSWORD')
        server = os.getenv('MT5_SERVER')

        success = await client.connect(login, password, server)

        if not success:
            print("❌ Falha na conexão")
            return

        print("✅ Conexão estabelecida com sucesso!")

        # Obter informações da conta
        print("\n📊 Informações da Conta:")
        print("-" * 30)

        account_info = await client.get_account_info()

        print(f"Login: {account_info.get('login', 'N/A')}")
        print(f"Servidor: {account_info.get('server', 'N/A')}")
        print(f"Empresa: {account_info.get('company', 'N/A')}")
        print(f"Moeda: {account_info.get('currency', 'N/A')}")
        print(f"Alavancagem: 1:{account_info.get('leverage', 'N/A')}")
        print(f"Saldo: ${account_info.get('balance', 0):.2f}")
        print(f"Equity: ${account_info.get('equity', 0):.2f}")
        print(f"Margem: ${account_info.get('margin', 0):.2f}")
        print(f"Margem Livre: ${account_info.get('free_margin', 0):.2f}")
        print(f"Nível de Margem: {account_info.get('margin_level', 0):.1f}%")

        # Verificar símbolo XAUUSD
        print("\n💎 Verificando Símbolo XAUUSD:")
        print("-" * 30)

        symbol_info = await client.get_symbol_info("XAUUSD")

        if symbol_info:
            print(f"✅ XAUUSD disponível")
            print(f"Spread: {symbol_info.get('spread', 0)} pontos")
            print(f"Lote Mínimo: {symbol_info.get('volume_min', 0)}")
            print(f"Lote Máximo: {symbol_info.get('volume_max', 0)}")
            print(f"Passo do Lote: {symbol_info.get('volume_step', 0)}")
            print(f"Contrato: {symbol_info.get('trade_contract_size', 0)} unidades")
            print(f"Dígitos: {symbol_info.get('digits', 0)}")
        else:
            print("❌ XAUUSD não encontrado")
            print("💡 Tentando XAUUSD_TDS...")

            symbol_info = await client.get_symbol_info("XAUUSD_TDS")
            if symbol_info:
                print(f"✅ XAUUSD_TDS encontrado")
                print(f"Spread: {symbol_info.get('spread', 0)} pontos")
            else:
                print("❌ Nenhum símbolo XAUUSD encontrado")

        # Obter dados de mercado
        print("\n📈 Dados de Mercado Recentes:")
        print("-" * 30)

        # Obter últimas barras H1
        bars = await client.get_bars("XAUUSD", "H1", 5)

        if bars:
            print(f"Últimas {len(bars)} barras H1:")
            for i, bar in enumerate(bars[-3:], 1):
                print(f"  Barra {i}: O={bar['open']:.2f} H={bar['high']:.2f} L={bar['low']:.2f} C={bar['close']:.2f}")

            current_price = bars[-1]['close']
            print(f"\n💰 Preço Atual: ${current_price:.2f}")
        else:
            print("❌ Não foi possível obter barras de preço")

        # Obter ticks recentes
        print("\n🔄 Ticks Recentes:")
        print("-" * 30)

        ticks = await client.get_ticks("XAUUSD", 3)

        if ticks:
            for i, tick in enumerate(ticks, 1):
                print(f"  Tick {i}: Bid={tick['bid']:.2f} Ask={tick['ask']:.2f}")

            current_spread = ticks[-1]['ask'] - ticks[-1]['bid']
            print(f"\n📊 Spread Atual: {current_spread * 100:.1f} pontos")
        else:
            print("❌ Não foi possível obter ticks")

        # Verificar posições abertas
        print("\n📋 Posições Abertas:")
        print("-" * 30)

        positions = await client.get_positions()

        if positions:
            print(f"Total de posições: {len(positions)}")
            for pos in positions:
                print(f"  {pos['type']} {pos['symbol']} - {pos['volume']} lotes - Lucro: ${pos['profit']:.2f}")
        else:
            print("✅ Nenhuma posição aberta")

        # Teste de latência
        print("\n⚡ Teste de Latência:")
        print("-" * 30)

        import time

        start_time = time.time()
        await client.get_account_info()
        latency = (time.time() - start_time) * 1000

        print(f"Latência de resposta: {latency:.2f}ms")

        if latency < 100:
            print("✅ Excelente")
        elif latency < 200:
            print("✅ Bom")
        elif latency < 500:
            print("⚠️ Regular")
        else:
            print("❌ Ruim - pode afetar trading")

        print("\n🎉 Teste de conexão concluído com sucesso!")

    except MT5ConnectionError as e:
        print(f"❌ Erro de conexão: {e}")
        print("💡 Verifique suas credenciais e se o MT5 está aberto")

    except MT5Error as e:
        print(f"❌ Erro do MT5: {e}")

    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Desconectar
        if 'client' in locals():
            await client.disconnect()
            print("\n🔌 Desconectado do MetaTrader 5")

if __name__ == "__main__":
    print("⚠️ ATENÇÃO: Certifique-se de que o MetaTrader 5 está aberto e conectado")
    print("⚠️ Este script usará as credenciais do arquivo .env")
    print()

    asyncio.run(main())