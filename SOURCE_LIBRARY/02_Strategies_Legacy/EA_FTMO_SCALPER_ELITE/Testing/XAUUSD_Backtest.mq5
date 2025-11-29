#property script_show_inputs
#property strict

input string Test_Symbol = "XAUUSD";
input ENUM_TIMEFRAMES Test_Timeframe = PERIOD_H1;
input datetime Test_Start_Date = D'2023.01.01';
input datetime Test_End_Date = D'2024.01.01';

void OnStart()
{
    // Configurar ambiente de teste
    TesterSetSymbol(Test_Symbol);
    TesterSetTimeframe(Test_Timeframe);
    TesterSetDate(Test_Start_Date, Test_End_Date);
    TesterSetOptimization(false);
    
    // Parâmetros específicos para ouro
    TesterSetParameter("Max_Risk_Per_Trade", 0.5); // Risco reduzido para ouro
    TesterSetParameter("ATR_Multiplier_SL", 2.0);  // SL mais amplo para volatilidade
    TesterSetParameter("ATR_Multiplier_TP", 3.0);  // TP maior para tendências
    TesterSetParameter("Enable_Advanced_Filters", true);
    TesterSetParameter("RSI_Oversold", 35.0);      // Ajustes para ouro
    TesterSetParameter("RSI_Overbought", 65.0);
    
    // Executar teste
    Print("Iniciando backtest para ", Test_Symbol, " de ", TimeToString(Test_Start_Date), " até ", TimeToString(Test_End_Date));
    TesterRun("EA_FTMO_SCALPER_ELITE_TESTE.ex5");
    
    // Gerar relatório
    TesterPrintReport();
    ChartRedraw();
}