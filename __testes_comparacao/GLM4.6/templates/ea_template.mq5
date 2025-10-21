//+------------------------------------------------------------------+
//|                                       EA_OPTIMIZER_XAUUSD.mq5 |
//|                        Gerado automaticamente pelo EA Optimizer AI |
//|                                 Versão: {{VERSION}} |
//+------------------------------------------------------------------+
#property copyright "EA Optimizer AI - {{TIMESTAMP}}"
#property version   "{{VERSION}}"
#property strict

//--- Bibliotecas padrão MQL5
#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\AccountInfo.mqh>

//--- Parâmetros Otimizados pelo EA Optimizer AI
input group "📊 Risk Management Parameters"
input double   Lots                    = {{LOTS}};              // Lot Size
input double   StopLoss                = {{STOP_LOSS}};          // Stop Loss (points)
input double   TakeProfit              = {{TAKE_PROFIT}};        // Take Profit (points)
input double   RiskFactor              = {{RISK_FACTOR}};        // Risk Factor (0.5-3.0)
input double   ATR_Multiplier          = {{ATR_MULTIPLIER}};     // ATR Multiplier for SL/TP
input double   MaxDrawdownPct          = {{MAX_DRAWDOWN}};       // Maximum Drawdown Percentage

input group "📈 Technical Indicators"
input int      MAPeriod                = {{MA_PERIOD}};          // Moving Average Period
input int      RSIPeriod               = {{RSI_PERIOD}};         // RSI Period
input int      RSI_Oversold            = {{RSI_OVERSOLD}};       // RSI Oversold Level
input int      RSI_Overbought          = {{RSI_OVERBOUGHT}};     // RSI Overbought Level
input double   BB_StdDev               = {{BB_STDDEV}};          // Bollinger Bands Standard Deviation

input group "⏰ Trading Sessions"
input int      AsianSessionStart       = {{ASIAN_START}};        // Asian Session Start (Hour)
input int      AsianSessionEnd         = {{ASIAN_END}};          // Asian Session End (Hour)
input int      EuropeanSessionStart    = {{EU_START}};           // European Session Start (Hour)
input int      EuropeanSessionEnd      = {{EU_END}};             // European Session End (Hour)
input int      USSessionStart          = {{US_START}};           // US Session Start (Hour)
input int      USSessionEnd            = {{US_END}};             // US Session End (Hour)

input group "🎯 Position Management"
input int      MaxPositions            = {{MAX_POSITIONS}};      // Maximum Concurrent Positions
input int      MagicNumber             = {{MAGIC_NUMBER}};       // Magic Number
input bool     UseTrailingStop         = true;                  // Enable Trailing Stop
input double   TrailingStopPoints     = 20.0;                  // Trailing Stop Distance
input double   TrailingStartPoints    = 30.0;                  // Trailing Stop Activation

input group "🔧 Additional Settings"
input string   TradeComment            = "EA_Optimizer_XAUUSD"; // Trade Comment
input int      Slippage                = 3;                     // Slippage (points)
input bool     EnableLogging           = true;                  // Enable Detailed Logging

//--- Objetos Globais
CTrade         trade;
CPositionInfo  position;
CAccountInfo   account;

//--- Handles de Indicadores
int            maHandle                = INVALID_HANDLE;
int            rsiHandle               = INVALID_HANDLE;
int            bbHandle                = INVALID_HANDLE;
int            atrHandle               = INVALID_HANDLE;

//--- Variáveis Globais
double         point;
datetime       lastBarTime             = 0;
int            currentPositions        = 0;
double         initialBalance          = 0;
double         maxEquity               = 0;
double         currentDrawdown         = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   //--- Configurar objeto de trade
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetSlippage(Slippage);
   trade.SetTypeFillingBySymbol(_Symbol);

   //--- Obter tamanho do ponto
   point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

   //--- Salvar balance inicial
   initialBalance = account.Balance();
   maxEquity = initialBalance;

   //--- Inicializar indicadores
   if(!InitializeIndicators())
   {
      Print("❌ Falha na inicialização dos indicadores");
      return(INIT_FAILED);
   }

   //--- Validar parâmetros
   if(!ValidateParameters())
   {
      Print("❌ Parâmetros inválidos");
      return(INIT_FAILED);
   }

   //--- Log de inicialização
   if(EnableLogging)
   {
      Print("✅ EA Optimizer XAUUSD inicializado com sucesso");
      Print("📊 Parâmetros Otimizados:");
      Print("   - Risk/Reward: 1:", TakeProfit/StopLoss);
      Print("   - Risk Factor: ", RiskFactor);
      Print("   - Lot Size: ", Lots);
      Print("   - MA Period: ", MAPeriod);
      Print("   - RSI Period: ", RSIPeriod);
   }

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   //--- Liberar recursos dos indicadores
   if(maHandle != INVALID_HANDLE) IndicatorRelease(maHandle);
   if(rsiHandle != INVALID_HANDLE) IndicatorRelease(rsiHandle);
   if(bbHandle != INVALID_HANDLE) IndicatorRelease(bbHandle);
   if(atrHandle != INVALID_HANDLE) IndicatorRelease(atrHandle);

   //--- Log final
   if(EnableLogging)
   {
      double finalBalance = account.Balance();
      double totalProfit = finalBalance - initialBalance;
      double profitPct = (totalProfit / initialBalance) * 100;

      Print("📈 EA Optimizer XAUUSD - Resumo Final:");
      Print("   - Saldo Inicial: $", NormalizeDouble(initialBalance, 2));
      Print("   - Saldo Final: $", NormalizeDouble(finalBalance, 2));
      Print("   - Lucro Total: $", NormalizeDouble(totalProfit, 2), " (", profitPct, "%)");
      Print("   - Drawdown Máximo: ", NormalizeDouble(currentDrawdown, 2), "%");
   }
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   //--- Verificar se é uma nova barra
   datetime currentBarTime = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(currentBarTime == lastBarTime)
      return;

   lastBarTime = currentBarTime;

   //--- Atualizar informações
   UpdateDrawdown();
   currentPositions = PositionsTotal();

   //--- Verificar se deve trading
   if(!ShouldTrade())
      return;

   //--- Gerenciar posições existentes
   ManageExistingPositions();

   //--- Verificar novas oportunidades
   if(currentPositions < MaxPositions)
   {
      CheckTradingOpportunities();
   }
}

//+------------------------------------------------------------------+
//| Inicializar indicadores técnicos                                 |
//+------------------------------------------------------------------+
bool InitializeIndicators()
{
   //--- Moving Average
   maHandle = iMA(_Symbol, PERIOD_CURRENT, MAPeriod, 0, MODE_SMA, PRICE_CLOSE);
   if(maHandle == INVALID_HANDLE)
   {
      Print("❌ Falha ao criar indicador MA");
      return(false);
   }

   //--- RSI
   rsiHandle = iRSI(_Symbol, PERIOD_CURRENT, RSIPeriod, PRICE_CLOSE);
   if(rsiHandle == INVALID_HANDLE)
   {
      Print("❌ Falha ao criar indicador RSI");
      return(false);
   }

   //--- Bollinger Bands
   bbHandle = iBands(_Symbol, PERIOD_CURRENT, 20, 0, BB_StdDev, PRICE_CLOSE);
   if(bbHandle == INVALID_HANDLE)
   {
      Print("❌ Falha ao criar indicador Bollinger Bands");
      return(false);
   }

   //--- ATR
   atrHandle = iATR(_Symbol, PERIOD_CURRENT, 14);
   if(atrHandle == INVALID_HANDLE)
   {
      Print("❌ Falha ao criar indicador ATR");
      return(false);
   }

   return(true);
}

//+------------------------------------------------------------------+
//| Validar parâmetros de entrada                                    |
//+------------------------------------------------------------------+
bool ValidateParameters()
{
   //--- Validar Risk Management
   if(StopLoss <= 0 || TakeProfit <= 0)
   {
      Print("❌ Stop Loss e Take Profit devem ser maiores que zero");
      return(false);
   }

   if(StopLoss >= TakeProfit)
   {
      Print("❌ Stop Loss deve ser menor que Take Profit");
      return(false);
   }

   if(Lots <= 0 || Lots > 1.0)
   {
      Print("❌ Lot Size inválido (0.01 - 1.0)");
      return(false);
   }

   //--- Validar períodos
   if(MAPeriod <= 0 || MAPeriod > 200)
   {
      Print("❌ MA Period inválido");
      return(false);
   }

   if(RSIPeriod <= 0 || RSIPeriod > 100)
   {
      Print("❌ RSI Period inválido");
      return(false);
   }

   //--- Validar sessões
   if(AsianSessionStart >= AsianSessionEnd ||
      EuropeanSessionStart >= EuropeanSessionEnd ||
      USSessionStart >= USSessionEnd)
   {
      Print("❌ Configuração de sessões inválida");
      return(false);
   }

   return(true);
}

//+------------------------------------------------------------------+
//| Verificar se deve trading                                        |
//+------------------------------------------------------------------+
bool ShouldTrade()
{
   //--- Verificar drawdown máximo
   if(currentDrawdown > MaxDrawdownPct)
   {
      if(EnableLogging)
         Print("⚠️ Drawdown máximo atingido: ", currentDrawdown, "%");
      return(false);
   }

   //--- Verificar se está dentro das sessões de trading
   if(!IsWithinTradingSession())
      return(false);

   //--- Verificar se mercado está aberto
   if(!IsMarketOpen())
      return(false);

   return(true);
}

//+------------------------------------------------------------------+
//| Verificar se está dentro das sessões de trading                  |
//+------------------------------------------------------------------+
bool IsWithinTradingSession()
{
   MqlDateTime time;
   TimeToStruct(TimeCurrent(), time);
   int currentHour = time.hour;

   //--- Sessão Asiática
   if(currentHour >= AsianSessionStart && currentHour < AsianSessionEnd)
      return(true);

   //--- Sessão Europeia
   if(currentHour >= EuropeanSessionStart && currentHour < EuropeanSessionEnd)
      return(true);

   //--- Sessão Americana
   if(currentHour >= USSessionStart && currentHour < USSessionEnd)
      return(true);

   return(false);
}

//+------------------------------------------------------------------+
//| Verificar se mercado está aberto                                  |
//+------------------------------------------------------------------+
bool IsMarketOpen()
{
   //--- Verificar fim de semana
   MqlDateTime time;
   TimeToStruct(TimeCurrent(), time);

   if(time.day_of_week == 0 || time.day_of_week == 6)  // Domingo ou Sábado
      return(false);

   //--- Verificar se símbolo está disponível para trading
   return(SymbolInfoInteger(_Symbol, SYMBOL_TRADE_MODE) == SYMBOL_TRADE_MODE_FULL);
}

//+------------------------------------------------------------------+
//| Atualizar drawdown atual                                          |
//+------------------------------------------------------------------+
void UpdateDrawdown()
{
   double currentEquity = account.Equity();

   if(currentEquity > maxEquity)
      maxEquity = currentEquity;

   double drawdownAmount = maxEquity - currentEquity;
   currentDrawdown = (drawdownAmount / maxEquity) * 100;
}

//+------------------------------------------------------------------+
//| Gerenciar posições existentes                                     |
//+------------------------------------------------------------------+
void ManageExistingPositions()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!position.SelectByIndex(i))
         continue;

      if(position.Symbol() != _Symbol || position.Magic() != MagicNumber)
         continue;

      //--- Trailing Stop
      if(UseTrailingStop)
      {
         ApplyTrailingStop();
      }
   }
}

//+------------------------------------------------------------------+
//| Aplicar Trailing Stop                                             |
//+------------------------------------------------------------------+
void ApplyTrailingStop()
{
   double openPrice = position.PriceOpen();
   double currentPrice = position.PositionType() == POSITION_TYPE_BUY ?
                         SymbolInfoDouble(_Symbol, SYMBOL_BID) :
                         SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   double stopLoss = position.StopLoss();
   double newStopLoss = 0;

   if(position.PositionType() == POSITION_TYPE_BUY)
   {
      //--- Compra
      double profitPoints = (currentPrice - openPrice) / point;

      if(profitPoints >= TrailingStartPoints)
      {
         newStopLoss = currentPrice - TrailingStopPoints * point;

         if(newStopLoss > stopLoss)
         {
            trade.PositionModify(position.Ticket(), newStopLoss, position.TakeProfit());
         }
      }
   }
   else
   {
      //--- Venda
      double profitPoints = (openPrice - currentPrice) / point;

      if(profitPoints >= TrailingStartPoints)
      {
         newStopLoss = currentPrice + TrailingStopPoints * point;

         if(newStopLoss < stopLoss || stopLoss == 0)
         {
            trade.PositionModify(position.Ticket(), newStopLoss, position.TakeProfit());
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Verificar oportunidades de trading                              |
//+------------------------------------------------------------------+
void CheckTradingOpportunities()
{
   //--- Obter dados dos indicadores
   double ma[1], rsi[1], bbUpper[1], bbLower[1], atr[1];

   if(CopyBuffer(maHandle, 0, 1, 1, ma) <= 0 ||
      CopyBuffer(rsiHandle, 0, 1, 1, rsi) <= 0 ||
      CopyBuffer(bbHandle, 1, 1, 1, bbUpper) <= 0 ||
      CopyBuffer(bbHandle, 2, 1, 1, bbLower) <= 0 ||
      CopyBuffer(atrHandle, 0, 1, 1, atr) <= 0)
   {
      return;
   }

   //--- Obter preços
   double close = iClose(_Symbol, PERIOD_CURRENT, 1);
   double high = iHigh(_Symbol, PERIOD_CURRENT, 1);
   double low = iLow(_Symbol, PERIOD_CURRENT, 1);

   //--- Calcular Stop Loss e Take Profit dinâmicos
   double dynamicSL = atr[0] * ATR_Multiplier;
   double dynamicTP = dynamicSL * (TakeProfit / StopLoss);

   //--- Lógica de Trading baseada nos indicadores otimizados
   CheckBuySignal(close, ma[0], rsi[0], bbLower[0], dynamicSL, dynamicTP);
   CheckSellSignal(close, ma[0], rsi[0], bbUpper[0], dynamicSL, dynamicTP);
}

//+------------------------------------------------------------------+
//| Verificar sinal de compra                                        |
//+------------------------------------------------------------------+
void CheckBuySignal(double close, double ma, double rsi, double bbLower, double sl, double tp)
{
   //--- Condições para compra (baseadas nos parâmetros otimizados)
   bool condition1 = close > ma;  // Tendência de alta
   bool condition2 = rsi < RSI_Oversold;  // RSI oversold
   bool condition3 = close <= bbLower;  // Preço na banda inferior

   //--- Lógica combinada
   if((condition1 && condition2) || (condition2 && condition3))
   {
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double stopLoss = ask - sl * point;
      double takeProfit = ask + tp * point;

      //--- Ajustar lot size baseado no Risk Factor
      double adjustedLots = CalculateOptimalLotSize();

      if(trade.Buy(adjustedLots, _Symbol, ask, stopLoss, takeProfit, TradeComment))
      {
         if(EnableLogging)
            Print("🟢 BUY: ", adjustedLots, " lots @ ", ask, " SL:", stopLoss, " TP:", takeProfit);
      }
   }
}

//+------------------------------------------------------------------+
//| Verificar sinal de venda                                         |
//+------------------------------------------------------------------+
void CheckSellSignal(double close, double ma, double rsi, double bbUpper, double sl, double tp)
{
   //--- Condições para venda (baseadas nos parâmetros otimizados)
   bool condition1 = close < ma;  // Tendência de baixa
   bool condition2 = rsi > RSI_Overbought;  // RSI overbought
   bool condition3 = close >= bbUpper;  // Preço na banda superior

   //--- Lógica combinada
   if((condition1 && condition2) || (condition2 && condition3))
   {
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double stopLoss = bid + sl * point;
      double takeProfit = bid - tp * point;

      //--- Ajustar lot size baseado no Risk Factor
      double adjustedLots = CalculateOptimalLotSize();

      if(trade.Sell(adjustedLots, _Symbol, bid, stopLoss, takeProfit, TradeComment))
      {
         if(EnableLogging)
            Print("🔴 SELL: ", adjustedLots, " lots @ ", bid, " SL:", stopLoss, " TP:", takeProfit);
      }
   }
}

//+------------------------------------------------------------------+
//| Calcular lot size ótimo baseado no Risk Factor                   |
//+------------------------------------------------------------------+
double CalculateOptimalLotSize()
{
   double balance = account.Balance();
   double riskAmount = balance * (RiskFactor / 100.0);

   //--- Calcular lot size baseado no risco
   double symbolLotSize = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   //--- Fórmula simplificada
   double calculatedLots = (riskAmount / (StopLoss * point * 100000)) * symbolLotSize;

   //--- Ajustar para limites do símbolo
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

   calculatedLots = MathMax(calculatedLots, minLot);
   calculatedLots = MathMin(calculatedLots, maxLot);

   //--- Arredondar para o step permitido
   calculatedLots = MathRound(calculatedLots / lotStep) * lotStep;

   return MathMin(calculatedLots, Lots);  // Limitar ao máximo configurado
}

//+------------------------------------------------------------------+
//| Função de diagnóstico                                            |
//+------------------------------------------------------------------+
string GetDiagnosticInfo()
{
   string info = "=== EA Optimizer XAUUSD - Diagnóstico ===\n";
   info += "Símbolo: " + _Symbol + "\n";
   info += "Timeframe: " + EnumToString(Period()) + "\n";
   info += "Saldo Atual: $" + DoubleToString(account.Balance(), 2) + "\n";
   info += "Equity Atual: $" + DoubleToString(account.Equity(), 2) + "\n";
   info += "Drawdown Atual: " + DoubleToString(currentDrawdown, 2) + "%\n";
   info += "Posições Abertas: " + IntegerToString(currentPositions) + "\n";
   info += "Magic Number: " + IntegerToString(MagicNumber) + "\n";
   info += "=========================================\n";

   return info;
}