//+------------------------------------------------------------------+
+//|                 EA_OPTIMIZER_XAUUSD.mq5                          |
+//|              Gerado automaticamente pelo EA Optimizer AI         |
+//+------------------------------------------------------------------+
+#property copyright "EA_SCALPER_XAUUSD"
+#property version   "1.2"
+#property strict
+#include <Trade/Trade.mqh>
+
+
input double Lots             = 0.10;
input double StopLossPoints   = 109.54;
input double TakeProfitPoints = 1126.02;
input double RiskFactor       = 0.17;
input double ATR_Multiplier   = 3.64;
input int    MagicNumber      = 8888;
input string SessionStart     = "08:00";
input string SessionEnd       = "17:00";

// Trailing stop por ATR
input bool   UseTrailingStop        = true;
input int    ATR_Period             = 14;
input double TrailingATRMultiplier  = 1.5;
input double TrailingMinPoints      = 100.0;

// Proteção de perda diária
input bool   EnableDailyLossGuard   = true;
input double DailyLossMaxPct        = 5.0;   // % do equity a partir do reset
input string DayResetTime           = "00:00"; // horário diário para reiniciar a contagem


CTrade trade;

bool TradingAllowed()
{
   return(AccountInfoInteger(ACCOUNT_TRADE_ALLOWED) && TerminalInfoInteger(TERMINAL_TRADE_ALLOWED));
}

bool IsInSession()
{
   datetime now = TimeCurrent();
   string dateStr = TimeToString(now, TIME_DATE);
   datetime start = StringToTime(dateStr + " " + SessionStart);
   datetime end   = StringToTime(dateStr + " " + SessionEnd);
   if (end <= start) end += 24*60*60; // sessão atravessa meia-noite
   return (now >= start && now <= end);
}

// --- Controle diário de perdas ---
datetime gCurrentAnchor = 0;    // horário do último reset (DayResetTime)
datetime gNextAnchor    = 0;    // próximo reset programado
double   gDailyStartEquity = 0; // equity no início do dia

datetime ComputeAnchor(datetime now)
{
   string dateStr = TimeToString(now, TIME_DATE);
   datetime anchor = StringToTime(dateStr + " " + DayResetTime);
   if(now < anchor) anchor -= 24*60*60; // se ainda não chegou o reset, usa o dia anterior
   return anchor;
}

void ResetDailyAnchor()
{
   datetime now = TimeCurrent();
   gCurrentAnchor = ComputeAnchor(now);
   gNextAnchor = gCurrentAnchor + 24*60*60;
   gDailyStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   Print("[DailyGuard] reset em ", TimeToString(gCurrentAnchor, TIME_DATE|TIME_MINUTES), 
         " start_equity=", gDailyStartEquity);
}

bool DailyLossExceeded()
{
   if(!EnableDailyLossGuard) return false;
   datetime now = TimeCurrent();
   if(now >= gNextAnchor || gCurrentAnchor==0) ResetDailyAnchor();
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   if(gDailyStartEquity <= 0) return false;
   double dd_pct = 100.0 * MathMax(0.0, (gDailyStartEquity - eq) / gDailyStartEquity);
   if(dd_pct > DailyLossMaxPct)
   {
      Print("[DailyGuard] bloqueado. dd=", DoubleToString(dd_pct,2), "% > ", DailyLossMaxPct);
      return true;
   }
   return false;
}

// --- Trailing stop por ATR ---
void ManageTrailing()
{
   if(!UseTrailingStop) return;
   if(!PositionSelect(_Symbol)) return;

   double pt = _Point;
   long   type = (long)PositionGetInteger(POSITION_TYPE);
   double sl   = PositionGetDouble(POSITION_SL);
   double price_open = PositionGetDouble(POSITION_PRICE_OPEN);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   double atr = iATR(_Symbol, PERIOD_M5, ATR_Period, 0) * ATR_Multiplier; // ATR da estratégia
   double trailPts = MathMax(TrailingMinPoints, (atr * TrailingATRMultiplier) / pt);

   bool modified = false;
   double new_sl = sl;
   if(type == POSITION_TYPE_BUY)
   {
      double target = bid - trailPts * pt;
      if((sl==0 || sl < target) && bid > price_open) { new_sl = target; modified = true; }
   }
   else if(type == POSITION_TYPE_SELL)
   {
      double target = ask + trailPts * pt;
      if(sl == 0 || sl > target) { new_sl = target; modified = true; }
   }
   if(modified)
   {
      trade.SetExpertMagicNumber(MagicNumber);
      bool ok = trade.PositionModify(_Symbol, new_sl, 0.0);
      if(!ok) Print("[Trailing] PositionModify falhou, erro=", GetLastError());
   }
}

int OnInit()
{
   Print("EA_OPTIMIZER_XAUUSD v1.2 iniciado. Sessão:", SessionStart, "-", SessionEnd,
         ", SLpts=", StopLossPoints, ", TPpts=", TakeProfitPoints, ", ATR_M=", ATR_Multiplier, ", Risk=", RiskFactor);
   ResetDailyAnchor();
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   Print("EA_OPTIMIZER_XAUUSD finalizado. reason=", reason);
}

void OnTick()
{
   if(!IsInSession()) return;
   if(!TradingAllowed()) return;
   if(DailyLossExceeded()) return; // bloqueia operações após atingir limite diário

   double atr = iATR(_Symbol, PERIOD_M5, ATR_Period, 0) * ATR_Multiplier;
   double ma50 = iMA(_Symbol, PERIOD_M5, 50, 0, MODE_SMA, PRICE_CLOSE, 0);
   double ma20 = iMA(_Symbol, PERIOD_M5, 20, 0, MODE_SMA, PRICE_CLOSE, 0);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double pt  = _Point;

   // Gestão de risco simples por lote
   double lot = Lots * MathMax(0.1, RiskFactor);
   double vmin = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double vmax = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double vstep= SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   lot = MathMax(vmin, MathMin(vmax, lot));
   lot = MathRound(lot / vstep) * vstep;

   // Filtro de tendência + volatilidade (MA20 acima/abaixo MA50 e ATR como gatilho)
   bool upTrend = (ma20 > ma50);
   bool dnTrend = (ma20 < ma50);

   // Trailing para posição existente
   ManageTrailing();

   // Evitar múltiplas posições simultâneas
   if(PositionSelect(_Symbol)) return; // já tem posição: apenas trailing

   double slBuy  = bid - StopLossPoints * pt;
   double tpBuy  = bid + TakeProfitPoints * pt;
   double slSell = ask + StopLossPoints * pt;
   double tpSell = ask - TakeProfitPoints * pt;

   if(upTrend && atr > 0)
   {
      trade.SetExpertMagicNumber(MagicNumber);
      bool ok = trade.Buy(lot, _Symbol, 0.0, slBuy, tpBuy, "EA_OPT_BUY");
      if(!ok) Print("trade.Buy falhou, erro=", GetLastError());
   }
   else if(dnTrend && atr > 0)
   {
      trade.SetExpertMagicNumber(MagicNumber);
      bool ok = trade.Sell(lot, _Symbol, 0.0, slSell, tpSell, "EA_OPT_SELL");
      if(!ok) Print("trade.Sell falhou, erro=", GetLastError());
   }
}
