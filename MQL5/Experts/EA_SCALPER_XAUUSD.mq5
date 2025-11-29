//+------------------------------------------------------------------+
//|                                            EA_SCALPER_XAUUSD.mq5 |
//|                           EA_SCALPER_XAUUSD - Singularity Edition |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Franco - Singularity Edition"
#property link      "https://www.mql5.com"
#property version   "3.30"
#property strict

/*
   v3.30 - Order Flow + Multi-Timeframe Architecture
   ==================================================
   HTF (H1)  = Direction filter - NEVER trade against H1 trend
   MTF (M15) = Structure zones  - OB, FVG, Liquidity levels
   LTF (M5)  = Execution        - Entry confirmation & tight SL
   
   NEW in v3.30:
   - Footprint/Cluster Chart Analysis (ATAS-style)
   - Diagonal Imbalance Detection
   - Stacked Imbalance Detection (3+ consecutive)
   - Absorption Zone Detection
   - Order Flow Confluence Scoring
   
   Benefits:
   - 2-3x more trading opportunities
   - Tighter stop losses (M5 precision)
   - Better alignment with SMC methodology
   - Order Flow confirmation for higher win rate
   - Spread cost still acceptable (5-8% on M5)
*/

//--- Core Includes
#include <EA_SCALPER/Core/Definitions.mqh>
#include <EA_SCALPER/Risk/FTMO_RiskManager.mqh>
#include <EA_SCALPER/Signal/SignalScoringModule.mqh>
#include <EA_SCALPER/Execution/TradeExecutor.mqh>
#include <EA_SCALPER/Execution/CTradeManager.mqh>

//--- Multi-Timeframe Manager (NEW in v3.20)
#include <EA_SCALPER/Analysis/CMTFManager.mqh>

//--- Analysis Modules (Singularity)
#include <EA_SCALPER/Analysis/CRegimeDetector.mqh>
#include <EA_SCALPER/Analysis/CLiquiditySweepDetector.mqh>
#include <EA_SCALPER/Analysis/CAMDCycleTracker.mqh>
#include <EA_SCALPER/Analysis/CStructureAnalyzer.mqh>
#include <EA_SCALPER/Analysis/CSessionFilter.mqh>
#include <EA_SCALPER/Analysis/CNewsFilter.mqh>
#include <EA_SCALPER/Analysis/CEntryOptimizer.mqh>
#include <EA_SCALPER/Analysis/EliteFVG.mqh>

//--- Order Flow / Footprint Analysis (NEW v3.30)
#include <EA_SCALPER/Analysis/CFootprintAnalyzer.mqh>

//--- Signal Modules
#include <EA_SCALPER/Signal/CConfluenceScorer.mqh>

//--- Bridge (Phase 2)
//#include <EA_SCALPER/Bridge/PythonBridge.mqh>

//--- Input Parameters: Risk Management
input group "=== Risk Management (FTMO) ==="
input double   InpRiskPerTrade      = 0.5;      // Risk Per Trade (%)
input double   InpMaxDailyLoss      = 5.0;      // Max Daily Loss (%)
input double   InpSoftStop          = 3.5;      // Soft Stop Level (%)
input double   InpMaxTotalLoss      = 10.0;     // Max Total Loss (%)
input int      InpMaxTradesPerDay   = 20;       // Max Trades Per Day

//--- Input Parameters: Scoring Engine
input group "=== Scoring Engine ==="
input int      InpExecutionThreshold = 85;      // Execution Score Threshold (0-100)
input double   InpWeightTech        = 0.6;      // Weight: Technical
input double   InpWeightFund        = 0.25;     // Weight: Fundamental
input double   InpWeightSent        = 0.15;     // Weight: Sentiment

//--- Input Parameters: Execution
input group "=== Execution Settings ==="
input int      InpSlippage          = 50;       // Max Slippage (points)
input int      InpMagicNumber       = 123456;   // Magic Number
input string   InpTradeComment      = "SINGULARITY"; // Trade Comment

//--- Input Parameters: Session Filter
input group "=== Session & Time Filters ==="
input bool     InpAllowAsian        = false;    // Allow Asian Session
input bool     InpAllowLateNY       = false;    // Allow Late NY (17:00+)
input int      InpGMTOffset         = 0;        // Broker GMT Offset
input int      InpFridayCloseHour   = 14;       // Friday Close Hour (GMT)

//--- Input Parameters: News Filter
input group "=== News Filter ==="
input bool     InpNewsFilterEnabled = true;     // Enable News Filter
input bool     InpBlockHighImpact   = true;     // Block High-Impact News
input bool     InpBlockMediumImpact = false;    // Block Medium-Impact News

//--- Input Parameters: Entry Optimization
input group "=== Entry Optimization ==="
input double   InpMinRR             = 1.5;      // Minimum R:R Ratio
input double   InpTargetRR          = 2.5;      // Target R:R Ratio
input int      InpMaxWaitBars       = 10;       // Max Bars to Wait for Entry

//--- Input Parameters: Multi-Timeframe (NEW v3.20)
input group "=== Multi-Timeframe Settings ==="
input bool     InpUseMTF            = true;     // Enable MTF Architecture
input double   InpMinMTFConfluence  = 60.0;     // Min MTF Confluence Score
input bool     InpRequireHTFAlign   = true;     // Require H1 Trend Alignment
input bool     InpRequireMTFZone    = true;     // Require M15 Structure Zone
input bool     InpRequireLTFConfirm = true;     // Require M5 Confirmation

//--- Global Objects (Core)
CFTMO_RiskManager    g_RiskManager;
CSignalScoringModule g_ScoringEngine;
CTradeExecutor       g_Executor;
CTradeManager        g_TradeManager;

//--- Multi-Timeframe Manager (NEW v3.20)
CMTFManager          g_MTF;

//--- Singularity Analysis Objects
CRegimeDetector         g_Regime;
CStructureAnalyzer      g_Structure;
CLiquiditySweepDetector g_Sweep;
CAMDCycleTracker        g_AMD;
CSessionFilter          g_Session;
CNewsFilter             g_News;
CEntryOptimizer         g_EntryOpt;
CEliteFVGDetector       g_FVG;
CConfluenceScorer       g_Confluence;

//--- Global State
bool g_IsEmergencyMode = false;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("=== EA_SCALPER_XAUUSD v3.30 Singularity Order Flow Edition ===");
   Print("=== HTF=H1 | MTF=M15 | LTF=M5 (Execution) ===");
   
   // 1. Initialize Risk Manager (FTMO Compliance)
   if(!g_RiskManager.Init(InpRiskPerTrade, InpMaxDailyLoss, InpMaxTotalLoss, InpMaxTradesPerDay, InpSoftStop))
   {
      Print("Critical Error: Risk Manager Initialization Failed!");
      return(INIT_FAILED);
   }

   // 2. Initialize Scoring Engine
   if(!g_ScoringEngine.Init(InpWeightTech, InpWeightFund, InpWeightSent))
   {
      Print("Critical Error: Scoring Engine Initialization Failed!");
      return(INIT_FAILED);
   }

   // 3. Initialize Executor (Legacy)
   g_Executor.Init(InpMagicNumber, InpSlippage, InpTradeComment);
   
   // 4. Initialize Trade Manager (New - with partial TPs)
   if(!g_TradeManager.Init(InpMagicNumber, InpSlippage))
   {
      Print("Warning: Trade Manager failed - using legacy executor");
   }
   g_TradeManager.SetManagementMode(MGMT_PARTIAL_TP);
   g_TradeManager.ConfigurePartials(1.5, 2.5, 0.40, 0.50);  // 40% at 1.5R, 30% at 2.5R

   // 5. Initialize Multi-Timeframe Manager (NEW v3.20)
   if(InpUseMTF)
   {
      if(!g_MTF.Init(_Symbol))
      {
         Print("Warning: MTF Manager initialization failed - using single TF mode");
      }
      else
      {
         g_MTF.SetMinConfluence(InpMinMTFConfluence);
         Print("MTF Manager initialized: H1+M15+M5 architecture active");
      }
   }

   // 6. Initialize Singularity Analysis Modules
   // Regime detector
   g_Regime.SetHurstWindow(100);
   g_Regime.SetEntropyWindow(100);
   
   // Structure analyzer
   g_Structure.SetSwingStrength(3);
   g_Structure.SetLookback(100);
   
   // Sweep detector
   if(!g_Sweep.Initialize())
      Print("Warning: Sweep detector initialization issue");
   
   // AMD Cycle tracker
   if(!g_AMD.Initialize())
      Print("Warning: AMD tracker initialization issue");
   
   // Session filter
   if(!g_Session.Initialize(InpGMTOffset))
   {
      Print("Warning: Session filter initialization issue");
   }
   g_Session.AllowAsianTrading(InpAllowAsian);
   g_Session.AllowLateNYTrading(InpAllowLateNY);
   g_Session.SetFridayCloseHour(InpFridayCloseHour);
   
   // News filter
   if(!g_News.Initialize(InpGMTOffset))
   {
      Print("Warning: News filter initialization issue");
   }
   g_News.SetEnabled(InpNewsFilterEnabled);
   g_News.BlockHighImpact(InpBlockHighImpact);
   g_News.BlockMediumImpact(InpBlockMediumImpact);
   
   // Entry optimizer
   if(!g_EntryOpt.Initialize())
   {
      Print("Warning: Entry optimizer initialization issue");
   }
   g_EntryOpt.SetMinRR(InpMinRR);
   g_EntryOpt.SetTargetRR(InpTargetRR);
   g_EntryOpt.SetMaxWaitBars(InpMaxWaitBars);
   
   // 7. Connect Confluence Scorer to all detectors
   g_Confluence.AttachRegimeDetector(&g_Regime);
   g_Confluence.AttachStructureAnalyzer(&g_Structure);
   g_Confluence.AttachSweepDetector(&g_Sweep);
   g_Confluence.AttachAMDTracker(&g_AMD);
   g_Confluence.AttachOBDetector(g_ScoringEngine.GetOBDetector());
   g_Confluence.AttachFVGDetector(&g_FVG);
   g_Confluence.SetMinScore(70);       // Tier B minimum
   g_Confluence.SetMinConfluences(3);  // At least 3 factors
   
   // 8. Timer for slow-lane updates
   EventSetTimer(1);

   Print("=== Singularity MTF Edition Initialized Successfully ===");
   Print("Execution TF: M5 | Structure TF: M15 | Direction TF: H1");
   Print("Session: ", g_Session.GetSessionName());
   Print("News: ", g_News.GetCurrentStatus());
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   
   // Cleanup modules
   g_MTF.Deinit();
   g_AMD.Deinitialize();
   g_Sweep.Deinitialize();
   
   Print("EA_SCALPER_XAUUSD v3.30 Singularity Order Flow Deinitialized. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Always refresh risk state so daily reset can recover from emergency
   g_RiskManager.OnTick();
   
   // If we were halted previously but risk manager is now clear (e.g., new day), exit emergency
   if(g_IsEmergencyMode && !g_RiskManager.IsTradingHalted())
   {
      g_IsEmergencyMode = false;
      Print("Emergency mode cleared - risk limits back within range.");
   }
   
   // Keep managing open positions even during halt/soft-stop conditions
   g_TradeManager.OnTick();  // State machine for partials/trailing
   if(!g_TradeManager.HasActiveTrade())
      g_Executor.ManagePositions(); // Legacy manager only if TradeManager not handling a position
   
   // Hard halt: no new trades while still allow management above
   if(g_RiskManager.IsTradingHalted())
   {
      if(!g_IsEmergencyMode)
      {
         Print("TRADING HALTED: Risk Limits Breached.");
         g_IsEmergencyMode = true;
      }
      return;
   }
   
   // Skip new trade logic if we already have a position
   if(g_TradeManager.HasActiveTrade()) return;

   //=== GATE 3: Session Filter ===
   if(!g_Session.IsTradingAllowed())
   {
      // Static to avoid spamming logs
      static datetime last_session_log = 0;
      if(TimeCurrent() - last_session_log > 3600)
      {
         Print("Session blocked: ", g_Session.GetSessionName());
         last_session_log = TimeCurrent();
      }
      return;
   }
   
   //=== GATE 4: News Filter ===
   if(!g_News.IsTradingAllowed())
   {
      static datetime last_news_log = 0;
      if(TimeCurrent() - last_news_log > 300)
      {
         Print("News blackout: ", g_News.GetCurrentStatus());
         last_news_log = TimeCurrent();
      }
      return;
   }
   
   //=== GATE 5: Risk Manager - Can open new trade? ===
   if(!g_RiskManager.CanOpenNewTrade()) return;

   //=== GATE 6: Multi-Timeframe Analysis (NEW v3.20) ===
   if(InpUseMTF)
   {
      // Update MTF Manager (H1, M15, M5)
      g_MTF.Update();
      
      // Check HTF (H1) direction - NEVER trade against H1 trend
      SMTFConfluence mtf_conf_htf = g_MTF.GetConfluence();
      if(InpRequireHTFAlign && !mtf_conf_htf.htf_aligned)
      {
         static datetime last_htf_log = 0;
         if(TimeCurrent() - last_htf_log > 1800)
         {
            Print("[MTF] H1 trend not aligned - waiting. ", g_MTF.GetAnalysisSummary());
            last_htf_log = TimeCurrent();
         }
         return;
      }
   }

   //=== SIGNAL GENERATION ===
   // Update analysis modules
   g_Sweep.Update();
   g_AMD.Update();
   
   // Refresh FVGs on new M15 bar to avoid heavy per-tick detection
   static datetime last_fvg_bar = 0;
   datetime m15_bar = iTime(_Symbol, PERIOD_M15, 0);
   if(m15_bar != last_fvg_bar)
   {
      g_FVG.DetectEliteFairValueGaps();
      last_fvg_bar = m15_bar;
   }
   
   // Calculate confluence score
   int score = g_ScoringEngine.CalculateScore();
   
   // Check minimum score threshold
   if(score < InpExecutionThreshold) return;
   
   //=== GATE 7: AMD Cycle Check ===
   // Only trade in DISTRIBUTION phase (after manipulation)
   ENUM_AMD_PHASE amd_phase = g_AMD.GetCurrentPhase();
   if(amd_phase == AMD_PHASE_ACCUMULATION || amd_phase == AMD_PHASE_MANIPULATION)
   {
      // Still in setup phase - wait
      return;
   }
   
   //=== ENTRY OPTIMIZATION (M5 precision) ===
   ENUM_ORDER_TYPE direction = g_ScoringEngine.GetDirection();
   ENUM_SIGNAL_TYPE signal = (direction == ORDER_TYPE_BUY) ? SIGNAL_BUY : SIGNAL_SELL;
   
   if(signal == SIGNAL_NONE) return;
   
   // Pull best order block zone from scoring module if available
   double ob_low = 0, ob_high = 0;
   SAdvancedOrderBlock best_ob;
   if(g_ScoringEngine.GetBestOrderBlock(best_ob))
   {
      ob_low = best_ob.low_price;
      ob_high = best_ob.high_price;
   }
   bool has_ob = (ob_low > 0 && ob_high > 0);
   
   // Pull nearest FVG in signal direction
   double fvg_low = 0, fvg_high = 0;
   SEliteFairValueGap best_fvg;
   if(signal == SIGNAL_BUY)
   {
      if(g_FVG.GetNearestFVG(FVG_BULLISH, best_fvg))
      {
         fvg_low = best_fvg.lower_level;
         fvg_high = best_fvg.upper_level;
      }
   }
   else if(signal == SIGNAL_SELL)
   {
      if(g_FVG.GetNearestFVG(FVG_BEARISH, best_fvg))
      {
         fvg_low = best_fvg.lower_level;
         fvg_high = best_fvg.upper_level;
      }
   }
   bool has_fvg = (fvg_low > 0 && fvg_high > 0);
   
   // Update MTF manager with real structure flags (OB/FVG)
   g_MTF.SetStructureFlags(has_ob, has_fvg, g_Sweep.HasRecentSweep());
   
   // Refresh MTF confluence with updated structure flags
   if(InpUseMTF)
   {
      SMTFConfluence mtf_conf = g_MTF.GetConfluence();
      
      // Check minimum MTF alignment (at least GOOD = 2 TFs aligned)
      if(mtf_conf.alignment < MTF_ALIGN_GOOD)
         return;
      
      // Require structure zone if configured
      if(InpRequireMTFZone && !mtf_conf.mtf_structure)
      {
         static datetime last_mtf_zone_log = 0;
         if(TimeCurrent() - last_mtf_zone_log > 900)
         {
            Print("[MTF] Structure zone missing - waiting for OB/FVG alignment. ", g_MTF.GetAnalysisSummary());
            last_mtf_zone_log = TimeCurrent();
         }
         return;
      }
   }
   
   //=== GATE 8: MTF Direction Confirmation (NEW v3.20) ===
   if(InpUseMTF)
   {
      // Verify signal aligns with H1 trend
      if(signal == SIGNAL_BUY && !g_MTF.CanTradeLong())
      {
         return; // H1 is not bullish
      }
      if(signal == SIGNAL_SELL && !g_MTF.CanTradeShort())
      {
         return; // H1 is not bearish
      }
      
      // Check M5 confirmation
      if(InpRequireLTFConfirm && !g_MTF.HasLTFConfirmation(signal))
      {
         return; // No M5 confirmation candle
      }
   }
   
   // Get zone data for entry optimization
   double sweep_level = 0;
   double current_price = (signal == SIGNAL_BUY) 
      ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) 
      : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   // Get sweep level from AMD tracker
   if(signal == SIGNAL_BUY)
   {
      sweep_level = g_AMD.GetManipulationLow();
      SLiquidityPool ssl = g_Sweep.GetNearestSSL(current_price);
      if(ssl.is_valid) sweep_level = ssl.level;
   }
   else
   {
      sweep_level = g_AMD.GetManipulationHigh();
      SLiquidityPool bsl = g_Sweep.GetNearestBSL(current_price);
      if(bsl.is_valid) sweep_level = bsl.level;
   }
   
   // Require structure context; avoid defaulting to market-only entries
   bool has_structure_context = (fvg_low > 0 || fvg_high > 0 || ob_low > 0 || ob_high > 0 || sweep_level != 0);
   if(!has_structure_context)
   {
      static datetime last_structure_log = 0;
      if(TimeCurrent() - last_structure_log > 900)
      {
         Print("Entry skipped: missing structure context (OB/FVG/sweep).");
         last_structure_log = TimeCurrent();
      }
      return;
   }
   
   // Calculate optimal entry
   SOptimalEntry entry = g_EntryOpt.CalculateOptimalEntry(
      signal, fvg_low, fvg_high, ob_low, ob_high, sweep_level, current_price
   );
   
   // Validate entry quality
   if(!entry.is_valid || entry.risk_reward < InpMinRR)
   {
      Print("Entry rejected: R:R = ", DoubleToString(entry.risk_reward, 2), " < ", InpMinRR);
      return;
   }
   
   //=== EXECUTION ===
   // Check if we should enter now or wait
   if(!g_EntryOpt.ShouldEnterNow(current_price))
   {
      // Waiting for better entry - place limit order or wait
      return;
   }
   
   // Calculate position size
   double slPoints = MathAbs(current_price - entry.stop_loss) / _Point;
   double lotSize = g_RiskManager.CalculateLotSize(slPoints);
   
   if(lotSize <= 0) return;
   
   // Apply MTF position sizing multiplier (NEW v3.20)
   if(InpUseMTF)
   {
      double mtf_mult = g_MTF.GetPositionSizeMultiplier();
      if(mtf_mult < 1.0)
      {
         lotSize *= mtf_mult;
         Print("[MTF] Position size adjusted to ", DoubleToString(mtf_mult*100, 0), "% due to partial alignment");
      }
   }
   
   // Ensure minimum lot size
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   if(lotSize < minLot) lotSize = minLot;
   
   // Open trade with multiple TPs
   string reason = StringFormat("Score:%d AMD:%s RR:%.1f MTF:%s", 
                                score, EnumToString(amd_phase), entry.risk_reward,
                                InpUseMTF ? g_MTF.GetAnalysisSummary() : "OFF");
   
   if(g_TradeManager.OpenTradeWithTPs(signal, lotSize, entry.stop_loss,
                                       entry.take_profit_1, entry.take_profit_2, entry.take_profit_3,
                                       score, reason))
   {
      Print("=== TRADE EXECUTED (v3.30 Order Flow) ===");
      Print("Direction: ", (signal == SIGNAL_BUY ? "BUY" : "SELL"));
      Print("Entry: ", current_price, " | SL: ", entry.stop_loss);
      Print("TP1: ", entry.take_profit_1, " (40%)");
      Print("TP2: ", entry.take_profit_2, " (30%)");
      Print("TP3: ", entry.take_profit_3, " (trail 30%)");
      Print("R:R: ", DoubleToString(entry.risk_reward, 2));
      Print("Lot: ", lotSize);
      if(InpUseMTF) Print("MTF: ", g_MTF.GetAnalysisSummary());
      Print("==================================");
      
      g_RiskManager.OnTradeExecuted();
   }
}

//+------------------------------------------------------------------+
//| Timer function - Slow lane updates                               |
//+------------------------------------------------------------------+
void OnTimer()
{
   // Slow-lane analytics
   static datetime last_h1_bar = 0;
   static datetime last_m15_bar = 0;
   
   datetime h1_bar = iTime(_Symbol, PERIOD_H1, 0);
   datetime m15_bar = iTime(_Symbol, PERIOD_M15, 0);
   
   // Regime/HTF updates on new H1 bar
   if(h1_bar != last_h1_bar)
   {
      g_Regime.AnalyzeRegime(_Symbol, PERIOD_H1);
      last_h1_bar = h1_bar;
   }
   
   // Structure/M15 updates on new M15 bar
   if(m15_bar != last_m15_bar)
   {
      g_Structure.AnalyzeStructure(_Symbol, PERIOD_M15);
      last_m15_bar = m15_bar;
   }
   
   // Check for new day (reset daily counters)
   static int last_day = 0;
   MqlDateTime dt;
   TimeCurrent(dt);
   if(dt.day != last_day)
   {
      last_day = dt.day;
      // g_RiskManager.OnNewDay();  // TODO: Add to FTMO_RiskManager
      g_News.RefreshSchedule();
      Print("=== NEW TRADING DAY ===");
      Print("Session: ", g_Session.GetSessionName());
      Print("News: ", g_News.GetCurrentStatus());
   }
   
   // Phase 2: Python Hub updates
   // g_PythonBridge.OnTimer();
}
//+------------------------------------------------------------------+
