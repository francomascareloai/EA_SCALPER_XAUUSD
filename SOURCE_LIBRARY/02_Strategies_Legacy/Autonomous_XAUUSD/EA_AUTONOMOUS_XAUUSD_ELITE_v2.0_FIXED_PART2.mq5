// === CONTINUATION OF EA_AUTONOMOUS_XAUUSD_ELITE_v2.0_FIXED ===
// === PART 2: TRADING LOGIC AND ANALYSIS FUNCTIONS ===

void SearchForTradingOpportunities()
{
    SConfluenceSignal signal = GenerateConfluenceSignal();
    
    if(signal.signal_type == SIGNAL_NONE) return;
    if(signal.confidence_score < InpConfluenceThreshold) 
    {
        Print("Signal rejected - Confidence: ", DoubleToString(signal.confidence_score, 1), 
              "% < Threshold: ", DoubleToString(InpConfluenceThreshold, 1), "%");
        return;
    }
    
    if(!ValidateAllFilters(signal)) return;
    
    ExecuteTrade(signal);
}

SConfluenceSignal GenerateConfluenceSignal()
{
    SConfluenceSignal signal;
    
    // Initialize signal
    signal.signal_type = SIGNAL_NONE;
    signal.confidence_score = 0.0;
    signal.entry_price = 0.0;
    signal.stop_loss = 0.0;
    signal.take_profit = 0.0;
    signal.risk_reward_ratio = 0.0;
    
    // Calculate individual scores
    signal.orderblock_score = CalculateOrderBlockScore();
    signal.fvg_score = CalculateFVGScore();
    signal.liquidity_score = CalculateLiquidityScore();
    signal.structure_score = CalculateStructureScore();
    signal.priceaction_score = CalculatePriceActionScore();
    signal.timeframe_score = CalculateTimeframeScore();
    
    // Calculate weighted confluence score
    signal.confidence_score = 
        signal.orderblock_score * 0.25 +
        signal.fvg_score * 0.20 +
        signal.liquidity_score * 0.20 +
        signal.structure_score * 0.15 +
        signal.priceaction_score * 0.10 +
        signal.timeframe_score * 0.10;
    
    // Determine signal direction
    signal.signal_type = DetermineSignalDirection();
    
    // Calculate trade parameters if signal is valid
    if(signal.signal_type != SIGNAL_NONE && signal.confidence_score >= InpConfluenceThreshold)
    {
        CalculateTradeParameters(signal);
    }
    
    return signal;
}

double CalculateOrderBlockScore()
{
    if(!InpEnableOrderBlocks) return 0.0;
    
    double score = 0.0;
    MqlRates rates[20];
    
    if(CopyRates(_Symbol, PERIOD_M15, 0, 20, rates) <= 0) return 0.0;
    ArraySetAsSeries(rates, true);
    
    // Simple order block detection
    for(int i = 3; i < 15; i++)
    {
        // Bullish order block
        if(rates[i].close < rates[i].open && // Bearish candle
           rates[i-1].close > rates[i-1].open && // Followed by bullish
           rates[i].low < rates[i+1].low && rates[i].low < rates[i-1].low)
        {
            double distance = MathAbs(symbol_info.Bid() - rates[i].low);
            if(distance <= 100 * _Point) // Within 10 pips
            {
                score += 60.0;
                break;
            }
        }
        
        // Bearish order block
        if(rates[i].close > rates[i].open && // Bullish candle
           rates[i-1].close < rates[i-1].open && // Followed by bearish
           rates[i].high > rates[i+1].high && rates[i].high > rates[i-1].high)
        {
            double distance = MathAbs(symbol_info.Bid() - rates[i].high);
            if(distance <= 100 * _Point) // Within 10 pips
            {
                score += 60.0;
                break;
            }
        }
    }
    
    return MathMin(score, 100.0);
}

double CalculateFVGScore()
{
    if(!InpEnableFVG) return 0.0;
    
    double score = 0.0;
    MqlRates rates[10];
    
    if(CopyRates(_Symbol, PERIOD_M15, 0, 10, rates) <= 0) return 0.0;
    ArraySetAsSeries(rates, true);
    
    // Look for Fair Value Gaps
    for(int i = 2; i < 8; i++)
    {
        // Bullish FVG
        if(rates[i+1].high < rates[i-1].low)
        {
            double gap_size = rates[i-1].low - rates[i+1].high;
            double current_price = symbol_info.Bid();
            
            if(current_price >= rates[i+1].high && current_price <= rates[i-1].low)
            {
                score += 50.0;
                if(gap_size > 20 * _Point) score += 20.0; // Large gap bonus
                break;
            }
        }
        
        // Bearish FVG
        if(rates[i+1].low > rates[i-1].high)
        {
            double gap_size = rates[i+1].low - rates[i-1].high;
            double current_price = symbol_info.Bid();
            
            if(current_price <= rates[i+1].low && current_price >= rates[i-1].high)
            {
                score += 50.0;
                if(gap_size > 20 * _Point) score += 20.0; // Large gap bonus
                break;
            }
        }
    }
    
    return MathMin(score, 100.0);
}

double CalculateLiquidityScore()
{
    if(!InpEnableLiquidity) return 0.0;
    
    double score = 0.0;
    MqlRates rates[50];
    
    if(CopyRates(_Symbol, PERIOD_H1, 0, 50, rates) <= 0) return 0.0;
    ArraySetAsSeries(rates, true);
    
    double current_price = symbol_info.Bid();
    
    // Look for recent highs and lows (liquidity zones)
    for(int i = 5; i < 45; i++)
    {
        // High liquidity (equal highs)
        if(rates[i].high >= rates[i-1].high && rates[i].high >= rates[i+1].high)
        {
            double distance = MathAbs(current_price - rates[i].high);
            if(distance <= 150 * _Point) // Within 15 pips
            {
                score += 30.0;
            }
        }
        
        // Low liquidity (equal lows)
        if(rates[i].low <= rates[i-1].low && rates[i].low <= rates[i+1].low)
        {
            double distance = MathAbs(current_price - rates[i].low);
            if(distance <= 150 * _Point) // Within 15 pips
            {
                score += 30.0;
            }
        }
    }
    
    return MathMin(score, 100.0);
}

double CalculateStructureScore()
{
    double score = 0.0;
    
    double ema_fast[3], ema_medium[3], ema_slow[3];
    
    if(CopyBuffer(h_ema_fast, 0, 0, 3, ema_fast) <= 0 ||
       CopyBuffer(h_ema_medium, 0, 0, 3, ema_medium) <= 0 ||
       CopyBuffer(h_ema_slow, 0, 0, 3, ema_slow) <= 0)
        return 0.0;
    
    ArraySetAsSeries(ema_fast, true);
    ArraySetAsSeries(ema_medium, true);
    ArraySetAsSeries(ema_slow, true);
    
    // EMA alignment
    if(ema_fast[0] > ema_medium[0] && ema_medium[0] > ema_slow[0])
        score += 40.0; // Bullish alignment
    else if(ema_fast[0] < ema_medium[0] && ema_medium[0] < ema_slow[0])
        score += 40.0; // Bearish alignment
    
    // Fresh crossover
    if((ema_fast[1] <= ema_medium[1] && ema_fast[0] > ema_medium[0]) ||
       (ema_fast[1] >= ema_medium[1] && ema_fast[0] < ema_medium[0]))
        score += 30.0;
    
    // Price position
    double current_price = symbol_info.Bid();
    if((current_price > ema_fast[0] && ema_fast[0] > ema_medium[0]) ||
       (current_price < ema_fast[0] && ema_fast[0] < ema_medium[0]))
        score += 30.0;
    
    return MathMin(score, 100.0);
}

double CalculatePriceActionScore()
{
    if(!InpEnablePriceAction) return 0.0;
    
    double score = 0.0;
    MqlRates rates[5];
    
    if(CopyRates(_Symbol, PERIOD_M15, 0, 5, rates) <= 0) return 0.0;
    ArraySetAsSeries(rates, true);
    
    // Bullish engulfing
    if(IsBullishEngulfing(rates, 1)) score += 35.0;
    
    // Bearish engulfing
    if(IsBearishEngulfing(rates, 1)) score += 35.0;
    
    // Pin bars
    if(IsBullishPinBar(rates, 1)) score += 30.0;
    if(IsBearishPinBar(rates, 1)) score += 30.0;
    
    // Doji
    if(IsDoji(rates, 1)) score += 20.0;
    
    return MathMin(score, 100.0);
}

double CalculateTimeframeScore()
{
    double score = 0.0;
    
    if(InpUseWeeklyBias && IsWeeklyBiasAligned()) score += 30.0;
    if(InpUseDailyTrend && IsDailyTrendValid()) score += 25.0;
    if(InpUseH4Structure && IsH4StructureValid()) score += 20.0;
    if(InpUseH1Setup && IsH1SetupValid()) score += 15.0;
    if(InpUseM15Execution && IsM15ExecutionValid()) score += 10.0;
    
    return MathMin(score, 100.0);
}

ENUM_SIGNAL_TYPE DetermineSignalDirection()
{
    double ema_fast[1], ema_medium[1];
    double rsi[1];
    
    if(CopyBuffer(h_ema_fast, 0, 0, 1, ema_fast) <= 0 ||
       CopyBuffer(h_ema_medium, 0, 0, 1, ema_medium) <= 0 ||
       CopyBuffer(h_rsi, 0, 0, 1, rsi) <= 0)
        return SIGNAL_NONE;
    
    double current_price = symbol_info.Ask();
    
    // Bullish conditions
    if(ema_fast[0] > ema_medium[0] && 
       current_price > ema_fast[0] && 
       rsi[0] > 30 && rsi[0] < 70)
        return SIGNAL_BUY;
    
    // Bearish conditions
    if(ema_fast[0] < ema_medium[0] && 
       current_price < ema_fast[0] && 
       rsi[0] > 30 && rsi[0] < 70)
        return SIGNAL_SELL;
    
    return SIGNAL_NONE;
}

void CalculateTradeParameters(SConfluenceSignal& signal)
{
    if(signal.signal_type == SIGNAL_BUY)
    {
        signal.entry_price = symbol_info.Ask();
        signal.stop_loss = signal.entry_price - InpStopLoss * _Point;
        signal.take_profit = signal.entry_price + InpTakeProfit * _Point;
    }
    else if(signal.signal_type == SIGNAL_SELL)
    {
        signal.entry_price = symbol_info.Bid();
        signal.stop_loss = signal.entry_price + InpStopLoss * _Point;
        signal.take_profit = signal.entry_price - InpTakeProfit * _Point;
    }
    
    double sl_distance = MathAbs(signal.entry_price - signal.stop_loss);
    double tp_distance = MathAbs(signal.take_profit - signal.entry_price);
    
    if(sl_distance > 0)
        signal.risk_reward_ratio = tp_distance / sl_distance;
}

bool ValidateAllFilters(const SConfluenceSignal& signal)
{
    if(!ValidateSessionFilter()) return false;
    if(!ValidateNewsFilter()) return false;
    if(!ValidateSpreadFilter()) return false;
    if(!ValidateRiskFilter(signal)) return false;
    
    return true;
}

bool ValidateSessionFilter()
{
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    
    bool session_ok = false;
    
    if(InpTradeLondonSession && dt.hour >= InpLondonStart && dt.hour <= InpLondonEnd)
        session_ok = true;
    
    if(InpTradeNYSession && dt.hour >= InpNYStart && dt.hour <= InpNYEnd)
        session_ok = true;
    
    if(InpTradeAsianSession && (dt.hour >= 0 && dt.hour <= 7))
        session_ok = true;
    
    return session_ok;
}

bool ValidateNewsFilter()
{
    if(!InpEnableNewsFilter) return true;
    
    return !IsHighImpactNewsTime();
}

bool ValidateSpreadFilter()
{
    double spread = symbol_info.Spread() * _Point;
    return (spread <= InpMaxSpread * _Point);
}

bool ValidateRiskFilter(const SConfluenceSignal& signal)
{
    double potential_loss = CalculatePotentialLoss(signal);
    double max_risk = account_info.Balance() * (InpRiskPercent / 100.0);
    
    return (potential_loss <= max_risk);
}

double CalculatePotentialLoss(const SConfluenceSignal& signal)
{
    double lot_size = CalculateLotSize(signal);
    double sl_distance = MathAbs(signal.entry_price - signal.stop_loss);
    
    return sl_distance * lot_size * symbol_info.TickValue() / _Point;
}

void ExecuteTrade(const SConfluenceSignal& signal)
{
    double lot_size = CalculateLotSize(signal);
    
    if(lot_size < symbol_info.LotsMin() || lot_size > symbol_info.LotsMax())
    {
        Print("Invalid lot size: ", lot_size);
        return;
    }
    
    bool result = false;
    
    if(signal.signal_type == SIGNAL_BUY)
    {
        result = trade.Buy(lot_size, _Symbol, signal.entry_price, 
                          signal.stop_loss, signal.take_profit, InpComment);
    }
    else if(signal.signal_type == SIGNAL_SELL)
    {
        result = trade.Sell(lot_size, _Symbol, signal.entry_price, 
                           signal.stop_loss, signal.take_profit, InpComment);
    }
    
    if(result)
    {
        g_trades_today++;
        g_ftmo_compliance.trades_today_count++;
        g_total_trades++;
        
        Print("✅ Trade executed successfully!");
        Print("Type: ", (signal.signal_type == SIGNAL_BUY ? "BUY" : "SELL"));
        Print("Confidence: ", DoubleToString(signal.confidence_score, 1), "%");
        Print("Lot Size: ", DoubleToString(lot_size, 2));
        Print("Entry: ", DoubleToString(signal.entry_price, 5));
        Print("SL: ", DoubleToString(signal.stop_loss, 5));
        Print("TP: ", DoubleToString(signal.take_profit, 5));
        Print("R:R: ", DoubleToString(signal.risk_reward_ratio, 2));
    }
    else
    {
        Print("❌ Trade execution failed: ", trade.ResultRetcode(), " - ", trade.ResultComment());
    }
}

double CalculateLotSize(const SConfluenceSignal& signal)
{
    double lot_size = InpLotSize;
    
    if(InpLotMethod == LOT_PERCENT_RISK)
    {
        double balance = account_info.Balance();
        double risk_amount = balance * (InpRiskPercent / 100.0);
        double sl_distance = MathAbs(signal.entry_price - signal.stop_loss);
        
        if(sl_distance > 0)
        {
            double tick_value = symbol_info.TickValue();
            double tick_size = symbol_info.TickSize();
            
            lot_size = risk_amount / (sl_distance / tick_size * tick_value);
        }
    }
    
    // Apply FTMO lot size limits
    double max_lot = account_info.Balance() * 0.0001; // Conservative 0.01% per $100
    lot_size = MathMin(lot_size, max_lot);
    
    // Normalize lot size
    double lot_step = symbol_info.LotsStep();
    lot_size = MathFloor(lot_size / lot_step) * lot_step;
    
    // Ensure within limits
    lot_size = MathMax(lot_size, symbol_info.LotsMin());
    lot_size = MathMin(lot_size, symbol_info.LotsMax());
    
    return lot_size;
}

void ManagePositions()
{
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(position_info.SelectByIndex(i))
        {
            if(position_info.Magic() == InpMagicNumber)
            {
                ulong ticket = position_info.Ticket();
                
                // Move to breakeven at 1:1 R:R
                MoveToBreakeven(ticket);
                
                // Take partial profit at 1.5:1 R:R
                TakePartialProfit(ticket);
                
                // Update trailing stop
                UpdateTrailingStop(ticket);
            }
        }
    }
}

void MoveToBreakeven(ulong ticket)
{
    if(!position_info.SelectByTicket(ticket)) return;
    
    double entry_price = position_info.PriceOpen();
    double current_sl = position_info.StopLoss();
    double current_price = (position_info.PositionType() == POSITION_TYPE_BUY) ? 
                          symbol_info.Bid() : symbol_info.Ask();
    
    double distance = MathAbs(current_price - entry_price);
    double sl_distance = MathAbs(entry_price - current_sl);
    
    // Move to breakeven when profit = 1x risk
    if(distance >= sl_distance * InpBreakevenRR)
    {
        if((position_info.PositionType() == POSITION_TYPE_BUY && current_sl < entry_price) ||
           (position_info.PositionType() == POSITION_TYPE_SELL && current_sl > entry_price))
        {
            trade.PositionModify(ticket, entry_price, position_info.TakeProfit());
            Print("Position ", ticket, " moved to breakeven");
        }
    }
}

void TakePartialProfit(ulong ticket)
{
    if(!position_info.SelectByTicket(ticket)) return;
    
    double entry_price = position_info.PriceOpen();
    double current_sl = position_info.StopLoss();
    double current_price = (position_info.PositionType() == POSITION_TYPE_BUY) ? 
                          symbol_info.Bid() : symbol_info.Ask();
    
    double distance = MathAbs(current_price - entry_price);
    double sl_distance = MathAbs(entry_price - current_sl);
    
    // Take partial profit when profit = 1.5x risk
    if(distance >= sl_distance * InpPartialProfitRR)
    {
        double partial_volume = position_info.Volume() * 0.5; // Close 50%
        
        if(partial_volume >= symbol_info.LotsMin())
        {
            trade.PositionClosePartial(ticket, partial_volume);
            Print("Partial profit taken on position ", ticket);
        }
    }
}

void UpdateTrailingStop(ulong ticket)
{
    if(!position_info.SelectByTicket(ticket)) return;
    
    double entry_price = position_info.PriceOpen();
    double current_sl = position_info.StopLoss();
    double current_price = (position_info.PositionType() == POSITION_TYPE_BUY) ? 
                          symbol_info.Bid() : symbol_info.Ask();
    
    double distance = MathAbs(current_price - entry_price);
    double sl_distance = MathAbs(entry_price - current_sl);
    
    // Start trailing when profit = 2x risk
    if(distance >= sl_distance * InpTrailingStartRR)
    {
        double atr[1];
        if(CopyBuffer(h_atr_m15, 0, 0, 1, atr) > 0)
        {
            double trail_distance = atr[0] * 1.5; // 1.5x ATR trailing distance
            double new_sl = 0;
            
            if(position_info.PositionType() == POSITION_TYPE_BUY)
            {
                new_sl = current_price - trail_distance;
                if(new_sl > current_sl)
                {
                    trade.PositionModify(ticket, new_sl, position_info.TakeProfit());
                    Print("Trailing stop updated for BUY position ", ticket);
                }
            }
            else
            {
                new_sl = current_price + trail_distance;
                if(new_sl < current_sl)
                {
                    trade.PositionModify(ticket, new_sl, position_info.TakeProfit());
                    Print("Trailing stop updated for SELL position ", ticket);
                }
            }
        }
    }
}