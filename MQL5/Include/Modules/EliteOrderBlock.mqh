//+------------------------------------------------------------------+
//|                                              EliteOrderBlock.mqh |
//|                                          EA_SCALPER_XAUUSD v1.00 |
//|                              Phase 1: Core MQL5 Modular Component |
//+------------------------------------------------------------------+
#property copyright "EA_SCALPER_XAUUSD"
#property version   "1.00"
#property strict

//+------------------------------------------------------------------+
//| Order Block Type                                                 |
//+------------------------------------------------------------------+
enum ENUM_OB_TYPE
{
   OB_TYPE_NONE = 0,
   OB_TYPE_BULLISH = 1,
   OB_TYPE_BEARISH = 2,
   OB_TYPE_BREAKER = 3
};

//+------------------------------------------------------------------+
//| Order Block State                                                |
//+------------------------------------------------------------------+
enum ENUM_OB_STATE
{
   OB_STATE_FRESH = 0,
   OB_STATE_TESTED = 1,
   OB_STATE_MITIGATED = 2
};

//+------------------------------------------------------------------+
//| Order Block Data Structure                                       |
//+------------------------------------------------------------------+
struct SOrderBlock
{
   datetime          time;
   double            high;
   double            low;
   double            entry;
   ENUM_OB_TYPE      type;
   ENUM_OB_STATE     state;
   double            score;
   double            displacement;
   int               touch_count;
   bool              is_fresh;
   bool              has_fvg;
   bool              has_liquidity;
};

//+------------------------------------------------------------------+
//| Class: CEliteOrderBlockModule                                    |
//| PRD 5.2: Order Block detection with score 0-100 for SignalScoring|
//+------------------------------------------------------------------+
class CEliteOrderBlockModule
{
private:
   string            m_symbol;
   ENUM_TIMEFRAMES   m_timeframe;
   
   double            m_min_displacement;
   double            m_min_body_ratio;
   double            m_volume_mult;
   int               m_lookback;
   
   SOrderBlock       m_blocks[];
   int               m_block_count;
   int               m_max_blocks;
   
   double            m_last_score;
   ENUM_OB_TYPE      m_last_signal;
   double            m_last_entry;
   double            m_last_sl;
   
   double            m_point;
   
public:
                     CEliteOrderBlockModule();
                    ~CEliteOrderBlockModule();
   
   bool              Initialize(string symbol = NULL, ENUM_TIMEFRAMES tf = PERIOD_M15);
   bool              Calculate();
   double            GetScore() const { return m_last_score; }
   
   ENUM_OB_TYPE      GetSignalType() const { return m_last_signal; }
   double            GetEntryPrice() const { return m_last_entry; }
   double            GetStopLoss() const { return m_last_sl; }
   int               GetBlockCount() const { return m_block_count; }
   bool              GetBlock(int index, SOrderBlock &ob) const;
   
   void              SetMinDisplacement(double pts) { m_min_displacement = pts; }
   void              SetMinBodyRatio(double ratio) { m_min_body_ratio = ratio; }
   void              SetVolumeMult(double mult) { m_volume_mult = mult; }
   void              SetLookback(int bars) { m_lookback = bars; }

private:
   bool              DetectOrderBlocks(const MqlRates &rates[], int count);
   bool              IsBullishOB(const MqlRates &rates[], int idx);
   bool              IsBearishOB(const MqlRates &rates[], int idx);
   double            CalcDisplacement(const MqlRates &rates[], int idx, ENUM_OB_TYPE type);
   double            CalcBodyRatio(const MqlRates &rate);
   bool              HasVolumeSpike(const MqlRates &rates[], int idx);
   double            ScoreOrderBlock(const SOrderBlock &ob);
   void              UpdateBlockStates(double price);
   void              SortByScore();
   bool              AddBlock(const SOrderBlock &ob);
};

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CEliteOrderBlockModule::CEliteOrderBlockModule() :
   m_symbol(NULL),
   m_timeframe(PERIOD_M15),
   m_min_displacement(150.0),
   m_min_body_ratio(0.5),
   m_volume_mult(1.3),
   m_lookback(100),
   m_block_count(0),
   m_max_blocks(30),
   m_last_score(0.0),
   m_last_signal(OB_TYPE_NONE),
   m_last_entry(0.0),
   m_last_sl(0.0),
   m_point(0.0)
{
   ArrayResize(m_blocks, m_max_blocks);
}

//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CEliteOrderBlockModule::~CEliteOrderBlockModule()
{
   ArrayFree(m_blocks);
}

//+------------------------------------------------------------------+
//| Initialize module                                                |
//+------------------------------------------------------------------+
bool CEliteOrderBlockModule::Initialize(string symbol = NULL, ENUM_TIMEFRAMES tf = PERIOD_M15)
{
   m_symbol = (symbol == NULL) ? _Symbol : symbol;
   m_timeframe = tf;
   m_point = SymbolInfoDouble(m_symbol, SYMBOL_POINT);
   
   if(m_point <= 0)
   {
      Print("[EliteOB] Failed to get SYMBOL_POINT for ", m_symbol);
      return false;
   }
   
   m_block_count = 0;
   m_last_score = 0.0;
   m_last_signal = OB_TYPE_NONE;
   
   return true;
}

//+------------------------------------------------------------------+
//| Main calculation - detect OBs and return best score              |
//+------------------------------------------------------------------+
bool CEliteOrderBlockModule::Calculate()
{
   m_last_score = 0.0;
   m_last_signal = OB_TYPE_NONE;
   m_last_entry = 0.0;
   m_last_sl = 0.0;
   
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   
   int copied = CopyRates(m_symbol, m_timeframe, 0, m_lookback, rates);
   if(copied < 20)
   {
      Print("[EliteOB] Insufficient data: ", copied, " bars");
      return false;
   }
   
   double bid = SymbolInfoDouble(m_symbol, SYMBOL_BID);
   UpdateBlockStates(bid);
   
   if(!DetectOrderBlocks(rates, copied))
      return false;
   
   if(m_block_count == 0)
      return true;
   
   SortByScore();
   
   for(int i = 0; i < m_block_count; i++)
   {
      if(m_blocks[i].state != OB_STATE_FRESH && m_blocks[i].state != OB_STATE_TESTED)
         continue;
         
      bool in_zone = false;
      if(m_blocks[i].type == OB_TYPE_BULLISH && bid >= m_blocks[i].low && bid <= m_blocks[i].high)
         in_zone = true;
      else if(m_blocks[i].type == OB_TYPE_BEARISH && bid >= m_blocks[i].low && bid <= m_blocks[i].high)
         in_zone = true;
      
      if(in_zone || m_blocks[i].is_fresh)
      {
         m_last_score = m_blocks[i].score;
         m_last_signal = m_blocks[i].type;
         m_last_entry = m_blocks[i].entry;
         m_last_sl = (m_blocks[i].type == OB_TYPE_BULLISH) ? m_blocks[i].low : m_blocks[i].high;
         break;
      }
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Detect Order Blocks from rate data                               |
//+------------------------------------------------------------------+
bool CEliteOrderBlockModule::DetectOrderBlocks(const MqlRates &rates[], int count)
{
   m_block_count = 0;
   
   for(int i = 3; i < count - 3; i++)
   {
      if(IsBullishOB(rates, i))
      {
         SOrderBlock ob;
         ob.time = rates[i].time;
         ob.high = rates[i].high;
         ob.low = rates[i].low;
         ob.entry = ob.low + (ob.high - ob.low) * 0.5;
         ob.type = OB_TYPE_BULLISH;
         ob.state = OB_STATE_FRESH;
         ob.displacement = CalcDisplacement(rates, i, OB_TYPE_BULLISH);
         ob.touch_count = 0;
         ob.is_fresh = true;
         ob.has_fvg = false;
         ob.has_liquidity = (ob.displacement > m_min_displacement * m_point * 1.5);
         ob.score = ScoreOrderBlock(ob);
         
         if(ob.score >= 50.0)
            AddBlock(ob);
      }
      
      if(IsBearishOB(rates, i))
      {
         SOrderBlock ob;
         ob.time = rates[i].time;
         ob.high = rates[i].high;
         ob.low = rates[i].low;
         ob.entry = ob.high - (ob.high - ob.low) * 0.5;
         ob.type = OB_TYPE_BEARISH;
         ob.state = OB_STATE_FRESH;
         ob.displacement = CalcDisplacement(rates, i, OB_TYPE_BEARISH);
         ob.touch_count = 0;
         ob.is_fresh = true;
         ob.has_fvg = false;
         ob.has_liquidity = (ob.displacement > m_min_displacement * m_point * 1.5);
         ob.score = ScoreOrderBlock(ob);
         
         if(ob.score >= 50.0)
            AddBlock(ob);
      }
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Detect Bullish Order Block                                       |
//+------------------------------------------------------------------+
bool CEliteOrderBlockModule::IsBullishOB(const MqlRates &rates[], int idx)
{
   if(idx < 1 || idx >= ArraySize(rates) - 3)
      return false;
   
   double body = rates[idx].close - rates[idx].open;
   if(body >= 0)
      return false;
   
   if(CalcBodyRatio(rates[idx]) < m_min_body_ratio)
      return false;
   
   bool has_displacement = false;
   for(int j = idx - 1; j >= MathMax(0, idx - 5); j--)
   {
      if(rates[j].close > rates[idx].high)
      {
         double disp = rates[j].close - rates[idx].high;
         if(disp >= m_min_displacement * m_point)
         {
            has_displacement = true;
            break;
         }
      }
   }
   
   if(!has_displacement)
      return false;
   
   if(!HasVolumeSpike(rates, idx))
      return false;
   
   return true;
}

//+------------------------------------------------------------------+
//| Detect Bearish Order Block                                       |
//+------------------------------------------------------------------+
bool CEliteOrderBlockModule::IsBearishOB(const MqlRates &rates[], int idx)
{
   if(idx < 1 || idx >= ArraySize(rates) - 3)
      return false;
   
   double body = rates[idx].close - rates[idx].open;
   if(body <= 0)
      return false;
   
   if(CalcBodyRatio(rates[idx]) < m_min_body_ratio)
      return false;
   
   bool has_displacement = false;
   for(int j = idx - 1; j >= MathMax(0, idx - 5); j--)
   {
      if(rates[j].close < rates[idx].low)
      {
         double disp = rates[idx].low - rates[j].close;
         if(disp >= m_min_displacement * m_point)
         {
            has_displacement = true;
            break;
         }
      }
   }
   
   if(!has_displacement)
      return false;
   
   if(!HasVolumeSpike(rates, idx))
      return false;
   
   return true;
}

//+------------------------------------------------------------------+
//| Calculate displacement size                                      |
//+------------------------------------------------------------------+
double CEliteOrderBlockModule::CalcDisplacement(const MqlRates &rates[], int idx, ENUM_OB_TYPE type)
{
   double max_disp = 0.0;
   
   for(int j = idx - 1; j >= MathMax(0, idx - 5); j--)
   {
      double disp = 0.0;
      if(type == OB_TYPE_BULLISH)
         disp = rates[j].close - rates[idx].high;
      else
         disp = rates[idx].low - rates[j].close;
      
      if(disp > max_disp)
         max_disp = disp;
   }
   
   return max_disp;
}

//+------------------------------------------------------------------+
//| Calculate body ratio                                             |
//+------------------------------------------------------------------+
double CEliteOrderBlockModule::CalcBodyRatio(const MqlRates &rate)
{
   double range = rate.high - rate.low;
   if(range <= 0)
      return 0.0;
   
   double body = MathAbs(rate.close - rate.open);
   return body / range;
}

//+------------------------------------------------------------------+
//| Check for volume spike                                           |
//+------------------------------------------------------------------+
bool CEliteOrderBlockModule::HasVolumeSpike(const MqlRates &rates[], int idx)
{
   if(idx >= ArraySize(rates) - 10)
      return true;
   
   long current_vol = rates[idx].tick_volume;
   long avg_vol = 0;
   int count = 0;
   
   for(int i = idx + 1; i < MathMin(idx + 11, ArraySize(rates)); i++)
   {
      avg_vol += rates[i].tick_volume;
      count++;
   }
   
   if(count == 0)
      return true;
   
   avg_vol /= count;
   return current_vol >= (long)(avg_vol * m_volume_mult);
}

//+------------------------------------------------------------------+
//| Score Order Block (0-100)                                        |
//+------------------------------------------------------------------+
double CEliteOrderBlockModule::ScoreOrderBlock(const SOrderBlock &ob)
{
   double score = 0.0;
   
   double disp_pts = ob.displacement / m_point;
   if(disp_pts >= 300)
      score += 30.0;
   else if(disp_pts >= 200)
      score += 22.0;
   else if(disp_pts >= 150)
      score += 15.0;
   else if(disp_pts >= 100)
      score += 10.0;
   
   if(ob.is_fresh)
      score += 20.0;
   
   if(ob.has_liquidity)
      score += 15.0;
   
   if(ob.has_fvg)
      score += 15.0;
   
   double range = ob.high - ob.low;
   double range_pts = range / m_point;
   if(range_pts >= 50 && range_pts <= 200)
      score += 10.0;
   else if(range_pts >= 30)
      score += 5.0;
   
   score += 10.0;
   
   return MathMin(score, 100.0);
}

//+------------------------------------------------------------------+
//| Update block states based on current price                       |
//+------------------------------------------------------------------+
void CEliteOrderBlockModule::UpdateBlockStates(double price)
{
   for(int i = 0; i < m_block_count; i++)
   {
      if(m_blocks[i].state == OB_STATE_MITIGATED)
         continue;
      
      if(m_blocks[i].type == OB_TYPE_BULLISH)
      {
         if(price < m_blocks[i].low - 50 * m_point)
         {
            m_blocks[i].state = OB_STATE_MITIGATED;
         }
         else if(price <= m_blocks[i].high && price >= m_blocks[i].low)
         {
            m_blocks[i].state = OB_STATE_TESTED;
            m_blocks[i].is_fresh = false;
            m_blocks[i].touch_count++;
         }
      }
      else if(m_blocks[i].type == OB_TYPE_BEARISH)
      {
         if(price > m_blocks[i].high + 50 * m_point)
         {
            m_blocks[i].state = OB_STATE_MITIGATED;
         }
         else if(price <= m_blocks[i].high && price >= m_blocks[i].low)
         {
            m_blocks[i].state = OB_STATE_TESTED;
            m_blocks[i].is_fresh = false;
            m_blocks[i].touch_count++;
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Sort blocks by score descending                                  |
//+------------------------------------------------------------------+
void CEliteOrderBlockModule::SortByScore()
{
   for(int i = 0; i < m_block_count - 1; i++)
   {
      for(int j = 0; j < m_block_count - i - 1; j++)
      {
         if(m_blocks[j].score < m_blocks[j + 1].score)
         {
            SOrderBlock temp = m_blocks[j];
            m_blocks[j] = m_blocks[j + 1];
            m_blocks[j + 1] = temp;
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Add block to array                                               |
//+------------------------------------------------------------------+
bool CEliteOrderBlockModule::AddBlock(const SOrderBlock &ob)
{
   if(m_block_count >= m_max_blocks)
      return false;
   
   m_blocks[m_block_count] = ob;
   m_block_count++;
   return true;
}

//+------------------------------------------------------------------+
//| Get block by index                                               |
//+------------------------------------------------------------------+
bool CEliteOrderBlockModule::GetBlock(int index, SOrderBlock &ob) const
{
   if(index < 0 || index >= m_block_count)
      return false;
   
   ob = m_blocks[index];
   return true;
}
