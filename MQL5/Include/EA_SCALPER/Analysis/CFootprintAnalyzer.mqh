//+------------------------------------------------------------------+
//|                                          CFootprintAnalyzer.mqh |
//|                           EA_SCALPER_XAUUSD - Singularity Edition |
//|                                                                  |
//|  VERSAO 3.0 - FOOTPRINT/CLUSTER CHART COMPLETO                  |
//|                                                                  |
//|  BASEADO EM:                                                     |
//|  - ATAS Footprint methodology                                    |
//|  - UCluster diagonal imbalance detection                         |
//|  - OrderFlowAnalyzer_v2.mqh (base code)                         |
//|                                                                  |
//|  FEATURES:                                                       |
//|  1. Diagonal Imbalance (ATAS-style)                             |
//|  2. Stacked Imbalance (3+ consecutivos)                         |
//|  3. Absorption Detection (high vol + low delta)                 |
//|  4. Unfinished Auction detection                                |
//|  5. Delta Divergence tracking                                   |
//|  6. POC/VAH/VAL calculation                                     |
//|  7. Trading signal generation                                   |
//+------------------------------------------------------------------+
#property copyright "EA_SCALPER_XAUUSD - Singularity Edition"
#property version   "3.00"
#property strict

#include "../Core/Definitions.mqh"

//+------------------------------------------------------------------+
//| Enums                                                             |
//+------------------------------------------------------------------+
enum ENUM_IMBALANCE_TYPE {
   IMBALANCE_NONE = 0,
   IMBALANCE_BUY,           // Compradores dominam (bullish)
   IMBALANCE_SELL           // Vendedores dominam (bearish)
};

enum ENUM_ABSORPTION_TYPE {
   ABSORPTION_NONE = 0,
   ABSORPTION_BUY,          // Vendas absorvidas por compradores
   ABSORPTION_SELL          // Compras absorvidas por vendedores
};

enum ENUM_AUCTION_TYPE {
   AUCTION_NONE = 0,
   AUCTION_UNFINISHED_UP,   // Close no high, delta positivo
   AUCTION_UNFINISHED_DOWN  // Close no low, delta negativo
};

enum ENUM_FOOTPRINT_SIGNAL {
   FP_SIGNAL_NONE = 0,
   FP_SIGNAL_STRONG_BUY,
   FP_SIGNAL_BUY,
   FP_SIGNAL_WEAK_BUY,
   FP_SIGNAL_NEUTRAL,
   FP_SIGNAL_WEAK_SELL,
   FP_SIGNAL_SELL,
   FP_SIGNAL_STRONG_SELL
};

//+------------------------------------------------------------------+
//| Estrutura de Nivel de Preco (Footprint Cell)                     |
//+------------------------------------------------------------------+
struct SFootprintLevel {
   double price;              // Preco do nivel (normalizado)
   long   bidVolume;          // Volume no bid (vendas agressivas)
   long   askVolume;          // Volume no ask (compras agressivas)
   long   delta;              // askVolume - bidVolume
   int    tickCount;          // Quantidade de ticks
   
   // Imbalances (calculados depois)
   bool   hasBuyImbalance;    // Ask domina (diagonal)
   bool   hasSellImbalance;   // Bid domina (diagonal)
   double imbalanceRatio;     // Ratio do imbalance
   
   void Reset() {
      price = 0;
      bidVolume = 0;
      askVolume = 0;
      delta = 0;
      tickCount = 0;
      hasBuyImbalance = false;
      hasSellImbalance = false;
      imbalanceRatio = 0;
   }
};

//+------------------------------------------------------------------+
//| Estrutura de Stacked Imbalance                                    |
//+------------------------------------------------------------------+
struct SStackedImbalance {
   double startPrice;         // Preco inicial do stack
   double endPrice;           // Preco final do stack
   int    levelCount;         // Quantos niveis consecutivos
   ENUM_IMBALANCE_TYPE type;  // BUY ou SELL
   double avgRatio;           // Ratio medio do stack
   bool   isActive;           // Preco ainda nao violou
   datetime detectionTime;    // Quando foi detectado
   
   void Reset() {
      startPrice = 0;
      endPrice = 0;
      levelCount = 0;
      type = IMBALANCE_NONE;
      avgRatio = 0;
      isActive = false;
      detectionTime = 0;
   }
};

//+------------------------------------------------------------------+
//| Estrutura de Zona de Absorcao                                     |
//+------------------------------------------------------------------+
struct SAbsorptionZone {
   double price;              // Nivel de preco
   long   totalVolume;        // Volume total
   long   delta;              // Delta (deve ser perto de zero)
   double deltaPercent;       // |delta| / totalVolume * 100
   ENUM_ABSORPTION_TYPE type; // Tipo de absorcao
   datetime detectionTime;
   
   void Reset() {
      price = 0;
      totalVolume = 0;
      delta = 0;
      deltaPercent = 0;
      type = ABSORPTION_NONE;
      detectionTime = 0;
   }
};

//+------------------------------------------------------------------+
//| Estrutura de Value Area                                           |
//+------------------------------------------------------------------+
struct SValueArea {
   double poc;                // Point of Control
   double vahigh;             // Value Area High (70%)
   double valow;              // Value Area Low (70%)
   long   pocVolume;          // Volume no POC
   long   totalVolume;        // Volume total
};

//+------------------------------------------------------------------+
//| Estrutura de Sinal de Trading                                     |
//+------------------------------------------------------------------+
struct SFootprintSignal {
   ENUM_FOOTPRINT_SIGNAL signal;
   int    strength;           // 0-100
   
   // Componentes do sinal
   bool   hasStackedBuyImbalance;
   bool   hasStackedSellImbalance;
   bool   hasBuyAbsorption;
   bool   hasSellAbsorption;
   bool   hasUnfinishedAuctionUp;
   bool   hasUnfinishedAuctionDown;
   bool   hasBullishDeltaDivergence;
   bool   hasBearishDeltaDivergence;
   bool   hasPOCDefense;
   
   // Delta info
   long   barDelta;
   long   cumulativeDelta;
   double deltaPercent;
   
   // Value Area
   SValueArea valueArea;
   
   // Niveis sugeridos
   double suggestedEntry;
   double suggestedSL;
   double suggestedTP;
   
   void Reset() {
      signal = FP_SIGNAL_NONE;
      strength = 0;
      hasStackedBuyImbalance = false;
      hasStackedSellImbalance = false;
      hasBuyAbsorption = false;
      hasSellAbsorption = false;
      hasUnfinishedAuctionUp = false;
      hasUnfinishedAuctionDown = false;
      hasBullishDeltaDivergence = false;
      hasBearishDeltaDivergence = false;
      hasPOCDefense = false;
      barDelta = 0;
      cumulativeDelta = 0;
      deltaPercent = 0;
      suggestedEntry = 0;
      suggestedSL = 0;
      suggestedTP = 0;
   }
};

//+------------------------------------------------------------------+
//| Classe Principal: Footprint Analyzer                              |
//+------------------------------------------------------------------+
class CFootprintAnalyzer {
private:
   // Configuracao
   string            m_symbol;
   ENUM_TIMEFRAMES   m_timeframe;
   double            m_clusterSize;        // Tamanho do cluster (ex: 0.50 para XAUUSD)
   double            m_tickSize;
   double            m_point;
   int               m_maxLevels;
   double            m_imbalanceRatio;     // Ratio minimo para imbalance (3.0 = 300%)
   int               m_minStackedLevels;   // Minimo de niveis para stacked (3)
   double            m_absorptionThreshold; // |delta%| < este valor = absorcao
   double            m_volumeMultiplier;   // Volume > avg * este = significativo
   
   // Dados dos niveis
   SFootprintLevel   m_levels[];
   int               m_levelCount;
   
   // Stacked Imbalances detectados
   SStackedImbalance m_stackedImbalances[];
   int               m_stackedCount;
   
   // Absorption Zones detectadas
   SAbsorptionZone   m_absorptionZones[];
   int               m_absorptionCount;
   
   // Tracking
   long              m_cumulativeDelta;
   datetime          m_lastBarTime;
   double            m_lastPrice;
   int               m_ticksProcessed;
   
   // Historico para divergencia
   long              m_deltaHistory[];
   double            m_priceHistory[];
   int               m_historyCount;
   int               m_maxHistory;
   
   // Cache
   bool              m_cacheValid;
   SFootprintSignal  m_cachedSignal;
   datetime          m_cachedBarTime;
   
   // Indicator handles
   int               m_atr_handle;
   
   // Metodos internos de processamento
   int               FindLevel(double price);
   int               AddLevel(double price);
   double            NormalizeToCluster(double price);
   void              ResetBarData();
   int               DetectDirection(const MqlTick &tick);
   
   // Metodos de calculo
   void              CalculateDiagonalImbalances();
   void              DetectStackedImbalances();
   void              DetectAbsorptionZones();
   ENUM_AUCTION_TYPE DetectUnfinishedAuction();
   bool              DetectDeltaDivergence(bool &bullish, bool &bearish);
   bool              DetectPOCDefense();
   void              CalculateValueArea(SValueArea &va);
   void              SortLevelsByPrice();
   
public:
                     CFootprintAnalyzer();
                    ~CFootprintAnalyzer();
   
   // Inicializacao
   bool              Init(string symbol = NULL, ENUM_TIMEFRAMES tf = PERIOD_M5,
                          double clusterSize = 0.50, double imbalanceRatio = 3.0);
   void              Deinit();
   
   // Configuracao
   void              SetClusterSize(double size) { m_clusterSize = size; }
   void              SetImbalanceRatio(double ratio) { m_imbalanceRatio = ratio; }
   void              SetMinStackedLevels(int levels) { m_minStackedLevels = levels; }
   void              SetAbsorptionThreshold(double pct) { m_absorptionThreshold = pct; }
   
   // Processamento
   bool              ProcessBarTicks(int barIndex = 0);
   bool              ProcessTick(const MqlTick &tick);
   void              Update();
   
   // Resultados
   SFootprintSignal  GetSignal();
   SValueArea        GetValueArea();
   long              GetBarDelta();
   long              GetCumulativeDelta();
   double            GetPOC();
   
   // Stacked Imbalances
   int               GetStackedImbalanceCount() { return m_stackedCount; }
   SStackedImbalance GetStackedImbalance(int index);
   bool              HasStackedBuyImbalance();
   bool              HasStackedSellImbalance();
   
   // Absorption Zones
   int               GetAbsorptionZoneCount() { return m_absorptionCount; }
   SAbsorptionZone   GetAbsorptionZone(int index);
   bool              HasBuyAbsorption();
   bool              HasSellAbsorption();
   
   // Trading helpers
   bool              IsBullish();
   bool              IsBearish();
   int               GetConfluenceScore();  // 0-100
   
   // Debug
   void              PrintDiagnostics();
   void              PrintLevels();
   void              PrintStackedImbalances();
};

//+------------------------------------------------------------------+
//| Constructor                                                       |
//+------------------------------------------------------------------+
CFootprintAnalyzer::CFootprintAnalyzer() {
   m_symbol = "";
   m_timeframe = PERIOD_M5;
   m_clusterSize = 0.50;
   m_tickSize = 0.01;
   m_point = 0.01;
   m_maxLevels = 100;
   m_imbalanceRatio = 3.0;
   m_minStackedLevels = 3;
   m_absorptionThreshold = 15.0;
   m_volumeMultiplier = 2.0;
   m_levelCount = 0;
   m_stackedCount = 0;
   m_absorptionCount = 0;
   m_cumulativeDelta = 0;
   m_lastBarTime = 0;
   m_lastPrice = 0;
   m_ticksProcessed = 0;
   m_historyCount = 0;
   m_maxHistory = 20;
   m_cacheValid = false;
   m_cachedBarTime = 0;
   m_atr_handle = INVALID_HANDLE;
}

//+------------------------------------------------------------------+
//| Destructor                                                        |
//+------------------------------------------------------------------+
CFootprintAnalyzer::~CFootprintAnalyzer() {
   Deinit();
}

//+------------------------------------------------------------------+
//| Inicializacao                                                     |
//+------------------------------------------------------------------+
bool CFootprintAnalyzer::Init(string symbol = NULL, ENUM_TIMEFRAMES tf = PERIOD_M5,
                               double clusterSize = 0.50, double imbalanceRatio = 3.0) {
   m_symbol = (symbol == NULL) ? _Symbol : symbol;
   m_timeframe = tf;
   m_clusterSize = clusterSize;
   m_imbalanceRatio = imbalanceRatio;
   
   // Obtem parametros do simbolo
   m_tickSize = SymbolInfoDouble(m_symbol, SYMBOL_TRADE_TICK_SIZE);
   m_point = SymbolInfoDouble(m_symbol, SYMBOL_POINT);
   
   if(m_tickSize == 0) m_tickSize = 0.01;
   if(m_point == 0) m_point = 0.01;
   
   // Ajusta cluster size para ser multiplo do tick size
   if(m_clusterSize < m_tickSize) m_clusterSize = m_tickSize;
   m_clusterSize = MathRound(m_clusterSize / m_tickSize) * m_tickSize;
   
   // Aloca arrays
   ArrayResize(m_levels, m_maxLevels);
   ArrayResize(m_stackedImbalances, 20);
   ArrayResize(m_absorptionZones, 20);
   ArrayResize(m_deltaHistory, m_maxHistory);
   ArrayResize(m_priceHistory, m_maxHistory);
   
   // ATR handle
   m_atr_handle = iATR(m_symbol, m_timeframe, 14);
   
   Print("CFootprintAnalyzer v3.0 Initialized:");
   Print("  Symbol: ", m_symbol);
   Print("  Timeframe: ", EnumToString(m_timeframe));
   Print("  Cluster Size: ", m_clusterSize);
   Print("  Imbalance Ratio: ", m_imbalanceRatio, "x (", m_imbalanceRatio * 100, "%)");
   Print("  Min Stacked Levels: ", m_minStackedLevels);
   
   return true;
}

//+------------------------------------------------------------------+
//| Desinicializacao                                                  |
//+------------------------------------------------------------------+
void CFootprintAnalyzer::Deinit() {
   ArrayFree(m_levels);
   ArrayFree(m_stackedImbalances);
   ArrayFree(m_absorptionZones);
   ArrayFree(m_deltaHistory);
   ArrayFree(m_priceHistory);
   
   if(m_atr_handle != INVALID_HANDLE)
      IndicatorRelease(m_atr_handle);
}

//+------------------------------------------------------------------+
//| Normaliza preco para cluster                                      |
//+------------------------------------------------------------------+
double CFootprintAnalyzer::NormalizeToCluster(double price) {
   return MathRound(price / m_clusterSize) * m_clusterSize;
}

//+------------------------------------------------------------------+
//| Encontra nivel por preco                                          |
//+------------------------------------------------------------------+
int CFootprintAnalyzer::FindLevel(double price) {
   double normPrice = NormalizeToCluster(price);
   for(int i = 0; i < m_levelCount; i++) {
      if(MathAbs(m_levels[i].price - normPrice) < m_clusterSize / 2)
         return i;
   }
   return -1;
}

//+------------------------------------------------------------------+
//| Adiciona novo nivel                                               |
//+------------------------------------------------------------------+
int CFootprintAnalyzer::AddLevel(double price) {
   if(m_levelCount >= m_maxLevels) {
      // Remove nivel com menor volume
      int minIdx = 0;
      long minVol = m_levels[0].bidVolume + m_levels[0].askVolume;
      for(int i = 1; i < m_levelCount; i++) {
         long vol = m_levels[i].bidVolume + m_levels[i].askVolume;
         if(vol < minVol) {
            minVol = vol;
            minIdx = i;
         }
      }
      m_levels[minIdx].Reset();
      m_levels[minIdx].price = NormalizeToCluster(price);
      return minIdx;
   }
   
   int idx = m_levelCount;
   m_levels[idx].Reset();
   m_levels[idx].price = NormalizeToCluster(price);
   m_levelCount++;
   return idx;
}

//+------------------------------------------------------------------+
//| Reset dados da barra                                              |
//+------------------------------------------------------------------+
void CFootprintAnalyzer::ResetBarData() {
   for(int i = 0; i < m_maxLevels; i++)
      m_levels[i].Reset();
   
   m_levelCount = 0;
   m_stackedCount = 0;
   m_absorptionCount = 0;
   m_ticksProcessed = 0;
   m_lastPrice = 0;
   m_cacheValid = false;
}

//+------------------------------------------------------------------+
//| Detecta direcao do tick                                           |
//+------------------------------------------------------------------+
int CFootprintAnalyzer::DetectDirection(const MqlTick &tick) {
   // Prioridade 1: TICK_FLAG (se disponivel)
   bool hasBuy = (tick.flags & TICK_FLAG_BUY) != 0;
   bool hasSell = (tick.flags & TICK_FLAG_SELL) != 0;
   
   if(hasBuy && !hasSell) return 1;
   if(hasSell && !hasBuy) return -1;
   
   // Prioridade 2: Comparacao bid/ask
   double tradePrice = (tick.last > 0) ? tick.last : tick.bid;
   
   if(tick.ask > 0 && tick.bid > 0) {
      double mid = (tick.bid + tick.ask) / 2;
      if(tradePrice >= mid) return 1;  // Trade no ask = buy
      return -1;  // Trade no bid = sell
   }
   
   // Prioridade 3: Comparacao com preco anterior
   if(m_lastPrice > 0) {
      if(tradePrice > m_lastPrice + m_point) return 1;
      if(tradePrice < m_lastPrice - m_point) return -1;
   }
   
   return 0;
}

//+------------------------------------------------------------------+
//| Processa tick individual                                          |
//+------------------------------------------------------------------+
bool CFootprintAnalyzer::ProcessTick(const MqlTick &tick) {
   // Valida tick
   if(tick.bid <= 0 && tick.ask <= 0 && tick.last <= 0)
      return false;
   
   // Verifica nova barra
   datetime barTime = iTime(m_symbol, m_timeframe, 0);
   if(barTime != m_lastBarTime) {
      // Salva historico antes de resetar
      if(m_levelCount > 0) {
         if(m_historyCount >= m_maxHistory) {
            // Shift history
            for(int i = 0; i < m_maxHistory - 1; i++) {
               m_deltaHistory[i] = m_deltaHistory[i+1];
               m_priceHistory[i] = m_priceHistory[i+1];
            }
            m_historyCount = m_maxHistory - 1;
         }
         m_deltaHistory[m_historyCount] = GetBarDelta();
         m_priceHistory[m_historyCount] = iClose(m_symbol, m_timeframe, 1);
         m_historyCount++;
      }
      
      ResetBarData();
      m_lastBarTime = barTime;
   }
   
   // Detecta direcao
   int direction = DetectDirection(tick);
   
   // Determina preco e volume
   double price = (tick.last > 0) ? tick.last : tick.bid;
   long volume = (tick.volume > 0) ? (long)tick.volume : 1;
   
   // Encontra ou cria nivel
   int idx = FindLevel(price);
   if(idx < 0) idx = AddLevel(price);
   
   // Atualiza nivel
   m_levels[idx].tickCount++;
   
   if(direction > 0) {
      m_levels[idx].askVolume += volume;
      m_cumulativeDelta += volume;
   }
   else if(direction < 0) {
      m_levels[idx].bidVolume += volume;
      m_cumulativeDelta -= volume;
   }
   
   // Atualiza delta do nivel
   m_levels[idx].delta = m_levels[idx].askVolume - m_levels[idx].bidVolume;
   
   m_lastPrice = price;
   m_ticksProcessed++;
   m_cacheValid = false;
   
   return true;
}

//+------------------------------------------------------------------+
//| Processa todos os ticks de uma barra                              |
//+------------------------------------------------------------------+
bool CFootprintAnalyzer::ProcessBarTicks(int barIndex = 0) {
   datetime barTime = iTime(m_symbol, m_timeframe, barIndex);
   int barSeconds = PeriodSeconds(m_timeframe);
   datetime barEnd = barTime + barSeconds;
   
   // Se e a mesma barra, nao reprocessa tudo (a menos que seja a barra atual)
   if(barIndex > 0 && barTime == m_lastBarTime && m_levelCount > 0)
      return true;
   
   ResetBarData();
   m_lastBarTime = barTime;
   
   MqlTick ticks[];
   int copied = CopyTicksRange(m_symbol, ticks, COPY_TICKS_ALL,
                               barTime * 1000, barEnd * 1000);
   
   if(copied <= 0) {
      Print("CFootprintAnalyzer: No ticks for bar ", barIndex);
      return false;
   }
   
   // Limita para performance
   int step = (copied > 20000) ? copied / 20000 : 1;
   
   for(int i = 0; i < copied; i += step) {
      ProcessTick(ticks[i]);
   }
   
   // Calcula imbalances e padroes
   SortLevelsByPrice();
   CalculateDiagonalImbalances();
   DetectStackedImbalances();
   DetectAbsorptionZones();
   
   return true;
}

//+------------------------------------------------------------------+
//| Update (chamado em cada nova barra)                               |
//+------------------------------------------------------------------+
void CFootprintAnalyzer::Update() {
   ProcessBarTicks(0);
}

//+------------------------------------------------------------------+
//| Ordena niveis por preco                                           |
//+------------------------------------------------------------------+
void CFootprintAnalyzer::SortLevelsByPrice() {
   // Insertion sort (array pequeno)
   for(int i = 1; i < m_levelCount; i++) {
      SFootprintLevel key = m_levels[i];
      int j = i - 1;
      while(j >= 0 && m_levels[j].price > key.price) {
         m_levels[j + 1] = m_levels[j];
         j--;
      }
      m_levels[j + 1] = key;
   }
}

//+------------------------------------------------------------------+
//| Calcula Imbalances DIAGONAIS (estilo ATAS)                        |
//+------------------------------------------------------------------+
void CFootprintAnalyzer::CalculateDiagonalImbalances() {
   // Reset imbalances
   for(int i = 0; i < m_levelCount; i++) {
      m_levels[i].hasBuyImbalance = false;
      m_levels[i].hasSellImbalance = false;
      m_levels[i].imbalanceRatio = 0;
   }
   
   // Calcula imbalances diagonais
   // Buy Imbalance: Ask[i+1] >= Bid[i] * ratio
   // Sell Imbalance: Bid[i] >= Ask[i+1] * ratio
   for(int i = 0; i < m_levelCount - 1; i++) {
      double askAbove = (double)m_levels[i + 1].askVolume;
      double bidCurrent = (double)m_levels[i].bidVolume;
      
      // Buy Imbalance
      if(bidCurrent > 0 && askAbove / bidCurrent >= m_imbalanceRatio) {
         m_levels[i].hasBuyImbalance = true;
         m_levels[i].imbalanceRatio = askAbove / bidCurrent;
      }
      
      // Sell Imbalance
      if(askAbove > 0 && bidCurrent / askAbove >= m_imbalanceRatio) {
         m_levels[i].hasSellImbalance = true;
         m_levels[i].imbalanceRatio = bidCurrent / askAbove;
      }
   }
}

//+------------------------------------------------------------------+
//| Detecta Stacked Imbalances (3+ consecutivos)                      |
//+------------------------------------------------------------------+
void CFootprintAnalyzer::DetectStackedImbalances() {
   m_stackedCount = 0;
   
   if(m_levelCount < m_minStackedLevels) return;
   
   int consecutiveBuy = 0;
   int consecutiveSell = 0;
   int buyStartIdx = -1;
   int sellStartIdx = -1;
   double buyRatioSum = 0;
   double sellRatioSum = 0;
   
   for(int i = 0; i < m_levelCount; i++) {
      // Buy imbalances
      if(m_levels[i].hasBuyImbalance) {
         if(consecutiveBuy == 0) buyStartIdx = i;
         consecutiveBuy++;
         buyRatioSum += m_levels[i].imbalanceRatio;
      }
      else {
         if(consecutiveBuy >= m_minStackedLevels && m_stackedCount < 20) {
            m_stackedImbalances[m_stackedCount].startPrice = m_levels[buyStartIdx].price;
            m_stackedImbalances[m_stackedCount].endPrice = m_levels[i - 1].price;
            m_stackedImbalances[m_stackedCount].levelCount = consecutiveBuy;
            m_stackedImbalances[m_stackedCount].type = IMBALANCE_BUY;
            m_stackedImbalances[m_stackedCount].avgRatio = buyRatioSum / consecutiveBuy;
            m_stackedImbalances[m_stackedCount].isActive = true;
            m_stackedImbalances[m_stackedCount].detectionTime = TimeCurrent();
            m_stackedCount++;
         }
         consecutiveBuy = 0;
         buyRatioSum = 0;
      }
      
      // Sell imbalances
      if(m_levels[i].hasSellImbalance) {
         if(consecutiveSell == 0) sellStartIdx = i;
         consecutiveSell++;
         sellRatioSum += m_levels[i].imbalanceRatio;
      }
      else {
         if(consecutiveSell >= m_minStackedLevels && m_stackedCount < 20) {
            m_stackedImbalances[m_stackedCount].startPrice = m_levels[sellStartIdx].price;
            m_stackedImbalances[m_stackedCount].endPrice = m_levels[i - 1].price;
            m_stackedImbalances[m_stackedCount].levelCount = consecutiveSell;
            m_stackedImbalances[m_stackedCount].type = IMBALANCE_SELL;
            m_stackedImbalances[m_stackedCount].avgRatio = sellRatioSum / consecutiveSell;
            m_stackedImbalances[m_stackedCount].isActive = true;
            m_stackedImbalances[m_stackedCount].detectionTime = TimeCurrent();
            m_stackedCount++;
         }
         consecutiveSell = 0;
         sellRatioSum = 0;
      }
   }
   
   // Check final sequences
   if(consecutiveBuy >= m_minStackedLevels && m_stackedCount < 20) {
      m_stackedImbalances[m_stackedCount].startPrice = m_levels[buyStartIdx].price;
      m_stackedImbalances[m_stackedCount].endPrice = m_levels[m_levelCount - 1].price;
      m_stackedImbalances[m_stackedCount].levelCount = consecutiveBuy;
      m_stackedImbalances[m_stackedCount].type = IMBALANCE_BUY;
      m_stackedImbalances[m_stackedCount].avgRatio = buyRatioSum / consecutiveBuy;
      m_stackedImbalances[m_stackedCount].isActive = true;
      m_stackedImbalances[m_stackedCount].detectionTime = TimeCurrent();
      m_stackedCount++;
   }
   
   if(consecutiveSell >= m_minStackedLevels && m_stackedCount < 20) {
      m_stackedImbalances[m_stackedCount].startPrice = m_levels[sellStartIdx].price;
      m_stackedImbalances[m_stackedCount].endPrice = m_levels[m_levelCount - 1].price;
      m_stackedImbalances[m_stackedCount].levelCount = consecutiveSell;
      m_stackedImbalances[m_stackedCount].type = IMBALANCE_SELL;
      m_stackedImbalances[m_stackedCount].avgRatio = sellRatioSum / consecutiveSell;
      m_stackedImbalances[m_stackedCount].isActive = true;
      m_stackedImbalances[m_stackedCount].detectionTime = TimeCurrent();
      m_stackedCount++;
   }
}

//+------------------------------------------------------------------+
//| Detecta Zonas de Absorcao                                         |
//+------------------------------------------------------------------+
void CFootprintAnalyzer::DetectAbsorptionZones() {
   m_absorptionCount = 0;
   
   if(m_levelCount == 0) return;
   
   // Calcula volume medio
   long totalVol = 0;
   for(int i = 0; i < m_levelCount; i++) {
      totalVol += m_levels[i].bidVolume + m_levels[i].askVolume;
   }
   double avgVol = (double)totalVol / m_levelCount;
   
   // Procura niveis com alto volume e delta baixo
   for(int i = 0; i < m_levelCount && m_absorptionCount < 20; i++) {
      long levelVol = m_levels[i].bidVolume + m_levels[i].askVolume;
      long levelDelta = m_levels[i].delta;
      
      // Volume significativo?
      if(levelVol < avgVol * m_volumeMultiplier) continue;
      
      // Delta baixo?
      double deltaPct = (levelVol > 0) ? MathAbs((double)levelDelta / levelVol * 100) : 100;
      
      if(deltaPct < m_absorptionThreshold) {
         m_absorptionZones[m_absorptionCount].price = m_levels[i].price;
         m_absorptionZones[m_absorptionCount].totalVolume = levelVol;
         m_absorptionZones[m_absorptionCount].delta = levelDelta;
         m_absorptionZones[m_absorptionCount].deltaPercent = deltaPct;
         m_absorptionZones[m_absorptionCount].detectionTime = TimeCurrent();
         
         // Determina tipo
         // Se preco estava caindo e vendas foram absorvidas = buy absorption
         // Se preco estava subindo e compras foram absorvidas = sell absorption
         double currentPrice = SymbolInfoDouble(m_symbol, SYMBOL_BID);
         if(currentPrice > m_levels[i].price) {
            m_absorptionZones[m_absorptionCount].type = ABSORPTION_BUY;
         }
         else {
            m_absorptionZones[m_absorptionCount].type = ABSORPTION_SELL;
         }
         
         m_absorptionCount++;
      }
   }
}

//+------------------------------------------------------------------+
//| Detecta Unfinished Auction                                        |
//+------------------------------------------------------------------+
ENUM_AUCTION_TYPE CFootprintAnalyzer::DetectUnfinishedAuction() {
   if(m_levelCount == 0) return AUCTION_NONE;
   
   double barOpen = iOpen(m_symbol, m_timeframe, 0);
   double barHigh = iHigh(m_symbol, m_timeframe, 0);
   double barLow = iLow(m_symbol, m_timeframe, 0);
   double barClose = iClose(m_symbol, m_timeframe, 0);
   
   long barDelta = GetBarDelta();
   
   // Close no high com delta positivo = bullish unfinished
   if(MathAbs(barClose - barHigh) < m_clusterSize && barDelta > 0) {
      // Verifica se tem buy imbalance no topo
      for(int i = m_levelCount - 1; i >= m_levelCount - 3 && i >= 0; i--) {
         if(m_levels[i].hasBuyImbalance)
            return AUCTION_UNFINISHED_UP;
      }
   }
   
   // Close no low com delta negativo = bearish unfinished
   if(MathAbs(barClose - barLow) < m_clusterSize && barDelta < 0) {
      // Verifica se tem sell imbalance no fundo
      for(int i = 0; i < 3 && i < m_levelCount; i++) {
         if(m_levels[i].hasSellImbalance)
            return AUCTION_UNFINISHED_DOWN;
      }
   }
   
   return AUCTION_NONE;
}

//+------------------------------------------------------------------+
//| Detecta Delta Divergence                                          |
//+------------------------------------------------------------------+
bool CFootprintAnalyzer::DetectDeltaDivergence(bool &bullish, bool &bearish) {
   bullish = false;
   bearish = false;
   
   if(m_historyCount < 3) return false;
   
   // Ultimos 3 pontos
   int last = m_historyCount - 1;
   
   // Bullish divergence: Price lower lows, Delta higher lows
   if(m_priceHistory[last] < m_priceHistory[last-1] && 
      m_priceHistory[last-1] < m_priceHistory[last-2]) {
      if(m_deltaHistory[last] > m_deltaHistory[last-1] &&
         m_deltaHistory[last-1] > m_deltaHistory[last-2]) {
         bullish = true;
      }
   }
   
   // Bearish divergence: Price higher highs, Delta lower highs
   if(m_priceHistory[last] > m_priceHistory[last-1] && 
      m_priceHistory[last-1] > m_priceHistory[last-2]) {
      if(m_deltaHistory[last] < m_deltaHistory[last-1] &&
         m_deltaHistory[last-1] < m_deltaHistory[last-2]) {
         bearish = true;
      }
   }
   
   return bullish || bearish;
}

//+------------------------------------------------------------------+
//| Detecta POC Defense                                               |
//+------------------------------------------------------------------+
bool CFootprintAnalyzer::DetectPOCDefense() {
   SValueArea va;
   CalculateValueArea(va);
   
   if(va.pocVolume == 0) return false;
   
   double currentPrice = SymbolInfoDouble(m_symbol, SYMBOL_BID);
   
   // Preco proximo do POC com alto volume
   if(MathAbs(currentPrice - va.poc) < m_clusterSize * 2) {
      if(va.pocVolume > va.totalVolume * 0.15) {
         return true;
      }
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Calcula Value Area                                                |
//+------------------------------------------------------------------+
void CFootprintAnalyzer::CalculateValueArea(SValueArea &va) {
   va.poc = 0;
   va.vahigh = 0;
   va.valow = 0;
   va.pocVolume = 0;
   va.totalVolume = 0;
   
   if(m_levelCount == 0) return;
   
   // Encontra POC
   int pocIdx = 0;
   long maxVol = 0;
   
   for(int i = 0; i < m_levelCount; i++) {
      long vol = m_levels[i].bidVolume + m_levels[i].askVolume;
      va.totalVolume += vol;
      if(vol > maxVol) {
         maxVol = vol;
         pocIdx = i;
      }
   }
   
   va.poc = m_levels[pocIdx].price;
   va.pocVolume = maxVol;
   
   // Calcula Value Area (70%)
   long targetVol = (long)(va.totalVolume * 0.70);
   long currentVol = maxVol;
   
   va.vahigh = va.poc;
   va.valow = va.poc;
   
   int upperIdx = pocIdx;
   int lowerIdx = pocIdx;
   
   while(currentVol < targetVol && (upperIdx < m_levelCount - 1 || lowerIdx > 0)) {
      long upperVol = 0, lowerVol = 0;
      
      if(upperIdx < m_levelCount - 1)
         upperVol = m_levels[upperIdx + 1].bidVolume + m_levels[upperIdx + 1].askVolume;
      if(lowerIdx > 0)
         lowerVol = m_levels[lowerIdx - 1].bidVolume + m_levels[lowerIdx - 1].askVolume;
      
      if(upperVol >= lowerVol && upperIdx < m_levelCount - 1) {
         upperIdx++;
         currentVol += upperVol;
         va.vahigh = m_levels[upperIdx].price;
      }
      else if(lowerIdx > 0) {
         lowerIdx--;
         currentVol += lowerVol;
         va.valow = m_levels[lowerIdx].price;
      }
      else break;
   }
}

//+------------------------------------------------------------------+
//| Obtem Delta da barra                                              |
//+------------------------------------------------------------------+
long CFootprintAnalyzer::GetBarDelta() {
   long delta = 0;
   for(int i = 0; i < m_levelCount; i++) {
      delta += m_levels[i].delta;
   }
   return delta;
}

//+------------------------------------------------------------------+
//| Obtem Delta cumulativo                                            |
//+------------------------------------------------------------------+
long CFootprintAnalyzer::GetCumulativeDelta() {
   return m_cumulativeDelta;
}

//+------------------------------------------------------------------+
//| Obtem POC                                                         |
//+------------------------------------------------------------------+
double CFootprintAnalyzer::GetPOC() {
   SValueArea va;
   CalculateValueArea(va);
   return va.poc;
}

//+------------------------------------------------------------------+
//| Obtem Value Area                                                  |
//+------------------------------------------------------------------+
SValueArea CFootprintAnalyzer::GetValueArea() {
   SValueArea va;
   CalculateValueArea(va);
   return va;
}

//+------------------------------------------------------------------+
//| Verifica se tem Stacked Buy Imbalance                             |
//+------------------------------------------------------------------+
bool CFootprintAnalyzer::HasStackedBuyImbalance() {
   for(int i = 0; i < m_stackedCount; i++) {
      if(m_stackedImbalances[i].type == IMBALANCE_BUY)
         return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Verifica se tem Stacked Sell Imbalance                            |
//+------------------------------------------------------------------+
bool CFootprintAnalyzer::HasStackedSellImbalance() {
   for(int i = 0; i < m_stackedCount; i++) {
      if(m_stackedImbalances[i].type == IMBALANCE_SELL)
         return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Verifica se tem Buy Absorption                                    |
//+------------------------------------------------------------------+
bool CFootprintAnalyzer::HasBuyAbsorption() {
   for(int i = 0; i < m_absorptionCount; i++) {
      if(m_absorptionZones[i].type == ABSORPTION_BUY)
         return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Verifica se tem Sell Absorption                                   |
//+------------------------------------------------------------------+
bool CFootprintAnalyzer::HasSellAbsorption() {
   for(int i = 0; i < m_absorptionCount; i++) {
      if(m_absorptionZones[i].type == ABSORPTION_SELL)
         return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Obtem Stacked Imbalance por indice                                |
//+------------------------------------------------------------------+
SStackedImbalance CFootprintAnalyzer::GetStackedImbalance(int index) {
   SStackedImbalance empty;
   empty.Reset();
   
   if(index < 0 || index >= m_stackedCount)
      return empty;
   
   return m_stackedImbalances[index];
}

//+------------------------------------------------------------------+
//| Obtem Absorption Zone por indice                                  |
//+------------------------------------------------------------------+
SAbsorptionZone CFootprintAnalyzer::GetAbsorptionZone(int index) {
   SAbsorptionZone empty;
   empty.Reset();
   
   if(index < 0 || index >= m_absorptionCount)
      return empty;
   
   return m_absorptionZones[index];
}

//+------------------------------------------------------------------+
//| Verifica se e bullish                                             |
//+------------------------------------------------------------------+
bool CFootprintAnalyzer::IsBullish() {
   SFootprintSignal sig = GetSignal();
   return sig.signal == FP_SIGNAL_STRONG_BUY || 
          sig.signal == FP_SIGNAL_BUY || 
          sig.signal == FP_SIGNAL_WEAK_BUY;
}

//+------------------------------------------------------------------+
//| Verifica se e bearish                                             |
//+------------------------------------------------------------------+
bool CFootprintAnalyzer::IsBearish() {
   SFootprintSignal sig = GetSignal();
   return sig.signal == FP_SIGNAL_STRONG_SELL || 
          sig.signal == FP_SIGNAL_SELL || 
          sig.signal == FP_SIGNAL_WEAK_SELL;
}

//+------------------------------------------------------------------+
//| Calcula score de confluencia (0-100)                              |
//+------------------------------------------------------------------+
int CFootprintAnalyzer::GetConfluenceScore() {
   SFootprintSignal sig = GetSignal();
   return sig.strength;
}

//+------------------------------------------------------------------+
//| Obtem sinal de trading completo                                   |
//+------------------------------------------------------------------+
SFootprintSignal CFootprintAnalyzer::GetSignal() {
   // Retorna cache se valido
   if(m_cacheValid && m_cachedBarTime == m_lastBarTime)
      return m_cachedSignal;
   
   m_cachedSignal.Reset();
   
   // Delta
   m_cachedSignal.barDelta = GetBarDelta();
   m_cachedSignal.cumulativeDelta = m_cumulativeDelta;
   
   long totalVol = 0;
   for(int i = 0; i < m_levelCount; i++) {
      totalVol += m_levels[i].bidVolume + m_levels[i].askVolume;
   }
   m_cachedSignal.deltaPercent = (totalVol > 0) ? 
      (double)m_cachedSignal.barDelta / totalVol * 100 : 0;
   
   // Value Area
   CalculateValueArea(m_cachedSignal.valueArea);
   
   // Componentes
   m_cachedSignal.hasStackedBuyImbalance = HasStackedBuyImbalance();
   m_cachedSignal.hasStackedSellImbalance = HasStackedSellImbalance();
   m_cachedSignal.hasBuyAbsorption = HasBuyAbsorption();
   m_cachedSignal.hasSellAbsorption = HasSellAbsorption();
   
   ENUM_AUCTION_TYPE auction = DetectUnfinishedAuction();
   m_cachedSignal.hasUnfinishedAuctionUp = (auction == AUCTION_UNFINISHED_UP);
   m_cachedSignal.hasUnfinishedAuctionDown = (auction == AUCTION_UNFINISHED_DOWN);
   
   bool bullDiv, bearDiv;
   DetectDeltaDivergence(bullDiv, bearDiv);
   m_cachedSignal.hasBullishDeltaDivergence = bullDiv;
   m_cachedSignal.hasBearishDeltaDivergence = bearDiv;
   
   m_cachedSignal.hasPOCDefense = DetectPOCDefense();
   
   // Calcula score
   int buyScore = 0;
   int sellScore = 0;
   
   // Stacked imbalances (peso alto)
   if(m_cachedSignal.hasStackedBuyImbalance) buyScore += 25;
   if(m_cachedSignal.hasStackedSellImbalance) sellScore += 25;
   
   // Absorption (peso medio-alto)
   if(m_cachedSignal.hasBuyAbsorption) buyScore += 20;
   if(m_cachedSignal.hasSellAbsorption) sellScore += 20;
   
   // Unfinished auction (peso medio)
   if(m_cachedSignal.hasUnfinishedAuctionUp) buyScore += 15;
   if(m_cachedSignal.hasUnfinishedAuctionDown) sellScore += 15;
   
   // Delta divergence (peso medio)
   if(m_cachedSignal.hasBullishDeltaDivergence) buyScore += 15;
   if(m_cachedSignal.hasBearishDeltaDivergence) sellScore += 15;
   
   // Delta percent (peso baixo)
   if(m_cachedSignal.deltaPercent > 30) buyScore += 10;
   if(m_cachedSignal.deltaPercent < -30) sellScore += 10;
   
   // POC defense (bonus)
   if(m_cachedSignal.hasPOCDefense) {
      if(m_cachedSignal.barDelta > 0) buyScore += 10;
      else sellScore += 10;
   }
   
   // Determina sinal
   int netScore = buyScore - sellScore;
   m_cachedSignal.strength = MathAbs(netScore);
   
   if(netScore >= 40)
      m_cachedSignal.signal = FP_SIGNAL_STRONG_BUY;
   else if(netScore >= 25)
      m_cachedSignal.signal = FP_SIGNAL_BUY;
   else if(netScore >= 10)
      m_cachedSignal.signal = FP_SIGNAL_WEAK_BUY;
   else if(netScore <= -40)
      m_cachedSignal.signal = FP_SIGNAL_STRONG_SELL;
   else if(netScore <= -25)
      m_cachedSignal.signal = FP_SIGNAL_SELL;
   else if(netScore <= -10)
      m_cachedSignal.signal = FP_SIGNAL_WEAK_SELL;
   else
      m_cachedSignal.signal = FP_SIGNAL_NEUTRAL;
   
   m_cacheValid = true;
   m_cachedBarTime = m_lastBarTime;
   
   return m_cachedSignal;
}

//+------------------------------------------------------------------+
//| Print diagnostics                                                 |
//+------------------------------------------------------------------+
void CFootprintAnalyzer::PrintDiagnostics() {
   SFootprintSignal sig = GetSignal();
   
   Print("=== FOOTPRINT ANALYZER v3.0 DIAGNOSTICS ===");
   Print("Symbol: ", m_symbol, " | TF: ", EnumToString(m_timeframe));
   Print("Cluster Size: ", m_clusterSize, " | Levels: ", m_levelCount);
   Print("");
   Print("--- DELTA ---");
   Print("Bar Delta: ", sig.barDelta);
   Print("Cumulative Delta: ", sig.cumulativeDelta);
   Print("Delta %: ", DoubleToString(sig.deltaPercent, 1), "%");
   Print("");
   Print("--- VALUE AREA ---");
   Print("POC: ", sig.valueArea.poc);
   Print("VAH: ", sig.valueArea.vahigh);
   Print("VAL: ", sig.valueArea.valow);
   Print("");
   Print("--- PATTERNS ---");
   Print("Stacked Buy Imbalance: ", sig.hasStackedBuyImbalance ? "YES" : "NO");
   Print("Stacked Sell Imbalance: ", sig.hasStackedSellImbalance ? "YES" : "NO");
   Print("Buy Absorption: ", sig.hasBuyAbsorption ? "YES" : "NO");
   Print("Sell Absorption: ", sig.hasSellAbsorption ? "YES" : "NO");
   Print("Unfinished Auction Up: ", sig.hasUnfinishedAuctionUp ? "YES" : "NO");
   Print("Unfinished Auction Down: ", sig.hasUnfinishedAuctionDown ? "YES" : "NO");
   Print("Bullish Delta Divergence: ", sig.hasBullishDeltaDivergence ? "YES" : "NO");
   Print("Bearish Delta Divergence: ", sig.hasBearishDeltaDivergence ? "YES" : "NO");
   Print("POC Defense: ", sig.hasPOCDefense ? "YES" : "NO");
   Print("");
   Print("--- SIGNAL ---");
   Print("Signal: ", EnumToString(sig.signal));
   Print("Strength: ", sig.strength);
   Print("============================================");
}

//+------------------------------------------------------------------+
//| Print levels                                                      |
//+------------------------------------------------------------------+
void CFootprintAnalyzer::PrintLevels() {
   Print("=== FOOTPRINT LEVELS ===");
   Print("Total: ", m_levelCount);
   
   for(int i = m_levelCount - 1; i >= 0; i--) {
      string imb = "";
      if(m_levels[i].hasBuyImbalance) imb = " [BUY IMB " + DoubleToString(m_levels[i].imbalanceRatio, 1) + "x]";
      if(m_levels[i].hasSellImbalance) imb = " [SELL IMB " + DoubleToString(m_levels[i].imbalanceRatio, 1) + "x]";
      
      Print(StringFormat("%.2f | Bid: %d | Ask: %d | Delta: %+d%s",
            m_levels[i].price,
            m_levels[i].bidVolume,
            m_levels[i].askVolume,
            m_levels[i].delta,
            imb));
   }
   Print("========================");
}

//+------------------------------------------------------------------+
//| Print stacked imbalances                                          |
//+------------------------------------------------------------------+
void CFootprintAnalyzer::PrintStackedImbalances() {
   Print("=== STACKED IMBALANCES ===");
   Print("Count: ", m_stackedCount);
   
   for(int i = 0; i < m_stackedCount; i++) {
      Print(StringFormat("%d. %s: %.2f - %.2f (%d levels, avg %.1fx)",
            i + 1,
            m_stackedImbalances[i].type == IMBALANCE_BUY ? "BUY" : "SELL",
            m_stackedImbalances[i].startPrice,
            m_stackedImbalances[i].endPrice,
            m_stackedImbalances[i].levelCount,
            m_stackedImbalances[i].avgRatio));
   }
   Print("==========================");
}
//+------------------------------------------------------------------+
