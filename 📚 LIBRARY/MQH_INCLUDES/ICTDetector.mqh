//+------------------------------------------------------------------+
//|                                                    ICTDetector.mqh |
//|                        Copyright 2025, MetaQuotes Software Corp. |
//|                                       https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include <Trade\Trade.mqh>

//+------------------------------------------------------------------+
//| Classe para detecção de padrões ICT                             |
//+------------------------------------------------------------------+
class CICTDetector
{
private:
   // Dados de detecção ICT
   bool m_order_block_detected;
   bool m_fvg_detected;
   bool m_swing_high_detected;
   bool m_swing_low_detected;
   bool m_liquidity_zone_detected;
   
   // Parâmetros
   double m_ob_threshold;
   double m_fvg_threshold;
   double m_swings_threshold;
   int    m_lookback_period;

public:
   // Construtor
   CICTDetector()
   {
      m_order_block_detected = false;
      m_fvg_detected = false;
      m_swing_high_detected = false;
      m_swing_low_detected = false;
      m_liquidity_zone_detected = false;
      
      m_ob_threshold = 0.7;
      m_fvg_threshold = 0.8;
      m_swings_threshold = 0.6;
      m_lookback_period = 20;
   }

   // Destrutor
   ~CICTDetector()
   {
   }

   // Inicialização
   bool Init()
   {
      return true;
   }

   // Configuração de parâmetros
   void SetParameters(double ob_threshold, double fvg_threshold,
                     double swings_threshold, int lookback_period)
   {
      m_ob_threshold = ob_threshold;
      m_fvg_threshold = fvg_threshold;
      m_swings_threshold = swings_threshold;
      m_lookback_period = lookback_period;
   }

   // Detectar Order Block
   bool DetectOrderBlock(double high[], double low[], int bars)
   {
      m_order_block_detected = false;
      
      // Lógica simplificada de detecção de Order Block
      for(int i = bars - m_lookback_period; i >= 0; i--)
      {
         double range = high[i] - low[i];
         double body = MathAbs((high[i] + low[i]) / 2 - (high[i+1] + low[i+1]) / 2);
         
         if(body / range >= m_ob_threshold)
         {
            m_order_block_detected = true;
            break;
         }
      }
      
      return m_order_block_detected;
   }

   // Detectar Fair Value Gap (FVG)
   bool DetectFVG(double high[], double low[], int bars)
   {
      m_fvg_detected = false;
      
      // Lógica simplificada de detecção de FVG
      for(int i = bars - m_lookback_period; i >= 2; i--)
      {
         if(low[i] > high[i-2])
         {
            double fvg_size = low[i] - high[i-2];
            double range = high[i-1] - low[i-1];
            
            if(fvg_size / range >= m_fvg_threshold)
            {
               m_fvg_detected = true;
               break;
            }
         }
      }
      
      return m_fvg_detected;
   }

   // Detectar Swing High
   bool DetectSwingHigh(double high[], int bars)
   {
      m_swing_high_detected = false;
      
      // Lógica simplificada de detecção de Swing High
      for(int i = bars - m_lookback_period; i >= 2; i--)
      {
         if(high[i] > high[i-1] && high[i] > high[i+1])
         {
            double left_range = high[i] - high[i-1];
            double right_range = high[i] - high[i+1];
            double total_range = high[i] - MathMin(low[i-1], low[i+1]);
            
            if((left_range + right_range) / total_range >= m_swings_threshold)
            {
               m_swing_high_detected = true;
               break;
            }
         }
      }
      
      return m_swing_high_detected;
   }

   // Detectar Swing Low
   bool DetectSwingLow(double low[], int bars)
   {
      m_swing_low_detected = false;
      
      // Lógica simplificada de detecção de Swing Low
      for(int i = bars - m_lookback_period; i >= 2; i--)
      {
         if(low[i] < low[i-1] && low[i] < low[i+1])
         {
            double left_range = low[i-1] - low[i];
            double right_range = low[i+1] - low[i];
            double total_range = MathMax(high[i-1], high[i+1]) - low[i];
            
            if((left_range + right_range) / total_range >= m_swings_threshold)
            {
               m_swing_low_detected = true;
               break;
            }
         }
      }
      
      return m_swing_low_detected;
   }

   // Detectar Zona de Liquidez
   bool DetectLiquidityZone(double high[], double low[], int bars)
   {
      m_liquidity_zone_detected = false;
      
      // Lógica simplificada de detecção de zona de liquidez
      for(int i = bars - m_lookback_period; i >= 0; i--)
      {
         double range = high[i] - low[i];
         double volume_factor = 1.0; // Simplificado - na realidade usaria dados de volume
         
         if(range * volume_factor >= m_fvg_threshold)
         {
            m_liquidity_zone_detected = true;
            break;
         }
      }
      
      return m_liquidity_zone_detected;
   }

   // Obter status de detecção
   bool GetOrderBlockDetected() { return m_order_block_detected; }
   bool GetFVGDetected() { return m_fvg_detected; }
   bool GetSwingHighDetected() { return m_swing_high_detected; }
   bool GetSwingLowDetected() { return m_swing_low_detected; }
   bool GetLiquidityZoneDetected() { return m_liquidity_zone_detected; }

   // Resetar detecções
   void Reset()
   {
      m_order_block_detected = false;
      m_fvg_detected = false;
      m_swing_high_detected = false;
      m_swing_low_detected = false;
      m_liquidity_zone_detected = false;
   }
};