//+------------------------------------------------------------------+
//|                                                   DynamicLevels.mqh |
//|                        Copyright 2025, MetaQuotes Software Corp. |
//|                                       https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include <Trade\Trade.mqh>

//+------------------------------------------------------------------+
//| Classe para cálculo dinâmico de níveis SL/TP                     |
//+------------------------------------------------------------------+
class CDynamicLevels
{
private:
   // Dados de níveis dinâmicos
   double m_dynamic_sl;
   double m_dynamic_tp;
   double m_atr_multiplier;
   double m_volatility_factor;
   
   // Parâmetros
   double m_sl_atr_period;
   double m_tp_atr_period;
   double m_min_distance;
   double m_max_distance;

public:
   // Construtor
   CDynamicLevels()
   {
      m_dynamic_sl = 0.0;
      m_dynamic_tp = 0.0;
      m_atr_multiplier = 2.0;
      m_volatility_factor = 1.0;
      m_sl_atr_period = 14;
      m_tp_atr_period = 14;
      m_min_distance = 10.0;
      m_max_distance = 100.0;
   }

   // Destrutor
   ~CDynamicLevels()
   {
   }

   // Inicialização
   bool Init()
   {
      return true;
   }

   // Configuração de parâmetros
   void SetParameters(double atr_multiplier, double volatility_factor, 
                     double sl_atr_period, double tp_atr_period,
                     double min_distance, double max_distance)
   {
      m_atr_multiplier = atr_multiplier;
      m_volatility_factor = volatility_factor;
      m_sl_atr_period = sl_atr_period;
      m_tp_atr_period = tp_atr_period;
      m_min_distance = min_distance;
      m_max_distance = max_distance;
   }

   // Calcular SL dinâmico
   double CalculateDynamicSL(double price, double atr_value, ENUM_ORDER_TYPE order_type)
   {
      double sl_distance = atr_value * m_atr_multiplier * m_volatility_factor;
      
      // Aplicar limites de distância
      sl_distance = MathMax(sl_distance, m_min_distance);
      sl_distance = MathMin(sl_distance, m_max_distance);
      
      if(order_type == ORDER_TYPE_BUY)
      {
         m_dynamic_sl = price - sl_distance;
      }
      else if(order_type == ORDER_TYPE_SELL)
      {
         m_dynamic_sl = price + sl_distance;
      }
      
      return m_dynamic_sl;
   }

   // Calcular TP dinâmico
   double CalculateDynamicTP(double price, double atr_value, ENUM_ORDER_TYPE order_type)
   {
      double tp_distance = atr_value * m_atr_multiplier * m_volatility_factor * 1.5;
      
      // Aplicar limites de distância
      tp_distance = MathMax(tp_distance, m_min_distance);
      tp_distance = MathMin(tp_distance, m_max_distance);
      
      if(order_type == ORDER_TYPE_BUY)
      {
         m_dynamic_tp = price + tp_distance;
      }
      else if(order_type == ORDER_TYPE_SELL)
      {
         m_dynamic_tp = price - tp_distance;
      }
      
      return m_dynamic_tp;
   }

   // Obter SL dinâmico
   double GetDynamicSL()
   {
      return m_dynamic_sl;
   }

   // Obter TP dinâmico
   double GetDynamicTP()
   {
      return m_dynamic_tp;
   }

   // Resetar níveis
   void Reset()
   {
      m_dynamic_sl = 0.0;
      m_dynamic_tp = 0.0;
   }
};