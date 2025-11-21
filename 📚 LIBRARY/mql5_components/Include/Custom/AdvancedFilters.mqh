//+------------------------------------------------------------------+
//|                                                AdvancedFilters.mqh |
//|                        Copyright 2025, MetaQuotes Software Corp. |
//|                                       https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include <Trade\Trade.mqh>

//+------------------------------------------------------------------+
//| Classe para filtros avançados de trading                         |
//+------------------------------------------------------------------+
class CAdvancedFilters
{
private:
   // Dados de filtros
   bool m_time_filter_enabled;
   bool m_volatility_filter_enabled;
   bool m_trend_filter_enabled;
   bool m_news_filter_enabled;
   
   // Parâmetros
   double m_min_volatility;
   double m_max_volatility;
   int    m_min_hour;
   int    m_max_hour;
   double m_trend_strength;
   bool   m_news_event_active;

public:
   // Construtor
   CAdvancedFilters()
   {
      m_time_filter_enabled = true;
      m_volatility_filter_enabled = true;
      m_trend_filter_enabled = true;
      m_news_filter_enabled = true;
      
      m_min_volatility = 0.5;
      m_max_volatility = 5.0;
      m_min_hour = 8;
      m_max_hour = 22;
      m_trend_strength = 0.6;
      m_news_event_active = false;
   }

   // Destrutor
   ~CAdvancedFilters()
   {
   }

   // Inicialização
   bool Init()
   {
      return true;
   }

   // Configuração de parâmetros
   void SetFilterParameters(double min_volatility, double max_volatility,
                          int min_hour, int max_hour,
                          double trend_strength)
   {
      m_min_volatility = min_volatility;
      m_max_volatility = max_volatility;
      m_min_hour = min_hour;
      m_max_hour = max_hour;
      m_trend_strength = trend_strength;
   }

   // Habilitar/desabilitar filtros
   void SetTimeFilter(bool enabled)
   {
      m_time_filter_enabled = enabled;
   }

   void SetVolatilityFilter(bool enabled)
   {
      m_volatility_filter_enabled = enabled;
   }

   void SetTrendFilter(bool enabled)
   {
      m_trend_filter_enabled = enabled;
   }

   void SetNewsFilter(bool enabled)
   {
      m_news_filter_enabled = enabled;
   }

   // Verificar filtro de tempo
   bool CheckTimeFilter()
   {
      if(!m_time_filter_enabled)
         return true;
      
      datetime current_time = TimeCurrent();
      int current_hour = TimeHour(current_time);
      
      // Verificar se está dentro do horário permitido
      if(current_hour >= m_min_hour && current_hour <= m_max_hour)
         return true;
      
      return false;
   }

   // Verificar filtro de volatilidade
   bool CheckVolatilityFilter(double current_volatility)
   {
      if(!m_volatility_filter_enabled)
         return true;
      
      if(current_volatility >= m_min_volatility && current_volatility <= m_max_volatility)
         return true;
      
      return false;
   }

   // Verificar filtro de tendência
   bool CheckTrendFilter(double trend_strength)
   {
      if(!m_trend_filter_enabled)
         return true;
      
      if(trend_strength >= m_trend_strength)
         return true;
      
      return false;
   }

   // Verificar filtro de notícias
   bool CheckNewsFilter()
   {
      if(!m_news_filter_enabled)
         return true;
      
      return !m_news_event_active;
   }

   // Verificar todos os filtros
   bool CheckAllFilters(double volatility, double trend_strength)
   {
      if(!CheckTimeFilter())
         return false;
      
      if(!CheckVolatilityFilter(volatility))
         return false;
      
      if(!CheckTrendFilter(trend_strength))
         return false;
      
      if(!CheckNewsFilter())
         return false;
      
      return true;
   }

   // Notificar evento de notícia
   void NotifyNewsEvent(bool active)
   {
      m_news_event_active = active;
   }

   // Resetar filtros
   void Reset()
   {
      m_news_event_active = false;
   }
};