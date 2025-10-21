//+------------------------------------------------------------------+
//|                                                  SignalConfluence.mqh |
//|                        Copyright 2025, MetaQuotes Software Corp. |
//|                                       https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include <Trade\Trade.mqh>

//+------------------------------------------------------------------+
//| Classe para sistema de confluência de sinais                     |
//+------------------------------------------------------------------+
class CSignalConfluence;
{
private:
   // Dados de confluência
   double m_confluence_score;
   int    m_signal_count;
   int    m_confirmation_count;
   
   // Parâmetros
   double m_min_confluence_threshold;
   double m_max_confluence_threshold;

public:
   // Construtor
   CSignalConfluence()
   {
      m_confluence_score = 0.0;
      m_signal_count = 0;
      m_confirmation_count = 0;
      m_min_confluence_threshold = 0.6;
      m_max_confluence_threshold = 1.0;
   }

   // Destrutor
   ~CSignalConfluence()
   {
   }

   // Inicialização
   bool Init()
   {
      return true;
   }

   // Configuração de parâmetros
   void SetThresholds(double min_threshold, double max_threshold)
   {
      m_min_confluence_threshold = min_threshold;
      m_max_confluence_threshold = max_threshold;
   }

   // Adicionar sinal
   void AddSignal(double signal_strength, bool is_confirmed)
   {
      m_confluence_score += signal_strength;
      m_signal_count++;
      
      if(is_confirmed)
         m_confirmation_count++;
   }

   // Calcular pontuação de confluência
   double CalculateConfluenceScore()
   {
      if(m_signal_count == 0)
         return 0.0;
      
      // Normalizar a pontuação
      double normalized_score = m_confluence_score / m_signal_count;
      
      // Ajustar com base nas confirmações
      if(m_confirmation_count > 0)
      {
         double confirmation_factor = (double)m_confirmation_count / m_signal_count;
         normalized_score *= (1.0 + confirmation_factor);
      }
      
      // Limitar entre 0 e 1
      return MathMin(MathMax(normalized_score, 0.0), 1.0);
   }

   // Verificar se há confluência suficiente
   bool HasSufficientConfluence()
   {
      double score = CalculateConfluenceScore();
      return (score >= m_min_confluence_threshold && score <= m_max_confluence_threshold);
   }

   // Obter pontuação de confluência
   double GetConfluenceScore()
   {
      return CalculateConfluenceScore();
   }

   // Resetar dados
   void Reset()
   {
      m_confluence_score = 0.0;
      m_signal_count = 0;
      m_confirmation_count = 0;
   }
};