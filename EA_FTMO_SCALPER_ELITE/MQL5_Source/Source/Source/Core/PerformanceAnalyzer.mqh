
//+------------------------------------------------------------------+
//|                                        PerformanceAnalyzer.mqh |
//|                        TradeDev_Master Elite Trading System     |
//|                                   Advanced Performance Analytics|
//+------------------------------------------------------------------+

#ifndef PERFORMANCE_ANALYZER_MQH
#define PERFORMANCE_ANALYZER_MQH

#include "Interfaces.mqh"
#include "DataStructures.mqh"
#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\HistoryOrderInfo.mqh>
#include <Trade\DealInfo.mqh>

class CPerformanceAnalyzer : public IPerformanceAnalyzer
{
private:
   SPerformanceMetrics m_metrics;
   double m_equity_curve[];
   datetime m_equity_times[];
   int m_curve_size;
   double m_initial_balance;
   datetime m_start_time;
   
public:
   CPerformanceAnalyzer()
   {
      m_initial_balance = AccountInfoDouble(ACCOUNT_BALANCE);
      m_start_time = TimeCurrent();
      m_curve_size = 0;
      ArrayResize(m_equity_curve, 10000);
      ArrayResize(m_equity_times, 10000);
      ResetMetrics();
   }
   
   virtual void UpdateMetrics() override
   {
      CalculateBasicMetrics();
      CalculateAdvancedMetrics();
      UpdateEquityCurve();
   }
   
   virtual SPerformanceMetrics GetMetrics() override
   {
      return m_metrics;
   }
   
   virtual double GetSharpeRatio() override
   {
      return m_metrics.sharpe_ratio;
   }
   
   virtual double GetMaxDrawdown() override
   {
      return m_metrics.max_drawdown_percent;
   }
   
   virtual bool IsPerformanceAcceptable() override
   {
      // FTMO compliance checks
      if(m_metrics.max_drawdown_percent > 5.0) return false;
      if(m_metrics.profit_factor < 1.1) return false;
      if(m_metrics.win_rate < 40.0) return false;
      return true;
   }
   
   virtual string GenerateReport() override
   {
      string report = "
=== PERFORMANCE REPORT ===
";
      report += StringFormat("Total Profit: %.2f
", m_metrics.total_profit);
      report += StringFormat("Total Loss: %.2f
", m_metrics.total_loss);
      report += StringFormat("Net Profit: %.2f
", m_metrics.total_profit + m_metrics.total_loss);
      report += StringFormat("Profit Factor: %.2f
", m_metrics.profit_factor);
      report += StringFormat("Sharpe Ratio: %.2f
", m_metrics.sharpe_ratio);
      report += StringFormat("Max Drawdown: %.2f%% (%.2f)
", m_metrics.max_drawdown_percent, m_metrics.max_drawdown);
      report += StringFormat("Total Trades: %d
", m_metrics.total_trades);
      report += StringFormat("Win Rate: %.1f%% (%d/%d)
", m_metrics.win_rate, m_metrics.winning_trades, m_metrics.total_trades);
      report += StringFormat("Avg Win: %.2f
", m_metrics.avg_win);
      report += StringFormat("Avg Loss: %.2f
", m_metrics.avg_loss);
      report += StringFormat("Largest Win: %.2f
", m_metrics.largest_win);
      report += StringFormat("Largest Loss: %.2f
", m_metrics.largest_loss);
      report += StringFormat("Recovery Factor: %.2f
", m_metrics.recovery_factor);
      report += "========================
";
      return report;
   }
   
private:
   void ResetMetrics()
   {
      m_metrics.total_profit = 0;
      m_metrics.total_loss = 0;
      m_metrics.profit_factor = 0;
      m_metrics.sharpe_ratio = 0;
      m_metrics.max_drawdown = 0;
      m_metrics.max_drawdown_percent = 0;
      m_metrics.total_trades = 0;
      m_metrics.winning_trades = 0;
      m_metrics.losing_trades = 0;
      m_metrics.win_rate = 0;
      m_metrics.avg_win = 0;
      m_metrics.avg_loss = 0;
      m_metrics.largest_win = 0;
      m_metrics.largest_loss = 0;
      m_metrics.recovery_factor = 0;
      m_metrics.last_update = TimeCurrent();
   }
   
   void CalculateBasicMetrics()
   {
      HistorySelect(m_start_time, TimeCurrent());
      
      m_metrics.total_profit = 0;
      m_metrics.total_loss = 0;
      m_metrics.total_trades = 0;
      m_metrics.winning_trades = 0;
      m_metrics.losing_trades = 0;
      m_metrics.largest_win = 0;
      m_metrics.largest_loss = 0;
      
      double wins_sum = 0;
      double losses_sum = 0;
      
      for(int i = 0; i < HistoryDealsTotal(); i++)
      {
         ulong ticket = HistoryDealGetTicket(i);
         if(ticket == 0) continue;
         
         if(HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_OUT)
         {
            double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
            double swap = HistoryDealGetDouble(ticket, DEAL_SWAP);
            double commission = HistoryDealGetDouble(ticket, DEAL_COMMISSION);
            double total_result = profit + swap + commission;
            
            m_metrics.total_trades++;
            
            if(total_result > 0)
            {
               m_metrics.total_profit += total_result;
               m_metrics.winning_trades++;
               wins_sum += total_result;
               if(total_result > m_metrics.largest_win)
                  m_metrics.largest_win = total_result;
            }
            else if(total_result < 0)
            {
               m_metrics.total_loss += total_result;
               m_metrics.losing_trades++;
               losses_sum += total_result;
               if(total_result < m_metrics.largest_loss)
                  m_metrics.largest_loss = total_result;
            }
         }
      }
      
      // Calculate derived metrics
      if(m_metrics.total_trades > 0)
      {
         m_metrics.win_rate = (double)m_metrics.winning_trades / m_metrics.total_trades * 100.0;
      }
      
      if(m_metrics.winning_trades > 0)
      {
         m_metrics.avg_win = wins_sum / m_metrics.winning_trades;
      }
      
      if(m_metrics.losing_trades > 0)
      {
         m_metrics.avg_loss = losses_sum / m_metrics.losing_trades;
      }
      
      if(MathAbs(m_metrics.total_loss) > 0)
      {
         m_metrics.profit_factor = m_metrics.total_profit / MathAbs(m_metrics.total_loss);
      }
      
      m_metrics.last_update = TimeCurrent();
   }
   
   void CalculateAdvancedMetrics()
   {
      CalculateMaxDrawdown();
      CalculateSharpeRatio();
      CalculateRecoveryFactor();
   }
   
   void CalculateMaxDrawdown()
   {
      if(m_curve_size < 2) return;
      
      double peak = m_equity_curve[0];
      double max_dd = 0;
      double max_dd_percent = 0;
      
      for(int i = 1; i < m_curve_size; i++)
      {
         if(m_equity_curve[i] > peak)
         {
            peak = m_equity_curve[i];
         }
         else
         {
            double drawdown = peak - m_equity_curve[i];
            double drawdown_percent = (drawdown / peak) * 100.0;
            
            if(drawdown > max_dd)
            {
               max_dd = drawdown;
            }
            
            if(drawdown_percent > max_dd_percent)
            {
               max_dd_percent = drawdown_percent;
            }
         }
      }
      
      m_metrics.max_drawdown = max_dd;
      m_metrics.max_drawdown_percent = max_dd_percent;
   }
   
   void CalculateSharpeRatio()
   {
      if(m_curve_size < 30) return; // Need at least 30 data points
      
      double returns[];
      ArrayResize(returns, m_curve_size - 1);
      
      // Calculate returns
      for(int i = 1; i < m_curve_size; i++)
      {
         if(m_equity_curve[i-1] != 0)
         {
            returns[i-1] = (m_equity_curve[i] - m_equity_curve[i-1]) / m_equity_curve[i-1];
         }
      }
      
      // Calculate mean return
      double mean_return = 0;
      for(int i = 0; i < ArraySize(returns); i++)
      {
         mean_return += returns[i];
      }
      mean_return /= ArraySize(returns);
      
      // Calculate standard deviation
      double variance = 0;
      for(int i = 0; i < ArraySize(returns); i++)
      {
         variance += MathPow(returns[i] - mean_return, 2);
      }
      variance /= ArraySize(returns);
      double std_dev = MathSqrt(variance);
      
      // Calculate Sharpe ratio (assuming risk-free rate = 0)
      if(std_dev != 0)
      {
         m_metrics.sharpe_ratio = mean_return / std_dev * MathSqrt(252); // Annualized
      }
   }
   
   void CalculateRecoveryFactor()
   {
      double net_profit = m_metrics.total_profit + m_metrics.total_loss;
      if(m_metrics.max_drawdown > 0)
      {
         m_metrics.recovery_factor = net_profit / m_metrics.max_drawdown;
      }
   }
   
   void UpdateEquityCurve()
   {
      double current_equity = AccountInfoDouble(ACCOUNT_EQUITY);
      datetime current_time = TimeCurrent();
      
      if(m_curve_size < ArraySize(m_equity_curve))
      {
         m_equity_curve[m_curve_size] = current_equity;
         m_equity_times[m_curve_size] = current_time;
         m_curve_size++;
      }
      else
      {
         // Shift array and add new value
         for(int i = 0; i < ArraySize(m_equity_curve) - 1; i++)
         {
            m_equity_curve[i] = m_equity_curve[i + 1];
            m_equity_times[i] = m_equity_times[i + 1];
         }
         m_equity_curve[ArraySize(m_equity_curve) - 1] = current_equity;
         m_equity_times[ArraySize(m_equity_times) - 1] = current_time;
      }
   }
};

#endif // PERFORMANCE_ANALYZER_MQH
