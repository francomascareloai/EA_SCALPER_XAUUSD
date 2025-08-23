
//+------------------------------------------------------------------+
//|                                                   Logger.mqh |
//|                        TradeDev_Master Elite Trading System     |
//|                                   Advanced Logging System       |
//+------------------------------------------------------------------+

#ifndef LOGGER_MQH
#define LOGGER_MQH

#include "Interfaces.mqh"
#include "DataStructures.mqh"

class CLogger : public ILogger
{
private:
   ENUM_LOG_LEVEL m_log_level;
   bool m_enabled;
   bool m_file_logging;
   string m_log_file;
   string m_log_buffer[];
   int m_buffer_size;
   datetime m_last_flush;
   
public:
   CLogger() : m_log_level(LOG_INFO), m_enabled(true), m_file_logging(false), 
               m_buffer_size(0), m_last_flush(0)
   {
      m_log_file = "EA_FTMO_Scalper_" + TimeToString(TimeCurrent(), TIME_DATE) + ".log";
      ArrayResize(m_log_buffer, 1000);
   }
   
   ~CLogger()
   {
      FlushBuffer();
   }
   
   virtual void SetLevel(ENUM_LOG_LEVEL level) override
   {
      m_log_level = level;
   }
   
   void Enable(bool enable) { m_enabled = enable; }
   void EnableFileLogging(bool enable) { m_file_logging = enable; }
   void SetLogFile(string filename) { m_log_file = filename; }
   
   virtual void Debug(string message) override
   {
      if(m_enabled && m_log_level <= LOG_DEBUG)
         WriteLog("DEBUG", message);
   }
   
   virtual void Info(string message) override
   {
      if(m_enabled && m_log_level <= LOG_INFO)
         WriteLog("INFO", message);
   }
   
   virtual void Warning(string message) override
   {
      if(m_enabled && m_log_level <= LOG_WARNING)
         WriteLog("WARNING", message);
   }
   
   virtual void Error(string message) override
   {
      if(m_enabled && m_log_level <= LOG_ERROR)
         WriteLog("ERROR", message);
   }
   
   virtual void Critical(string message) override
   {
      if(m_enabled && m_log_level <= LOG_CRITICAL)
      {
         WriteLog("CRITICAL", message);
         FlushBuffer(); // Flush immediately for critical messages
      }
   }
   
   virtual bool SaveToFile(string filename) override
   {
      if(m_buffer_size == 0) return true;
      
      int file_handle = FileOpen(filename, FILE_WRITE|FILE_TXT);
      if(file_handle == INVALID_HANDLE)
         return false;
         
      for(int i = 0; i < m_buffer_size; i++)
      {
         FileWrite(file_handle, m_log_buffer[i]);
      }
      
      FileClose(file_handle);
      return true;
   }
   
private:
   void WriteLog(string level, string message)
   {
      string timestamp = TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS);
      string log_entry = StringFormat("[%s] [%s] %s", timestamp, level, message);
      
      // Print to terminal
      Print(log_entry);
      
      // Add to buffer
      if(m_buffer_size < ArraySize(m_log_buffer))
      {
         m_log_buffer[m_buffer_size] = log_entry;
         m_buffer_size++;
      }
      
      // Auto-flush every 5 minutes or when buffer is full
      if(TimeCurrent() - m_last_flush > 300 || m_buffer_size >= ArraySize(m_log_buffer))
      {
         FlushBuffer();
      }
   }
   
   void FlushBuffer()
   {
      if(m_file_logging && m_buffer_size > 0)
      {
         SaveToFile(m_log_file);
      }
      m_buffer_size = 0;
      m_last_flush = TimeCurrent();
   }
};

#endif // LOGGER_MQH
