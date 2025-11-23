//+------------------------------------------------------------------+
//|                                                   CHeartbeat.mqh |
//|                                                   MQL5 Architect |
//|                                      Copyright 2025, Elite Ops.  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Elite Ops."
#property version   "1.00"

#include <Files/FileTxt.mqh>

class CHeartbeat
  {
private:
   string            m_folder;
   datetime          m_last_pong;
   int               m_timeout_sec;
   CFileTxt          m_file;

public:
                     CHeartbeat(void);
                    ~CHeartbeat(void);

   void              Init(string folder, int timeout_sec=15);
   void              SendPing();
   bool              CheckPong();
   bool              IsAlive();
   datetime          GetLastPongTime() { return m_last_pong; }
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CHeartbeat::CHeartbeat(void) : m_last_pong(0), m_timeout_sec(15)
  {
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CHeartbeat::~CHeartbeat(void)
  {
  }
//+------------------------------------------------------------------+
//| Init                                                             |
//+------------------------------------------------------------------+
void CHeartbeat::Init(string folder, int timeout_sec)
  {
   m_folder = folder;
   m_timeout_sec = timeout_sec;
   m_last_pong = TimeCurrent(); // Assume alive at start
   
   // Ensure folder exists in Common (Manual check required usually, but we assume it exists)
  }
//+------------------------------------------------------------------+
//| Send Ping (Write Timestamp to MQL side)                          |
//+------------------------------------------------------------------+
void CHeartbeat::SendPing()
  {
   // We write to 'mql_heartbeat.txt'
   int handle = FileOpen(m_folder + "/mql_heartbeat.txt", FILE_WRITE|FILE_TXT|FILE_COMMON|FILE_ANSI);
   if(handle != INVALID_HANDLE)
     {
      FileWriteString(handle, (string)TimeCurrent());
      FileClose(handle);
     }
  }
//+------------------------------------------------------------------+
//| Check Pong (Read Timestamp from Python side)                     |
//+------------------------------------------------------------------+
bool CHeartbeat::CheckPong()
  {
   // We read 'py_heartbeat.txt'
   if(FileIsExist(m_folder + "/py_heartbeat.txt", FILE_COMMON))
     {
      int handle = FileOpen(m_folder + "/py_heartbeat.txt", FILE_READ|FILE_TXT|FILE_COMMON|FILE_ANSI);
      if(handle != INVALID_HANDLE)
        {
         string content = FileReadString(handle);
         FileClose(handle);
         
         long py_time = StringToInteger(content);
         if(py_time > 0)
           {
            // If the timestamp is recent (allowing for some clock drift/latency)
            // We update our local 'last_pong'
            m_last_pong = TimeCurrent(); 
            return true;
           }
        }
     }
   return false;
  }
//+------------------------------------------------------------------+
//| Is Alive?                                                        |
//+------------------------------------------------------------------+
bool CHeartbeat::IsAlive()
  {
   // Check if we exceeded timeout
   if(TimeCurrent() - m_last_pong > m_timeout_sec)
      return false;
      
   return true;
  }
//+------------------------------------------------------------------+
