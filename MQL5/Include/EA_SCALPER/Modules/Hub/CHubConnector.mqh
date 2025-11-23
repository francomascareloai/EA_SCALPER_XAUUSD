//+------------------------------------------------------------------+
//|                                                CHubConnector.mqh |
//|                                                   MQL5 Architect |
//|                                      Copyright 2025, Elite Ops.  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Elite Ops."
#property version   "1.00"

#include <Files/FileTxt.mqh>
#include "../../Utils/CJson.mqh"

class CHubConnector
  {
private:
   string            m_folder;
   CJson             m_json_parser;

public:
                     CHubConnector(void);
                    ~CHubConnector(void);

   void              Init(string folder);
   
   //--- Outbox (MQL -> Python)
   void              SendSnapshot(double bid, double ask, double atr);
   
   //--- Inbox (Python -> MQL)
   string            ReadInbox(); // Returns raw JSON if new, else ""
   
   //--- Helpers
   string            GetFolder() { return m_folder; }
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CHubConnector::CHubConnector(void)
  {
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CHubConnector::~CHubConnector(void)
  {
  }
//+------------------------------------------------------------------+
//| Init                                                             |
//+------------------------------------------------------------------+
void CHubConnector::Init(string folder)
  {
   m_folder = folder;
  }
//+------------------------------------------------------------------+
//| Send Snapshot                                                    |
//+------------------------------------------------------------------+
void CHubConnector::SendSnapshot(double bid, double ask, double atr)
  {
   m_json_parser.InitObject();
   m_json_parser.AddString("type", "snapshot");
   m_json_parser.AddDouble("bid", bid);
   m_json_parser.AddDouble("ask", ask);
   m_json_parser.AddDouble("atr", atr);
   m_json_parser.AddString("timestamp", (string)TimeCurrent(), true); // Last one
   m_json_parser.CloseObject();
   
   string json = m_json_parser.GetJSON();
   
   int handle = FileOpen(m_folder + "/mql_to_py.json", FILE_WRITE|FILE_TXT|FILE_COMMON|FILE_ANSI);
   if(handle != INVALID_HANDLE)
     {
      FileWriteString(handle, json);
      FileClose(handle);
     }
  }
//+------------------------------------------------------------------+
//| Read Inbox                                                       |
//+------------------------------------------------------------------+
string CHubConnector::ReadInbox()
  {
   string filename = m_folder + "/py_to_mql.json";
   
   if(FileIsExist(filename, FILE_COMMON))
     {
      int handle = FileOpen(filename, FILE_READ|FILE_TXT|FILE_COMMON|FILE_ANSI);
      if(handle != INVALID_HANDLE)
        {
         string content = "";
         while(!FileIsEnding(handle))
            content += FileReadString(handle);
            
         FileClose(handle);
         
         // Optional: Delete or Rename to avoid re-reading? 
         // For now, we assume the content has a 'req_id' we track elsewhere to avoid dupes.
         // Or we can delete it.
         // FileDelete(filename, FILE_COMMON); 
         
         return content;
        }
     }
   return "";
  }
//+------------------------------------------------------------------+
