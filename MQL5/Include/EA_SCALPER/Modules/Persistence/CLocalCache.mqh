//+------------------------------------------------------------------+
//|                                                  CLocalCache.mqh |
//|                                                   MQL5 Architect |
//|                                      Copyright 2025, Elite Ops.  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Elite Ops."
#property version   "1.00"

#include <Files/FileTxt.mqh>
#include "../../Utils/CJson.mqh"
#include "../Core/CState.mqh"

class CLocalCache
  {
private:
   string            m_folder;
   CJson             m_json_parser;

public:
                     CLocalCache(void);
                    ~CLocalCache(void);

   void              Init(string folder);
   
   SRiskParams       LoadRiskParams();
   // SNewsEvent[] LoadNews(); // Complex parsing needed for arrays, skipping for MVP
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CLocalCache::CLocalCache(void)
  {
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CLocalCache::~CLocalCache(void)
  {
  }
//+------------------------------------------------------------------+
//| Init                                                             |
//+------------------------------------------------------------------+
void CLocalCache::Init(string folder)
  {
   m_folder = folder;
  }
//+------------------------------------------------------------------+
//| Load Risk Params                                                 |
//+------------------------------------------------------------------+
SRiskParams CLocalCache::LoadRiskParams()
  {
   SRiskParams params; // Default values
   
   string filename = m_folder + "/risk_params.json";
   if(FileIsExist(filename, FILE_COMMON))
     {
      int handle = FileOpen(filename, FILE_READ|FILE_TXT|FILE_COMMON|FILE_ANSI);
      if(handle != INVALID_HANDLE)
        {
         string content = "";
         while(!FileIsEnding(handle))
            content += FileReadString(handle);
         FileClose(handle);
         
         m_json_parser.SetJSON(content);
         
         double r = m_json_parser.GetDouble("risk_per_trade");
         if(r > 0) params.risk_per_trade = r;
         
         double d = m_json_parser.GetDouble("max_daily_loss");
         if(d > 0) params.max_daily_loss = d;
         
         Print("CLocalCache: Risk Params Loaded. Risk=", params.risk_per_trade);
        }
     }
   else
     {
      Print("CLocalCache: No risk_params.json found. Using defaults.");
     }
     
   return params;
  }
//+------------------------------------------------------------------+
