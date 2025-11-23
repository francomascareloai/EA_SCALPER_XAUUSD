//+------------------------------------------------------------------+
//|                                                        CJson.mqh |
//|                                                   MQL5 Architect |
//|                                      Copyright 2025, Elite Ops.  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Elite Ops."
#property version   "1.00"

class CJson
  {
private:
   string            m_json;

public:
                     CJson(void) { m_json = "{}"; }
                    ~CJson(void) {}

   //--- Setters
   void              SetJSON(string json) { m_json = json; }
   string            GetJSON() { return m_json; }

   //--- Simple Parser (Flat JSON only for MVP)
   string            GetString(string key)
     {
      int keyPos = StringFind(m_json, "\"" + key + "\"");
      if(keyPos == -1) return "";

      int colonPos = StringFind(m_json, ":", keyPos);
      if(colonPos == -1) return "";

      int startQuote = StringFind(m_json, "\"", colonPos);
      if(startQuote == -1) return "";

      int endQuote = StringFind(m_json, "\"", startQuote + 1);
      if(endQuote == -1) return "";

      return StringSubstr(m_json, startQuote + 1, endQuote - startQuote - 1);
     }

   double            GetDouble(string key)
     {
      int keyPos = StringFind(m_json, "\"" + key + "\"");
      if(keyPos == -1) return 0.0;

      int colonPos = StringFind(m_json, ":", keyPos);
      if(colonPos == -1) return 0.0;

      // Find value start (skip spaces)
      int valStart = colonPos + 1;
      while(valStart < StringLen(m_json) && (StringGetCharacter(m_json, valStart) == ' ' || StringGetCharacter(m_json, valStart) == '\t'))
         valStart++;

      // Find value end (comma or closing brace)
      int valEnd = valStart;
      while(valEnd < StringLen(m_json))
        {
         ushort c = StringGetCharacter(m_json, valEnd);
         if(c == ',' || c == '}' || c == ' ' || c == '\n') break;
         valEnd++;
        }

      string valStr = StringSubstr(m_json, valStart, valEnd - valStart);
      return StringToDouble(valStr);
     }
     
   //--- Builder
   void              InitObject() { m_json = "{"; }
   void              AddString(string key, string value, bool isLast=false)
     {
      m_json += "\"" + key + "\":\"" + value + "\"" + (isLast ? "" : ",");
     }
   void              AddDouble(string key, double value, bool isLast=false)
     {
      m_json += "\"" + key + "\":" + DoubleToString(value, 5) + (isLast ? "" : ",");
     }
   void              CloseObject() { m_json += "}"; }
  };
//+------------------------------------------------------------------+
