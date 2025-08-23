
//+------------------------------------------------------------------+
//|                                            ConfigManager.mqh |
//|                        TradeDev_Master Elite Trading System     |
//|                                   Advanced Configuration Manager|
//+------------------------------------------------------------------+

#ifndef CONFIG_MANAGER_MQH
#define CONFIG_MANAGER_MQH

#include "Interfaces.mqh"
#include "DataStructures.mqh"

class CConfigManager : public IConfigManager
{
private:
   SConfigData m_config_data[];
   int m_data_count;
   string m_current_file;
   
public:
   CConfigManager() : m_data_count(0), m_current_file("")
   {
      ArrayResize(m_config_data, 100);
   }
   
   virtual bool LoadConfig(string filename) override
   {
      m_current_file = filename;
      
      int file_handle = FileOpen(filename, FILE_READ|FILE_TXT);
      if(file_handle == INVALID_HANDLE)
         return false;
         
      m_data_count = 0;
      string current_section = "";
      
      while(!FileIsEnding(file_handle))
      {
         string line = FileReadString(file_handle);
         line = StringTrimLeft(StringTrimRight(line));
         
         if(StringLen(line) == 0 || StringGetCharacter(line, 0) == ';')
            continue;
            
         if(StringGetCharacter(line, 0) == '[' && StringGetCharacter(line, StringLen(line)-1) == ']')
         {
            current_section = StringSubstr(line, 1, StringLen(line)-2);
            continue;
         }
         
         int pos = StringFind(line, "=");
         if(pos > 0)
         {
            string key = StringTrimRight(StringSubstr(line, 0, pos));
            string value = StringTrimLeft(StringSubstr(line, pos+1));
            
            if(m_data_count < ArraySize(m_config_data))
            {
               m_config_data[m_data_count].section = current_section;
               m_config_data[m_data_count].key = key;
               m_config_data[m_data_count].value = value;
               m_config_data[m_data_count].description = "";
               m_config_data[m_data_count].is_encrypted = false;
               m_data_count++;
            }
         }
      }
      
      FileClose(file_handle);
      return true;
   }
   
   virtual bool SaveConfig(string filename) override
   {
      int file_handle = FileOpen(filename, FILE_WRITE|FILE_TXT);
      if(file_handle == INVALID_HANDLE)
         return false;
         
      string current_section = "";
      
      for(int i = 0; i < m_data_count; i++)
      {
         if(m_config_data[i].section != current_section)
         {
            current_section = m_config_data[i].section;
            if(StringLen(current_section) > 0)
               FileWrite(file_handle, "[" + current_section + "]");
         }
         
         FileWrite(file_handle, m_config_data[i].key + "=" + m_config_data[i].value);
      }
      
      FileClose(file_handle);
      return true;
   }
   
   virtual string GetValue(string section, string key, string default_value = "") override
   {
      for(int i = 0; i < m_data_count; i++)
      {
         if(m_config_data[i].section == section && m_config_data[i].key == key)
            return m_config_data[i].value;
      }
      return default_value;
   }
   
   virtual bool SetValue(string section, string key, string value) override
   {
      // Try to update existing
      for(int i = 0; i < m_data_count; i++)
      {
         if(m_config_data[i].section == section && m_config_data[i].key == key)
         {
            m_config_data[i].value = value;
            return true;
         }
      }
      
      // Add new if not found
      if(m_data_count < ArraySize(m_config_data))
      {
         m_config_data[m_data_count].section = section;
         m_config_data[m_data_count].key = key;
         m_config_data[m_data_count].value = value;
         m_config_data[m_data_count].description = "";
         m_config_data[m_data_count].is_encrypted = false;
         m_data_count++;
         return true;
      }
      
      return false;
   }
   
   virtual bool DeleteKey(string section, string key) override
   {
      for(int i = 0; i < m_data_count; i++)
      {
         if(m_config_data[i].section == section && m_config_data[i].key == key)
         {
            // Shift array elements
            for(int j = i; j < m_data_count - 1; j++)
            {
               m_config_data[j] = m_config_data[j + 1];
            }
            m_data_count--;
            return true;
         }
      }
      return false;
   }
   
   virtual void Reset() override
   {
      m_data_count = 0;
      m_current_file = "";
   }
   
   // Additional utility methods
   double GetDoubleValue(string section, string key, double default_value = 0.0)
   {
      string str_value = GetValue(section, key);
      if(StringLen(str_value) == 0)
         return default_value;
      return StringToDouble(str_value);
   }
   
   int GetIntValue(string section, string key, int default_value = 0)
   {
      string str_value = GetValue(section, key);
      if(StringLen(str_value) == 0)
         return default_value;
      return (int)StringToInteger(str_value);
   }
   
   bool GetBoolValue(string section, string key, bool default_value = false)
   {
      string str_value = GetValue(section, key);
      if(StringLen(str_value) == 0)
         return default_value;
      return (StringToLower(str_value) == "true" || str_value == "1");
   }
};

#endif // CONFIG_MANAGER_MQH
