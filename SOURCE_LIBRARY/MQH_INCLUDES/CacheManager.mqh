
//+------------------------------------------------------------------+
//|                                             CacheManager.mqh |
//|                        TradeDev_Master Elite Trading System     |
//|                                   High-Performance Cache System |
//+------------------------------------------------------------------+

#ifndef CACHE_MANAGER_MQH
#define CACHE_MANAGER_MQH

#include "Interfaces.mqh"
#include "DataStructures.mqh"

class CCacheManager : public ICacheManager
{
private:
   SCacheEntry m_cache_entries[];
   int m_entry_count;
   int m_max_entries;
   datetime m_last_cleanup;
   
public:
   CCacheManager(int max_entries = 1000) : m_entry_count(0), m_max_entries(max_entries), m_last_cleanup(0)
   {
      ArrayResize(m_cache_entries, m_max_entries);
   }
   
   virtual bool Set(string key, string value, int expiry_seconds = 3600) override
   {
      CleanupExpired();
      
      // Try to update existing entry
      for(int i = 0; i < m_entry_count; i++)
      {
         if(m_cache_entries[i].key == key)
         {
            m_cache_entries[i].value = value;
            m_cache_entries[i].timestamp = TimeCurrent();
            m_cache_entries[i].expiry = TimeCurrent() + expiry_seconds;
            m_cache_entries[i].is_valid = true;
            return true;
         }
      }
      
      // Add new entry if space available
      if(m_entry_count < m_max_entries)
      {
         m_cache_entries[m_entry_count].key = key;
         m_cache_entries[m_entry_count].value = value;
         m_cache_entries[m_entry_count].timestamp = TimeCurrent();
         m_cache_entries[m_entry_count].expiry = TimeCurrent() + expiry_seconds;
         m_cache_entries[m_entry_count].is_valid = true;
         m_entry_count++;
         return true;
      }
      
      // Cache is full, remove oldest entry
      RemoveOldestEntry();
      return Set(key, value, expiry_seconds);
   }
   
   virtual string Get(string key) override
   {
      for(int i = 0; i < m_entry_count; i++)
      {
         if(m_cache_entries[i].key == key && m_cache_entries[i].is_valid)
         {
            if(TimeCurrent() <= m_cache_entries[i].expiry)
            {
               return m_cache_entries[i].value;
            }
            else
            {
               m_cache_entries[i].is_valid = false;
            }
         }
      }
      return "";
   }
   
   virtual bool Exists(string key) override
   {
      for(int i = 0; i < m_entry_count; i++)
      {
         if(m_cache_entries[i].key == key && m_cache_entries[i].is_valid)
         {
            if(TimeCurrent() <= m_cache_entries[i].expiry)
            {
               return true;
            }
            else
            {
               m_cache_entries[i].is_valid = false;
            }
         }
      }
      return false;
   }
   
   virtual bool Delete(string key) override
   {
      for(int i = 0; i < m_entry_count; i++)
      {
         if(m_cache_entries[i].key == key)
         {
            // Shift array elements
            for(int j = i; j < m_entry_count - 1; j++)
            {
               m_cache_entries[j] = m_cache_entries[j + 1];
            }
            m_entry_count--;
            return true;
         }
      }
      return false;
   }
   
   virtual void Clear() override
   {
      m_entry_count = 0;
   }
   
   virtual int GetSize() override
   {
      return m_entry_count;
   }
   
   // Additional utility methods
   double GetDouble(string key, double default_value = 0.0)
   {
      string value = Get(key);
      if(StringLen(value) == 0)
         return default_value;
      return StringToDouble(value);
   }
   
   int GetInt(string key, int default_value = 0)
   {
      string value = Get(key);
      if(StringLen(value) == 0)
         return default_value;
      return (int)StringToInteger(value);
   }
   
   bool GetBool(string key, bool default_value = false)
   {
      string value = Get(key);
      if(StringLen(value) == 0)
         return default_value;
      return (StringToLower(value) == "true" || value == "1");
   }
   
   void SetDouble(string key, double value, int expiry_seconds = 3600)
   {
      Set(key, DoubleToString(value, 8), expiry_seconds);
   }
   
   void SetInt(string key, int value, int expiry_seconds = 3600)
   {
      Set(key, IntegerToString(value), expiry_seconds);
   }
   
   void SetBool(string key, bool value, int expiry_seconds = 3600)
   {
      Set(key, value ? "true" : "false", expiry_seconds);
   }
   
private:
   void CleanupExpired()
   {
      if(TimeCurrent() - m_last_cleanup < 60) // Cleanup every minute
         return;
         
      for(int i = m_entry_count - 1; i >= 0; i--)
      {
         if(!m_cache_entries[i].is_valid || TimeCurrent() > m_cache_entries[i].expiry)
         {
            // Shift array elements
            for(int j = i; j < m_entry_count - 1; j++)
            {
               m_cache_entries[j] = m_cache_entries[j + 1];
            }
            m_entry_count--;
         }
      }
      
      m_last_cleanup = TimeCurrent();
   }
   
   void RemoveOldestEntry()
   {
      if(m_entry_count == 0) return;
      
      int oldest_index = 0;
      datetime oldest_time = m_cache_entries[0].timestamp;
      
      for(int i = 1; i < m_entry_count; i++)
      {
         if(m_cache_entries[i].timestamp < oldest_time)
         {
            oldest_time = m_cache_entries[i].timestamp;
            oldest_index = i;
         }
      }
      
      // Shift array elements
      for(int j = oldest_index; j < m_entry_count - 1; j++)
      {
         m_cache_entries[j] = m_cache_entries[j + 1];
      }
      m_entry_count--;
   }
};

#endif // CACHE_MANAGER_MQH
