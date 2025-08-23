
//+------------------------------------------------------------------+
//|                                              Interfaces.mqh |
//|                        TradeDev_Master Elite Trading System     |
//|                                   Advanced Interface Definitions|
//+------------------------------------------------------------------+

#ifndef INTERFACES_MQH
#define INTERFACES_MQH

#include "DataStructures.mqh"

// Interface base para todos os detectores
class IDetector
{
public:
   virtual bool Initialize() = 0;
   virtual bool Update() = 0;
   virtual void Reset() = 0;
   virtual bool IsValid() = 0;
   virtual string GetStatus() = 0;
};

// Interface para análise de confluência
class IConfluenceAnalyzer
{
public:
   virtual double CalculateScore() = 0;
   virtual bool IsValidSignal() = 0;
   virtual SSignalConfluence GetConfluenceData() = 0;
   virtual void UpdateWeights(double &weights[]) = 0;
};

// Interface para gestão de cache
class ICacheManager
{
public:
   virtual bool Set(string key, string value, int expiry_seconds = 3600) = 0;
   virtual string Get(string key) = 0;
   virtual bool Exists(string key) = 0;
   virtual bool Delete(string key) = 0;
   virtual void Clear() = 0;
   virtual int GetSize() = 0;
};

// Interface para análise de performance
class IPerformanceAnalyzer
{
public:
   virtual void UpdateMetrics() = 0;
   virtual SPerformanceMetrics GetMetrics() = 0;
   virtual double GetSharpeRatio() = 0;
   virtual double GetMaxDrawdown() = 0;
   virtual bool IsPerformanceAcceptable() = 0;
   virtual string GenerateReport() = 0;
};

// Interface para gestão de configuração
class IConfigManager
{
public:
   virtual bool LoadConfig(string filename) = 0;
   virtual bool SaveConfig(string filename) = 0;
   virtual string GetValue(string section, string key, string default_value = "") = 0;
   virtual bool SetValue(string section, string key, string value) = 0;
   virtual bool DeleteKey(string section, string key) = 0;
   virtual void Reset() = 0;
};

// Interface para logging
class ILogger
{
public:
   virtual void SetLevel(ENUM_LOG_LEVEL level) = 0;
   virtual void Debug(string message) = 0;
   virtual void Info(string message) = 0;
   virtual void Warning(string message) = 0;
   virtual void Error(string message) = 0;
   virtual void Critical(string message) = 0;
   virtual bool SaveToFile(string filename) = 0;
};

// Interface para detectores ICT/SMC
class IOrderBlockDetector : public IDetector
{
public:
   virtual SOrderBlock[] GetOrderBlocks() = 0;
   virtual bool IsOrderBlockValid(SOrderBlock &ob) = 0;
   virtual double GetOrderBlockStrength(SOrderBlock &ob) = 0;
};

class IFVGDetector : public IDetector
{
public:
   virtual SFVG[] GetFVGs() = 0;
   virtual bool IsFVGValid(SFVG &fvg) = 0;
   virtual double GetFVGStrength(SFVG &fvg) = 0;
};

class ILiquidityDetector : public IDetector
{
public:
   virtual SLiquidityLevel[] GetLiquidityLevels() = 0;
   virtual bool IsLiquidityLevelValid(SLiquidityLevel &level) = 0;
   virtual double GetLiquidityStrength(SLiquidityLevel &level) = 0;
};

class IMarketStructureAnalyzer : public IDetector
{
public:
   virtual SMarketStructure GetMarketStructure() = 0;
   virtual bool IsStructureBullish() = 0;
   virtual bool IsStructureBearish() = 0;
   virtual bool IsCHoCHDetected() = 0;
   virtual bool IsBOSDetected() = 0;
};

#endif // INTERFACES_MQH
