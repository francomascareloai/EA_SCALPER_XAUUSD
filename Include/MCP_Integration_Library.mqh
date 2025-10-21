// MCP Integration Library (stub)
// This header is intentionally lightweight. Implementations live in the EA.
// It exists to satisfy the #include and keep the EA self-contained.

// Declarations only where safe. Implementations are provided inside the EA
// after all structs/types are defined to avoid forward-declaration issues.

bool InitializeMCPIntegration();
void CleanupMCPIntegration();
void CheckAIOptimization();
double GetEffectiveConfluenceThreshold();
bool CheckEnhancedEmergencyConditions();

// Using out-param to avoid returning incomplete type if included earlier.
// The EA also provides a convenience wrapper returning by value.
struct SConfluenceSignal; // forward decl
bool GenerateAIEnhancedConfluenceSignal(SConfluenceSignal &out_signal);
bool ValidateTradeWithAI(const SConfluenceSignal& signal, bool &approved);

