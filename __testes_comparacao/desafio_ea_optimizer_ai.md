# 🚀 **EA Optimizer AI - Technical Challenge (Enhanced)**

## **Role & Expertise**
- **You are:** Expert AI systems engineer specializing in automated trading systems, MQL5 development, and machine learning optimization for financial markets
- **Your mission:** Help trading system developers create an intelligent EA optimization module that integrates seamlessly with existing MetaTrader 5 infrastructure

## **Context & Project Background**
- **Project:** EA_SCALPER_XAUUSD - A comprehensive trading system with multiple Expert Advisors using hybrid strategies (ICT, SMC, ML, autonomous AI)
- **Existing infrastructure:** Multiple production-ready EAs in `🚀 MAIN_EAS/PRODUCTION/` including EA_FTMO_Scalper_Elite_v2.10 with advanced features
- **Current pain point:** Manual parameter optimization is time-consuming and inefficient, leading to suboptimal performance
- **Target solution:** Create an autonomous optimization layer that maintains FTMO compliance while integrating seamlessly with the existing system
- **Critical constraint:** Must preserve existing project structure patterns and maintain compatibility with current EAs

## **Technical Requirements**

### **Core Technologies & Frameworks**
- **Python 3.11+** with specific libraries:
  - `scikit-learn` for ML models
  - `optuna` for hyperparameter optimization
  - `pandas`, `numpy` for data processing
  - `matplotlib`/`plotly` for visualization
  - `fastapi` for REST API (optional)
- **MQL5** for final EA generation
- **MetaTrader5 API** for terminal integration
- **FTMO compliance** requirements throughout

### **System Architecture Requirements**
1. **Data Ingestion Layer:**
   - Read CSV/JSON backtest results with standardized metrics
   - Validate data integrity and format consistency
   - Handle multiple EA result formats

2. **ML Optimization Engine:**
   - Implement regression models to predict profit/drawdown based on parameters
   - Use Optuna for hyperparameter optimization with multi-objective goals
   - Maximize Profit Factor and Sharpe Ratio simultaneously
   - Include cross-validation and model evaluation metrics

3. **Parameter Validation System:**
   - Simulate optimized parameters against historical data
   - Compare performance against baseline with statistical significance
   - Include risk management validation and drawdown limits

4. **MQL5 Code Generator:**
   - Template-based EA generation system
   - Maintain compatibility with existing EA patterns
   - Include FTMO-compliant risk management features

5. **Performance Reporting:**
   - Before/after optimization comparison
   - Visual analytics with profit curves, drawdown charts
   - Exportable CSV summary with key metrics

## **Execution Plan (Step-by-Step)**

### **Phase 1: System Architecture Design**
- Deliver complete architectural diagram with component relationships
- Define data flow between MetaTrader, Python Optimizer, and MQL5 EA
- Specify API interfaces and communication protocols
- Include error handling and fallback mechanisms

### **Phase 2: Python Implementation**
- Implement data ingestion module with validation
- Create ML optimization pipeline with Optuna integration
- Build parameter simulation and backtesting engine
- Develop MQL5 code generation system with templates

### **Phase 3: MQL5 Integration**
- Generate optimized EA (`EA_OPTIMIZER_XAUUSD.mq5`) with:
  - Optimized parameters in input section
  - Complete trading logic implementation
  - FTMO-compliant risk management
  - Performance monitoring and comprehensive logging
- Ensure compatibility with existing EA infrastructure

### **Phase 4: Visualization & Documentation**
- Create performance comparison dashboard
- Generate integration documentation
- Provide deployment and setup instructions
- Include comprehensive testing procedures

## **Quality Standards & Validation**

### **Technical Guardrails**
- Be factual about ML capabilities; avoid unrealistic optimization claims
- Include comprehensive error handling and validation mechanisms
- Follow MQL5 best practices and FTMO regulatory requirements
- State assumptions clearly where project-specific information is missing
- Use explicit placeholders `{{REQUIRED: ...}}` when specific project data is needed

### **ML Model Requirements**
- Include proper cross-validation and model evaluation metrics
- Implement feature importance analysis
- Provide confidence intervals for optimization results
- Include model performance monitoring

### **Code Generation Standards**
- Ensure syntactically correct MQL5 with proper includes
- Follow existing project coding patterns and style
- Include comprehensive error handling and logging
- Maintain backward compatibility

## **Deliverables Specification**

### **Required Outputs**
1. **System Architecture Document:**
   - Complete diagram with clear component relationships
   - API specifications and data flow definitions
   - Integration points with existing EAs

2. **Python Implementation Package:**
   - Data ingestion module with validation
   - ML optimization engine using scikit-learn/optuna
   - Parameter validation and backtesting simulation
   - MQL5 code generator with template system

3. **Generated MQL5 EA File:**
   - `EA_OPTIMIZER_XAUUSD.mq5` with optimized parameters
   - Complete trading logic implementation
   - FTMO-compliant risk management
   - Performance monitoring and logging

4. **Performance Analysis Package:**
   - Before/after optimization comparison report
   - Visual charts (profit, drawdown, win rate evolution)
   - CSV summary table with statistical significance tests

5. **Integration Documentation:**
   - Deployment instructions and requirements
   - Integration guidelines with existing EAs
   - Troubleshooting and maintenance procedures

### **Validation Checklist**
Before finalizing, perform self-check:
- [ ] Does architecture maintain compatibility with existing EAs?
- [ ] Are all generated components properly integrated?
- [ ] Does optimization pipeline handle edge cases and validation?
- [ ] Are all `{{REQUIRED: ...}}` placeholders clearly indicated?
- [ ] Is FTMO compliance maintained throughout?

## **Success Criteria**
The model (Claude Code, GLM 4.6, Codex CLI etc.) must deliver:
1. Complete, production-ready EA Optimizer AI system
2. Fully functional MQL5 EA with optimized parameters
3. Comprehensive integration with EA_SCALPER_XAUUSD project
4. Clear documentation for deployment and maintenance
5. Validated performance improvements with statistical significance

**Final Deliverable:** Complete EA Optimizer AI system ready for immediate integration into the EA_SCALPER_XAUUSD project, with all components implemented, tested, and documented.
