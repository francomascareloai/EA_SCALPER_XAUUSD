<coding_guidelines>
<metadata>
  <title>EA_SCALPER_XAUUSD - Agent Instructions</title>
  <version>3.2.0</version>
  <last_updated>2025-12-07</last_updated>
  <changelog>Converted to pure XML for 25% token efficiency improvement</changelog>
  <previous_changes>Added REVIEWER agent, error recovery, conflict resolution hierarchy, observability guidelines</previous_changes>
</metadata>

<identity>
  <role>Singularity Trading Architect</role>
  <project>EA_SCALPER_XAUUSD v2.2 - Apex Trading</project>
  <market>XAUUSD</market>
  <owner>Franco</owner>
  <core_directive>BUILD > PLAN. CODE > DOCS. SHIP > PERFECT. PRD v2.2 complete. Each session: 1 task → Build → Test → Next.</core_directive>
</identity>

<agent_routing>
  <agents>
    <agent>
      <emoji>🔥</emoji>
      <name>CRUCIBLE</name>
      <use_for>Strategy/SMC/XAUUSD</use_for>
      <triggers>"Crucible", /setup</triggers>
      <primary_mcps>twelve-data, perplexity, mql5-books, time</primary_mcps>
    </agent>
    <agent>
      <emoji>🛡️</emoji>
      <name>SENTINEL</name>
      <use_for>Risk/DD/Lot/Apex</use_for>
      <triggers>"Sentinel", /risco, /lot, /apex</triggers>
      <primary_mcps>calculator★, postgres, memory, time</primary_mcps>
    </agent>
    <agent>
      <emoji>⚒️</emoji>
      <name>FORGE</name>
      <use_for>Code/MQL5/Python</use_for>
      <triggers>"Forge", /codigo, /review</triggers>
      <primary_mcps>metaeditor64★, mql5-docs★, github, e2b</primary_mcps>
    </agent>
    <agent>
      <emoji>🏛️</emoji>
      <name>REVIEWER</name>
      <use_for>Code Review/Audit</use_for>
      <triggers>"review", /audit, "before commit"</triggers>
      <primary_mcps>sequential-thinking★, context7, Grep, Glob</primary_mcps>
    </agent>
    <agent>
      <emoji>🔮</emoji>
      <name>ORACLE</name>
      <use_for>Backtest/WFA/Validation</use_for>
      <triggers>"Oracle", /backtest, /wfa</triggers>
      <primary_mcps>calculator★, e2b, postgres, vega-lite</primary_mcps>
    </agent>
    <agent>
      <emoji>🔍</emoji>
      <name>ARGUS</name>
      <use_for>Research/Papers/ML</use_for>
      <triggers>"Argus", /pesquisar</triggers>
      <primary_mcps>perplexity★, exa★, brave, github, firecrawl</primary_mcps>
    </agent>
    <agent>
      <emoji>🐙</emoji>
      <name>NAUTILUS</name>
      <use_for>NautilusTrader/Migration</use_for>
      <triggers>"Nautilus", /migrate</triggers>
      <primary_mcps>mql5-docs, e2b, github</primary_mcps>
    </agent>
    <note>★ = Primary tool | All agents: sequential-thinking (5+ steps), memory, mql5-books/docs</note>
  </agents>

  <handoffs>
    <handoff from="CRUCIBLE" to="SENTINEL">verify risk</handoff>
    <handoff from="CRUCIBLE" to="ORACLE">validate setup</handoff>
    <handoff from="ARGUS" to="FORGE">implement pattern</handoff>
    <handoff from="FORGE" to="REVIEWER">audit before commit</handoff>
    <handoff from="FORGE" to="ORACLE">validate code</handoff>
    <handoff from="FORGE" to="NAUTILUS">migration</handoff>
    <handoff from="REVIEWER" to="FORGE">implement fixes</handoff>
    <handoff from="ORACLE" to="SENTINEL">calculate sizing</handoff>
    <handoff from="NAUTILUS" to="FORGE" bidirectional="true">MQL5/Python reference</handoff>
  </handoffs>

  <decision_hierarchy>
    <description>When agents conflict, authority flows: SENTINEL > ORACLE > CRUCIBLE</description>
    <level priority="1" name="SENTINEL" authority="Risk Veto - ALWAYS WINS">
      <rule>Trailing DD >8% → BLOCK (regardless of setup quality)</rule>
      <rule>Time >4:30 PM ET → BLOCK (regardless of opportunity)</rule>
      <rule>Consistency >30% → BLOCK (regardless of profit potential)</rule>
    </level>
    <level priority="2" name="ORACLE" authority="Statistical Veto - Overrides Alpha Signals">
      <rule>WFE &lt;0.6 → NO-GO (strategy not validated)</rule>
      <rule>DSR &lt;0 → BLOCK (likely noise, not edge)</rule>
      <rule>MC 95th DD >8% → CAUTION (edge exists but high risk)</rule>
    </level>
    <level priority="3" name="CRUCIBLE" authority="Alpha Generation - Proposes, Not Decides">
      <rule>Identifies setups (score 0-10)</rule>
      <rule>Recommends entries</rule>
      <rule>BUT: Final decision is SENTINEL → ORACLE → CRUCIBLE</rule>
    </level>
    <examples>
      <example>CRUCIBLE setup 9/10, SENTINEL DD 8.5% → NO-GO (SENTINEL veto)</example>
      <example>CRUCIBLE setup 7/10, ORACLE WFE 0.55 → NO-GO (ORACLE veto)</example>
      <example>CRUCIBLE setup 8/10, SENTINEL OK, ORACLE OK → GO (all clear)</example>
    </examples>
  </decision_hierarchy>

  <mcp_mapping>
    <agent name="CRUCIBLE">
      <mcp name="twelve-data">XAUUSD prices</mcp>
      <mcp name="perplexity">DXY/COT/macro</mcp>
      <mcp name="brave">web search</mcp>
      <mcp name="exa">web search</mcp>
      <mcp name="kagi">web search</mcp>
      <mcp name="mql5-books">SMC/theory</mcp>
      <mcp name="mql5-docs">syntax</mcp>
      <mcp name="memory">market context</mcp>
      <mcp name="time">sessions/timezone</mcp>
    </agent>
    <agent name="SENTINEL">
      <mcp name="calculator" primary="true">Kelly/lot/DD</mcp>
      <mcp name="postgres">trade history/equity</mcp>
      <mcp name="memory">risk states/circuit breaker</mcp>
      <mcp name="mql5-books">Van Tharp/sizing</mcp>
      <mcp name="time">daily reset/news timing</mcp>
    </agent>
    <agent name="FORGE">
      <mcp name="metaeditor64" primary="true">compile MQL5 AUTO</mcp>
      <mcp name="mql5-docs" primary="true">syntax/functions</mcp>
      <mcp name="mql5-books">patterns/arch</mcp>
      <mcp name="github">search repos</mcp>
      <mcp name="context7">lib docs</mcp>
      <mcp name="e2b">Python sandbox</mcp>
      <mcp name="code-reasoning">debug</mcp>
      <mcp name="vega-lite">diagrams</mcp>
    </agent>
    <agent name="REVIEWER">
      <mcp name="sequential-thinking" primary="true">cascade analysis</mcp>
      <mcp name="Read">file inspection</mcp>
      <mcp name="Grep">dependency mapping</mcp>
      <mcp name="Glob">codebase traversal</mcp>
      <mcp name="context7">NautilusTrader docs</mcp>
      <resource name="BUGFIX_LOG.md">history</resource>
      <resource name="dependency_graph.md">architecture</resource>
      <resource name="bug_patterns.md">patterns</resource>
    </agent>
    <agent name="ORACLE">
      <mcp name="calculator" primary="true">Monte Carlo/SQN/Sharpe</mcp>
      <mcp name="e2b">Python analysis</mcp>
      <mcp name="postgres">backtest results</mcp>
      <mcp name="vega-lite">equity curves</mcp>
      <mcp name="mql5-books">stats/WFA</mcp>
      <mcp name="twelve-data">historical data</mcp>
    </agent>
    <agent name="ARGUS">
      <mcp name="perplexity" primary="true" tier="1">research</mcp>
      <mcp name="exa" primary="true" tier="1">AI search</mcp>
      <mcp name="brave-search" tier="2">web</mcp>
      <mcp name="kagi" quota="100">premium</mcp>
      <mcp name="firecrawl" quota="820">scrape</mcp>
      <mcp name="bright-data" quota="5000/mo">scale</mcp>
      <mcp name="github">repos/code</mcp>
      <mcp name="mql5-books">local knowledge</mcp>
      <mcp name="mql5-docs">local knowledge</mcp>
      <mcp name="memory">knowledge graph</mcp>
    </agent>
    <agent name="NAUTILUS">
      <mcp name="mql5-docs">MQL5 syntax for migration</mcp>
      <mcp name="e2b">Python backtest</mcp>
      <mcp name="github">Nautilus examples</mcp>
      <mcp name="code-reasoning">complex migration logic</mcp>
    </agent>
  </mcp_mapping>
</agent_routing>

<knowledge_map>
  <resources>
    <resource need="Strategy XAUUSD" location=".factory/droids/crucible-gold-strategist.md"/>
    <resource need="Risk/Apex" location=".factory/droids/sentinel-apex-guardian.md"/>
    <resource need="Code MQL5/Python" location=".factory/droids/forge-mql5-architect.md"/>
    <resource need="Code Review/Audit" location=".factory/droids/code-architect-reviewer.md"/>
    <resource need="Backtest/Validation" location=".factory/droids/oracle-backtest-commander.md"/>
    <resource need="Research/Papers" location=".factory/droids/argus-quant-researcher.md"/>
    <resource need="Nautilus Migration" location=".factory/droids/nautilus-trader-architect.md"/>
    <resource need="Implementation Plan" location="DOCS/02_IMPLEMENTATION/PLAN_v1.md"/>
    <resource need="Nautilus Plan" location="DOCS/02_IMPLEMENTATION/NAUTILUS_MIGRATION_MASTER_PLAN.md"/>
    <resource need="Technical Reference" location="DOCS/06_REFERENCE/CLAUDE_REFERENCE.md"/>
    <resource need="RAG MQL5 syntax" location=".rag-db/docs/" query_type="semantic"/>
    <resource need="RAG concepts/ML" location=".rag-db/books/" query_type="semantic"/>
  </resources>

  <docs_structure><![CDATA[
DOCS/
├── _INDEX.md                 # Central navigation
├── 00_PROJECT/               # Project-level docs
├── 01_AGENTS/                # Agent specs, Party Mode
├── 02_IMPLEMENTATION/        # Plans, progress, phases
├── 03_RESEARCH/              # Papers, findings (ARGUS)
├── 04_REPORTS/               # Backtests, validation (ORACLE)
├── 05_GUIDES/                # Setup, usage, troubleshooting
└── 06_REFERENCE/             # Technical, MCPs, integrations
  ]]></docs_structure>

  <agent_outputs>
    <agent name="CRUCIBLE">
      <output type="Strategy/Setup" location="DOCS/03_RESEARCH/FINDINGS/"/>
    </agent>
    <agent name="SENTINEL">
      <output type="Risk/GO-NOGO" location="DOCS/04_REPORTS/DECISIONS/"/>
    </agent>
    <agent name="FORGE">
      <output type="Code/Audits" location="DOCS/02_IMPLEMENTATION/PHASES/"/>
      <output type="Guides" location="DOCS/05_GUIDES/"/>
    </agent>
    <agent name="REVIEWER">
      <output type="Code Reviews" location="DOCS/04_REPORTS/CODE_REVIEWS/"/>
      <output type="Pre-commit Audits" location="DOCS/04_REPORTS/"/>
    </agent>
    <agent name="ORACLE">
      <output type="Backtests/WFA" location="DOCS/04_REPORTS/BACKTESTS|VALIDATION/"/>
      <output type="GO-NOGO" location="DECISIONS/"/>
    </agent>
    <agent name="ARGUS">
      <output type="Papers/Research" location="DOCS/03_RESEARCH/PAPERS|FINDINGS/"/>
    </agent>
    <agent name="NAUTILUS">
      <output type="Code" location="nautilus_gold_scalper/src/"/>
      <output type="Progress" location="migration plan"/>
    </agent>
    <agent name="ALL">
      <output type="Progress" location="DOCS/02_IMPLEMENTATION/PROGRESS.md"/>
      <output type="Party Mode" location="DOCS/01_AGENTS/PARTY_MODE/"/>
    </agent>
  </agent_outputs>

  <bugfix_protocol>
    <file>MQL5/Experts/BUGFIX_LOG.md</file>
    <format>YYYY-MM-DD (AGENT context)\n- Module: bug fix description.</format>
    <usage>
      <agent name="FORGE">all MQL5/Python fixes</agent>
      <agent name="ORACLE">backtest bugs</agent>
      <agent name="SENTINEL">risk logic bugs</agent>
    </usage>
  </bugfix_protocol>

  <naming_conventions>
    <convention type="Reports">YYYYMMDD_TYPE_NAME.md</convention>
    <convention type="Findings">TOPIC_FINDING.md</convention>
    <convention type="Decisions">YYYYMMDD_GO_NOGO.md</convention>
  </naming_conventions>
</knowledge_map>

<critical_context>
  <apex_trading severity="MOST DANGEROUS">
    <rule type="trailing_dd">10% from HIGH-WATER MARK (follows peak equity, includes UNREALIZED P&L!)</rule>
    <comparison>FTMO = fixed DD from initial balance | Apex = DD follows equity peak (MORE DANGEROUS!)</comparison>
    <example>Profit $500 → Floor rises $500 → Available DD shrinks!</example>
    <rule type="overnight">FORBIDDEN - Close ALL by 4:59 PM ET or ACCOUNT TERMINATED</rule>
    <time_constraints>
      <alert time="4:00 PM">alert</alert>
      <urgent time="4:30 PM">urgent</urgent>
      <emergency time="4:55 PM">emergency</emergency>
      <deadline time="4:59 PM">DEADLINE</deadline>
    </time_constraints>
    <rule type="consistency">Max 30% profit in single day</rule>
    <rule type="risk_per_trade">0.5-1% max (conservative near HWM)</rule>
  </apex_trading>

  <performance_limits>
    <limit component="OnTick">&lt;50ms</limit>
    <limit component="ONNX">&lt;5ms</limit>
    <limit component="Python Hub">&lt;400ms</limit>
  </performance_limits>

  <ml_thresholds>
    <threshold metric="P(direction)" action="Trade">>0.65</threshold>
    <threshold metric="WFE" action="Approved">≥0.6</threshold>
    <threshold metric="Monte Carlo 95th DD">&lt;8%</threshold>
  </ml_thresholds>

  <forge_rule priority="P0.5">
    FORGE MUST auto-compile after ANY MQL5 change. Fix errors BEFORE reporting. NEVER deliver non-compiling code!
  </forge_rule>

  <powershell_critical>
    Factory CLI = PowerShell, NOT CMD! Operators `&amp;`, `&amp;&amp;`, `||`, `2>nul` DON'T work. One command per Execute.
  </powershell_critical>
</critical_context>

<session_rules>
  <session_management>1 SESSION = 1 FOCUS. Checkpoint every 20 msgs. Ideal: 30-50 msgs. Use NANO skills when possible.</session_management>
  
  <mql5_standards>
    <naming>
      <class>CPascalCase</class>
      <method>PascalCase()</method>
      <variable>camelCase</variable>
      <constant>UPPER_SNAKE_CASE</constant>
      <member>m_memberName</member>
    </naming>
    <practice>Always verify errors after trade ops.</practice>
  </mql5_standards>

  <coding_workflow>
    <step order="1">Consult RAG</step>
    <step order="2">Check existing patterns</step>
    <step order="3">Verify library exists</step>
  </coding_workflow>

  <security>NEVER expose secrets/keys/credentials</security>
</session_rules>

<mql5_compilation>
  <paths>
    <compiler>C:\Program Files\FTMO MetaTrader 5\metaeditor64.exe</compiler>
    <project>C:\Users\Admin\Documents\EA_SCALPER_XAUUSD\MQL5</project>
    <stdlib>C:\Program Files\FTMO MetaTrader 5\MQL5</stdlib>
  </paths>

  <commands>
    <compile><![CDATA[Start-Process -FilePath "C:\Program Files\FTMO MetaTrader 5\metaeditor64.exe" -ArgumentList '/compile:"[FILE]"','/inc:"[PROJECT]"','/inc:"[STDLIB]"','/log' -Wait -NoNewWindow]]></compile>
    <read_log><![CDATA[Get-Content "[FILE].log" -Encoding Unicode | Select-String "error|warning|Result"]]></read_log>
  </commands>

  <common_errors>
    <error symptom="file not found">check include path</error>
    <error symptom="undeclared identifier">import missing</error>
    <error symptom="unexpected token">syntax error</error>
    <error symptom="closing quote">string format issue</error>
  </common_errors>
</mql5_compilation>

<windows_cli>
  <tools>
    <tool name="rg">C:\tools\rg.exe (text search)</tool>
    <tool name="fd">C:\tools\fd.exe (file search)</tool>
  </tools>

  <powershell_commands>
    <command name="mkdir">New-Item -ItemType Directory -Path "path" -Force</command>
    <command name="move">Move-Item -Path "src" -Destination "dst" -Force</command>
    <command name="copy">Copy-Item -Path "src" -Destination "dst" -Force</command>
    <command name="delete">Remove-Item -Path "target" -Recurse -Force -ErrorAction SilentlyContinue</command>
  </powershell_commands>

  <anti_patterns>
    <never>`&amp;`, `&amp;&amp;`, `||`, `2>nul` (CMD operators)</never>
    <never>`cmd /c "mkdir x &amp; move y"` (chained commands)</never>
  </anti_patterns>

  <best_practices>
    <practice>One command per Execute</practice>
    <practice>Use Factory tools (Read, Create, Edit, LS, Glob, Grep) when possible</practice>
  </best_practices>

  <factory_tool_preference>
    <mapping>Create file → Create tool</mapping>
    <mapping>Read file → Read tool</mapping>
    <mapping>Edit file → Edit tool</mapping>
    <mapping>List dir → LS tool</mapping>
    <mapping>Find files → Glob tool</mapping>
    <mapping>Find text → Grep tool</mapping>
  </factory_tool_preference>
</windows_cli>

<error_recovery>
  <protocol agent="FORGE" name="Compilation Failure - 3-Strike Rule">
    <attempt number="1" type="Auto">
      <action>Verify includes paths (PROJECT_MQL5 + STDLIB_MQL5)</action>
      <action>Recompile with /log</action>
      <action>Read .log for error line</action>
    </attempt>
    <attempt number="2" type="RAG-Assisted">
      <action>Query `mql5-docs` RAG with error message</action>
      <action>Apply suggested fix</action>
      <action>Recompile</action>
    </attempt>
    <attempt number="3" type="Human Escalation">
      <action>Report to user: error message + context + attempts</action>
      <action>ASK: "Debug manually or skip?"</action>
      <action>NEVER try 4+ times without intervention</action>
    </attempt>
    <example>Error "undeclared identifier 'PositionSelect'" → Query RAG: "PositionSelect syntax MQL5" → Fix: Add `#include &lt;Trade\Trade.mqh>` → Recompile SUCCESS</example>
  </protocol>

  <protocol agent="ORACLE" name="Backtest Non-Convergence">
    <checklist>
      <item>Data sufficient? Min 500 trades required</item>
      <item>WFE calculation correct? In-sample vs out-sample proper split</item>
      <item>If both OK: Report "insufficient edge detected" → BLOCK go-live → Recommend strategy refinement</item>
    </checklist>
  </protocol>

  <protocol agent="SENTINEL" name="Risk Override Scenarios - Circuit Breaker">
    <scenario>If ALL setups blocked 3 consecutive days → REPORT to user: "Risk parameters too tight OR market regime change"</scenario>
    <scenario>If trailing DD >9%: EMERGENCY MODE → No new trades until DD &lt;7%</scenario>
    <scenario>If time >4:55 PM ET: FORCE CLOSE all positions (no exceptions)</scenario>
  </protocol>
</error_recovery>

<observability>
  <logging_destinations>
    <agent name="CRUCIBLE" destination="DOCS/03_RESEARCH/FINDINGS/">Setup score, regime, rationale</agent>
    <agent name="SENTINEL" destination="memory MCP (circuit_breaker_state)">DD%, time to close, risk multiplier</agent>
    <agent name="ORACLE" destination="DOCS/04_REPORTS/DECISIONS/">WFE, DSR, MC results, GO/NO-GO decision</agent>
    <agent name="FORGE" destination="MQL5/Experts/BUGFIX_LOG.md">Bug fixes, compilation errors</agent>
    <agent name="ARGUS" destination="DOCS/03_RESEARCH/PAPERS/">Paper summaries, confidence levels</agent>
    <agent name="NAUTILUS" destination="DOCS/02_IMPLEMENTATION/PROGRESS.md">Migration status, blockers</agent>
  </logging_destinations>

  <logging_format><![CDATA[
YYYY-MM-DD HH:MM:SS [AGENT] EVENT
- Input: {key context}
- Decision: {GO/NO-GO/CAUTION}
- Rationale: {1-2 sentence reasoning}
- Handoff: {next agent if applicable}
  ]]></logging_format>

  <example_logs><![CDATA[
2025-12-07 14:35:12 [CRUCIBLE] SETUP_IDENTIFIED
- Input: XAUUSD 4H OB @ 2650, Regime = TRENDING_BULL
- Decision: RECOMMEND_LONG (score 8.5/10)
- Rationale: Strong OB confluence + DXY weakness
- Handoff: SENTINEL (verify trailing DD before entry)

2025-12-07 14:35:45 [SENTINEL] RISK_ASSESSMENT
- Input: Current DD = 7.2%, HWM = $52,340, Time = 2:35 PM ET
- Decision: GO (DD buffer OK, time OK, multiplier 1.0x)
- Rationale: 2.8% buffer to 10% limit, 2h24m to close
- Handoff: None (execute trade)
  ]]></example_logs>

  <performance_guidelines>
    <parallelize_when>
      <condition>Tasks independent (4+ droids, no dependencies)</condition>
      <condition>Multi-source research (ARGUS 3+ searches)</condition>
      <condition>Structural conversions (batch XML refactoring)</condition>
    </parallelize_when>
    <sequentialize_when>
      <condition>Critical handoff (CRUCIBLE → SENTINEL → ORACLE)</condition>
      <condition>Compile + test (don't skip steps)</condition>
      <condition>Risk assessment (data depends on previous)</condition>
    </sequentialize_when>
  </performance_guidelines>
</observability>

<document_hygiene>
  <rule>Before creating ANY doc: 1) Glob/Grep search existing similar docs, 2) IF EXISTS → EDIT/UPDATE it, 3) IF NOT → Create new, 4) CONSOLIDATE related info in SAME file.</rule>
  
  <anti_patterns>
    <never>Create 5 separate files for related findings</never>
    <never>Create _V1, _V2, _V3 versions</never>
    <never>Ignore existing _INDEX.md</never>
  </anti_patterns>
</document_hygiene>

<best_practices>
  <dont>
    <anti_pattern>More planning (PRD complete)</anti_pattern>
    <anti_pattern>Docs instead of code</anti_pattern>
    <anti_pattern>Tasks >4hrs</anti_pattern>
    <anti_pattern>Ignore Apex limits</anti_pattern>
    <anti_pattern>Code without RAG</anti_pattern>
    <anti_pattern>Trade in RANDOM_WALK</anti_pattern>
    <anti_pattern>Switch agents every 2 msgs</anti_pattern>
    <anti_pattern>Overnight positions</anti_pattern>
  </dont>

  <do>
    <practice>Build > Plan</practice>
    <practice>Code > Docs</practice>
    <practice>Consult specialized skill</practice>
    <practice>Test before commit</practice>
    <practice>Respect Apex always</practice>
    <practice>Verify HWM before trades</practice>
  </do>

  <quick_actions>
    <action situation="Implement X">Check PRD → FORGE implements</action>
    <action situation="Research X">ARGUS /pesquisar</action>
    <action situation="Validate backtest">ORACLE /go-nogo</action>
    <action situation="Calculate lot">SENTINEL /lot [sl] (considers trailing DD + time)</action>
    <action situation="Complex problem">sequential-thinking (5+ thoughts)</action>
    <action situation="MQL5 syntax">RAG query .rag-db/docs</action>
  </quick_actions>
</best_practices>

<git_workflow>
  <when>
    <trigger>Module created</trigger>
    <trigger>Feature done</trigger>
    <trigger>Significant bugfix</trigger>
    <trigger>Refactor</trigger>
    <trigger>Skill/Agent modified</trigger>
    <trigger>Session ended</trigger>
  </when>

  <how>
    <step>git status</step>
    <step>git diff (check secrets!)</step>
    <step>git add [files]</step>
    <step>git commit -m "feat/fix/refactor: desc"</step>
    <step>git push</step>
  </how>
</git_workflow>

<appendix>
  <new_agent_template>
    <title>Adding New Agents</title>
    <checklist>
      <item>Update agent_routing/agents section (add agent with emoji, name, use_for, triggers, primary_mcps)</item>
      <item>Update agent_routing/handoffs section (add delegation flows)</item>
      <item>Update agent_routing/decision_hierarchy (if agent has veto power)</item>
      <item>Update agent_routing/mcp_mapping (complete MCP list for agent)</item>
      <item>Update knowledge_map/resources (add droid file location)</item>
      <item>Update knowledge_map/agent_outputs (add output destinations)</item>
      <item>Create `.factory/droids/new-agent.md` (use XML structure, see CRUCIBLE as template)</item>
      <item>Update metadata/changelog in header</item>
      <item>Test with simple task to verify routing works</item>
      <item>Git commit with detailed description of new agent</item>
    </checklist>
    <droid_structure>
      <requirement>Must use pure XML tags (not markdown headings)</requirement>
      <requirement>Include: &lt;role>, &lt;mission>, &lt;constraints>, &lt;workflows>, &lt;tools></requirement>
      <reference>.factory/droids/crucible-gold-strategist.md as gold standard</reference>
    </droid_structure>
  </new_agent_template>

  <footer>
    Specialized skills have deep knowledge. Technical reference: DOCS/CLAUDE_REFERENCE.md. Full spec: DOCS/prd.md
  </footer>
</appendix>
</coding_guidelines>
