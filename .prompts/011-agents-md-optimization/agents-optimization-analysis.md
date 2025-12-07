# AGENTS.md Optimization Analysis

```xml
<optimization_analysis>
  <current_state>
    <line_count>2607</line_count>
    <character_count>114063</character_count>
    <estimated_tokens>~28000</estimated_tokens>
    <major_sections>
      - metadata (15 lines)
      - identity (10 lines)
      - platform_support (50 lines)
      - strategic_intelligence (~1200 lines) ← MAJOR BLOAT
      - agent_routing (~200 lines)
      - knowledge_map (~120 lines)
      - critical_context (~80 lines)
      - session_rules (~40 lines)
      - mql5_compilation (~30 lines)
      - windows_cli (~40 lines)
      - error_recovery (~60 lines)
      - observability (~70 lines)
      - document_hygiene (~20 lines)
      - best_practices (~50 lines)
      - git_workflow (~30 lines)
      - appendix (~40 lines)
    </major_sections>
  </current_state>
  
  <findings>
    <redundancies>
      <item section="strategic_intelligence" issue="7 questions concept repeated in 5+ places" savings="50"/>
      <item section="strategic_intelligence" issue="Apex constraints mentioned 8+ times across sections" savings="40"/>
      <item section="amplifier_protocols + intelligence_amplifiers" issue="Amplifiers described twice with redundant detail" savings="100"/>
      <item section="pattern_recognition_library + pattern_learning" issue="Pattern concepts duplicated" savings="80"/>
      <item section="genius_mode_templates" issue="4 verbose CDATA templates with examples" savings="150"/>
      <item section="amplifier_protocols/usage_examples" issue="4 extremely verbose examples (~300 lines)" savings="200"/>
      <item section="thinking_observability" issue="Huge audit trail template + example" savings="100"/>
      <item section="thinking_conflicts" issue="Detailed resolution framework with long examples" savings="100"/>
      <item section="compressed_protocols" issue="Ironically verbose fast_mode section" savings="60"/>
      <item section="feedback_loop" issue="Verbose metrics and calibration examples" savings="80"/>
      <item section="agent_intelligence_gates" issue="Repetitive handoff gate definitions" savings="80"/>
    </redundancies>
    
    <optimization_opportunities>
      <opportunity type="structural" section="strategic_intelligence" 
                   description="Merge related subsections (genius_mode_triggers + genius_mode_templates)" 
                   impact="high"/>
      <opportunity type="content" section="amplifier_protocols" 
                   description="Reduce 4 examples to 2 concise ones" 
                   impact="high"/>
      <opportunity type="template" section="genius_mode_templates" 
                   description="Convert CDATA to compact tables/lists" 
                   impact="high"/>
      <opportunity type="content" section="thinking_observability" 
                   description="Remove verbose audit trail example, keep format only" 
                   impact="medium"/>
      <opportunity type="content" section="pattern_learning" 
                   description="Compress workflow steps and example" 
                   impact="medium"/>
      <opportunity type="structural" section="thinking_conflicts" 
                   description="Reduce resolution framework to essential steps" 
                   impact="medium"/>
      <opportunity type="example" section="compressed_protocols" 
                   description="Remove verbose fast_mode example" 
                   impact="medium"/>
      <opportunity type="content" section="feedback_loop" 
                   description="Compress metrics to table format" 
                   impact="medium"/>
      <opportunity type="structural" section="agent_intelligence_gates" 
                   description="Use attributes instead of nested descriptions" 
                   impact="medium"/>
      <opportunity type="content" section="proactive_problem_detection" 
                   description="Convert to bulleted lists" 
                   impact="low"/>
    </optimization_opportunities>
    
    <critical_content>
      <section name="metadata" reason="Version tracking and changelog - MUST preserve"/>
      <section name="identity" reason="Core agent role definition - MUST preserve"/>
      <section name="platform_support" reason="Nautilus/MQL5 routing rules - MUST preserve"/>
      <section name="mandatory_reflection_protocol" reason="Core thinking framework - MUST preserve (can compress)"/>
      <section name="agent_routing/agents" reason="Agent definitions and triggers - MUST preserve"/>
      <section name="decision_hierarchy" reason="SENTINEL > ORACLE > CRUCIBLE authority - MUST preserve"/>
      <section name="critical_context/apex_trading" reason="Risk management rules - MUST preserve"/>
      <section name="critical_context/forge_rule" reason="Code validation requirement - MUST preserve"/>
      <section name="error_recovery" reason="3-strike protocols - MUST preserve"/>
      <section name="mql5_compilation" reason="Compiler paths and commands - MUST preserve"/>
    </critical_content>
  </findings>
  
  <optimization_plan>
    <phase number="1" focus="Structural compression">
      <action>Merge intelligence_amplifiers INTO amplifier_protocols</action>
      <action>Combine pattern_recognition_library with pattern_learning concepts</action>
      <action>Remove redundant Apex constraint mentions (reference critical_context)</action>
      <action>Flatten deeply nested XML where attributes suffice</action>
    </phase>
    <phase number="2" focus="Content deduplication">
      <action>Replace repeated "7 questions" explanations with reference</action>
      <action>Consolidate amplifier descriptions to single source</action>
      <action>Remove duplicate pattern definitions</action>
      <action>Use "See section X" references instead of repeating</action>
    </phase>
    <phase number="3" focus="Template optimization">
      <action>Reduce genius_mode_templates from 4 to 2 (keep new_feature + bug_fix)</action>
      <action>Convert CDATA output_formats to compact markdown tables</action>
      <action>Remove before/after examples (pattern clear from checklist)</action>
    </phase>
    <phase number="4" focus="Example consolidation">
      <action>Reduce amplifier_protocols/usage_examples from 4 to 1 best example</action>
      <action>Remove verbose thinking_observability audit trail example</action>
      <action>Compress pattern_learning example to essential steps</action>
      <action>Remove fast_mode verbose example</action>
    </phase>
    <phase number="5" focus="Prose reduction">
      <action>Convert verbose descriptions to imperative directives</action>
      <action>Remove philosophical/motivational content</action>
      <action>Use bullet points instead of prose paragraphs</action>
      <action>Eliminate redundant "MUST", "ALWAYS" where context is clear</action>
    </phase>
  </optimization_plan>
  
  <projected_outcome>
    <target_line_count>1250-1400</target_line_count>
    <target_characters>55000-60000</target_characters>
    <target_tokens>~14000</target_tokens>
    <reduction_percentage>48-52%</reduction_percentage>
    <risks>
      <risk level="low">Agent routing clarity may need verification after compression</risk>
      <risk level="low">Some context in examples provides implicit guidance</risk>
      <mitigation>Validate all agent triggers and handoffs still clear</mitigation>
      <mitigation>Keep one representative example per major concept</mitigation>
    </risks>
  </projected_outcome>
</optimization_analysis>
```

## Summary

The AGENTS.md file has significant optimization potential. The `strategic_intelligence` section alone accounts for ~46% of the file (1200+ lines) with extensive redundancy in:

1. **Amplifier descriptions** - Defined twice with 4 verbose examples
2. **Pattern systems** - Duplicated across 2 sections
3. **Templates** - 4 CDATA templates when 2 would suffice
4. **Verbose examples** - Many sections have 100+ line examples

Target reduction: **50%** (from 2607 to ~1300 lines) while preserving ALL critical functionality.
