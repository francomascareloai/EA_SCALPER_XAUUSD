# Data Models: AI Agents (Python)

## Overview
This document outlines the data structures used for agent configuration and coordination in the AI Agents system (`🤖 AI_AGENTS`).

## Agent Configuration
Agents are defined using JSON configuration files.

### Agent Definition Schema
| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | string | Unique identifier (e.g., `agent_master_coordinator`) |
| `name` | string | Human-readable name |
| `description` | string | Functional description |
| `version` | string | Agent version |
| `status` | string | Current status (`active`, `inactive`) |
| `mcp_servers` | array[string] | List of required MCP servers |
| `capabilities` | array[string] | List of agent capabilities |

## Coordination Configuration
Defined in `AGENT_COORDINATION.json`.

### Coordination Schema
| Field | Type | Description |
|-------|------|-------------|
| `section` | string | System section identifier |
| `description` | string | System description |
| `agent_types` | object | Map of agent roles to capabilities |
| `communication_protocols` | object | Definitions of communication channels |
| `mcp_integration` | object | MCP settings and supported tools |

## Communication Protocols

### Message Structure (Inferred)
| Field | Type | Description |
|-------|------|-------------|
| `sender_id` | string | ID of the sending agent |
| `recipient_id` | string | ID of the target agent (or `broadcast`) |
| `message_type` | string | Type of message (`task`, `event`, `data`) |
| `payload` | object | The actual data content |
| `timestamp` | string | ISO 8601 timestamp |
