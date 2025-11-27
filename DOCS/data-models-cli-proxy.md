# Data Models: CLI Proxy (Go)

## Overview
This document outlines the key data structures used in the CLI Proxy (`CLIPROXY`).

## Core Structures

### ModelInfo
Defines the properties of a supported AI model.

| Field | Type | Description |
|-------|------|-------------|
| `ID` | string | Unique model identifier (e.g., `gpt-5`, `claude-3-5-sonnet`) |
| `Object` | string | Object type (usually "model") |
| `Created` | int64 | Creation timestamp |
| `OwnedBy` | string | Provider name (`openai`, `anthropic`, `google`) |
| `Type` | string | Internal type classifier |
| `DisplayName` | string | Human-readable name |
| `Description` | string | Model capabilities description |
| `ContextLength` | int | Max context window size |
| `MaxCompletionTokens` | int | Max output tokens |

### Server Configuration (`config.Config`)
*Inferred from usage*

| Field | Type | Description |
|-------|------|-------------|
| `Port` | int | Server listening port |
| `Debug` | bool | Enable debug logging |
| `RequestLog` | bool | Enable request logging |
| `AuthDir` | string | Directory for OAuth tokens |
| `OpenAICompatibility` | []Provider | List of compatible providers |
| `RemoteManagement` | struct | Management API settings |

## Supported Models
The proxy maintains static definitions for:
*   **Claude**: `claude-3-5-sonnet`, `claude-3-opus`, etc.
*   **Gemini**: `gemini-1.5-pro`, `gemini-1.5-flash`, etc.
*   **OpenAI**: `gpt-4o`, `gpt-4-turbo`, etc.
*   **Qwen**: `qwen-max`, `qwen-plus`, etc.

## Authentication Models
*   **OAuthCallback**: Stores `code`, `state`, and `error` for OAuth flows.
