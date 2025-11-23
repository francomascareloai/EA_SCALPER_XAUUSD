# API Contracts: CLI Proxy (Go)

## Overview
The CLI Proxy (`CLIPROXY`) acts as a unified gateway for various AI providers (OpenAI, Anthropic, Google Gemini), exposing standard API compatible endpoints.

## Base URL
`http://localhost:{port}` (Default port configured in `config.yaml`)

## Endpoints

### OpenAI Compatible API (`/v1`)
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/models` | List available models (Unified) |
| `POST` | `/v1/chat/completions` | OpenAI Chat Completions API |
| `POST` | `/v1/completions` | Legacy Completions API |

### Anthropic Compatible API (`/v1`)
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/messages` | Claude Messages API |
| `POST` | `/v1/messages/count_tokens` | Token counting endpoint |

### Google Gemini Compatible API (`/v1beta`)
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1beta/models` | List Gemini models |
| `POST` | `/v1beta/models/:action` | Generic Gemini action handler |
| `GET` | `/v1beta/models/:action` | Generic Gemini getter handler |

### Management API (`/v0/management`)
*Requires Management Secret*

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v0/management/config` | Get current configuration |
| `PUT` | `/v0/management/config.yaml` | Update configuration |
| `GET` | `/v0/management/usage` | Get usage statistics |
| `GET` | `/v0/management/logs` | Retrieve server logs |

## Authentication
*   **Standard API**: Bearer Token (Provider API Key or Proxy Token)
*   **Management API**: `Authorization: Bearer <MANAGEMENT_PASSWORD>` or `X-Local-Password` header.

## Websockets
*   `/v1/ws`: Websocket upgrade endpoint for real-time communication (if enabled).
