"""
Simple Trading Proxy for Roo Code
Proxy HTTP simples com cache para OpenRouter
"""
import json
import hashlib
import time
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import httpx
from dotenv import load_dotenv

load_dotenv()

class TradingProxyHandler(BaseHTTPRequestHandler):
    # Cache em memória da classe
    cache = {}
    request_count = 0
    last_request_time = 0
    
    def __init__(self, *args, **kwargs):
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        self.openrouter_base = "https://openrouter.ai/api/v1"
        
        # Modelos disponíveis
        self.models = {
            "qwen-coder": "qwen/qwen3-coder:free",
            "deepseek-r1": "deepseek/deepseek-r1-0528:free",
            "default": "deepseek/deepseek-r1-0528:free"
        }
        
        super().__init__(*args, **kwargs)
    
    def do_OPTIONS(self):
        """Manipular CORS preflight"""
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()
    
    def send_cors_headers(self):
        """Headers CORS para Roo Code"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    
    def do_GET(self):
        """Health check e informações"""
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            
            response = {
                "status": "healthy",
                "models": list(self.models.keys()),
                "cache_size": len(self.cache),
                "request_count": self.request_count
            }
            self.wfile.write(json.dumps(response).encode())
            
        elif self.path == '/v1/models' or self.path == '/models':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            
            models_list = {
                "data": [
                    {"id": model, "object": "model", "created": int(time.time()), "owned_by": "trading-proxy"}
                    for model in self.models.keys()
                ]
            }
            self.wfile.write(json.dumps(models_list).encode())
            
        elif self.path == '/v1/model/info' or self.path == '/model/info':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            
            model_info = {
                "status": "active",
                "available_models": list(self.models.keys()),
                "default_model": "deepseek-r1",
                "features": ["chat", "completion", "caching"],
                "version": "1.0.0"
            }
            self.wfile.write(json.dumps(model_info).encode())
            
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        """Processar requests do Roo Code"""
        if self.path == '/v1/chat/completions' or self.path == '/chat/completions':
            self.handle_chat_completion()
        else:
            self.send_response(404)
            self.end_headers()
    
    def handle_chat_completion(self):
        """Manipular chat completion com cache"""
        try:
            # Ler dados da requisição
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode())
            
            # Rate limiting básico
            now = time.time()
            if now - self.last_request_time < 2.0:
                time.sleep(2.0 - (now - self.last_request_time))
            
            # Extrair informações
            model = request_data.get('model', 'default')
            messages = request_data.get('messages', [])
            
            # Gerar cache key
            cache_content = f"{model}:{json.dumps(messages)}"
            cache_key = hashlib.md5(cache_content.encode()).hexdigest()
            
            # Verificar cache
            if cache_key in self.cache:
                print(f"💾 Cache hit: {model} - {cache_key[:8]}")
                self.send_cached_response(self.cache[cache_key])
                return
            
            # Mapear modelo
            openrouter_model = self.models.get(model, self.models['default'])
            
            # Preparar requisição para OpenRouter
            or_request = {
                "model": openrouter_model,
                "messages": messages,
                "max_tokens": request_data.get('max_tokens', 1000),
                "temperature": request_data.get('temperature', 0.1)
            }
            
            # Headers para OpenRouter
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/trading-organizer",
                "X-Title": "Simple Trading Proxy"
            }
            
            print(f"🔄 Request: {model} -> {openrouter_model}")
            
            # Fazer requisição para OpenRouter
            with httpx.Client() as client:
                response = client.post(
                    f"{self.openrouter_base}/chat/completions",
                    headers=headers,
                    json=or_request,
                    timeout=30.0
                )
                
                self.last_request_time = time.time()
                self.request_count += 1
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Salvar no cache
                    self.cache[cache_key] = result
                    print(f"💾 Cached: {cache_key[:8]} (total: {len(self.cache)})")
                    
                    # Enviar resposta
                    self.send_json_response(result)
                    
                else:
                    error_response = {
                        "error": {
                            "message": f"OpenRouter error: {response.status_code}",
                            "type": "api_error",
                            "code": response.status_code
                        }
                    }
                    self.send_json_response(error_response, status=response.status_code)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            error_response = {
                "error": {
                    "message": f"Proxy error: {str(e)}",
                    "type": "internal_error"
                }
            }
            self.send_json_response(error_response, status=500)
    
    def send_cached_response(self, data):
        """Enviar resposta do cache"""
        self.send_json_response(data)
    
    def send_json_response(self, data, status=200):
        """Enviar resposta JSON"""
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def log_message(self, format, *args):
        """Log customizado"""
        print(f"🌐 {self.address_string()} - {format % args}")

def run_proxy(host='0.0.0.0', port=4000):
    """Iniciar o servidor proxy"""
    
    # Verificar API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ OPENROUTER_API_KEY não encontrada no arquivo .env")
        return False
    
    # Obter IP local da máquina
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "localhost"
    
    print("🚀 SIMPLE TRADING PROXY INICIANDO...")
    print("=" * 50)
    print(f"📡 Host: {host}:{port} (escutando todos IPs)")
    print(f"🌐 IP Local: {local_ip}:{port}")
    print(f"🔑 API Key: {api_key[:15]}...")
    print(f"🤖 Modelos: qwen-coder, deepseek-r1")
    print(f"💾 Cache: Ativo em memória")
    print("=" * 50)
    print("")
    print("🔌 PARA ROO CODE:")
    print(f"   Base URL: http://{local_ip}:{port}")
    print(f"   Base URL (com v1): http://{local_ip}:{port}/v1")
    print(f"   Localhost: http://127.0.0.1:{port}")
    print(f"   Localhost (com v1): http://127.0.0.1:{port}/v1")
    print(f"   API Key: qualquer-chave-funciona")
    print(f"   Modelo: qwen-coder ou deepseek-r1")
    print("")
    print(f"📊 Health Check: http://{local_ip}:{port}/health")
    print("🛑 Para parar: Ctrl+C")
    print("=" * 50)
    
    try:
        server = HTTPServer((host, port), TradingProxyHandler)
        server.serve_forever()
        
    except KeyboardInterrupt:
        print("\n🛑 Proxy interrompido pelo usuário")
        server.shutdown()
        
    except Exception as e:
        print(f"❌ Erro no servidor: {e}")
        return False
    
    return True

if __name__ == "__main__":
    run_proxy()
