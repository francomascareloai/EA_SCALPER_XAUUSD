import os
import json
import hashlib
import litellm
from litellm import completion
from diskcache import Cache

class HierarchicalPromptCache:
    def __init__(self, cache_dir="./prompt_cache"):
        self.memory_cache = {}
        self.disk_cache = Cache(cache_dir)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _generate_key(self, prompt, model):
        """Generate unique key for prompt+model combination"""
        return hashlib.sha256(f"{model}-{prompt}".encode()).hexdigest()
    
    def get(self, prompt, model):
        key = self._generate_key(prompt, model)
        
        # Check memory cache first
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Check disk cache
        if key in self.disk_cache:
            result = self.disk_cache[key]
            # Store in memory for future access
            self.memory_cache[key] = result
            return result
            
        return None
    
    def set(self, prompt, model, result):
        key = self._generate_key(prompt, model)
        
        # Store in memory
        self.memory_cache[key] = result
        
        # Store on disk
        self.disk_cache[key] = result
        
    def cache_stats(self):
        return {
            "memory_items": len(self.memory_cache),
            "disk_items": len(self.disk_cache),
            "cache_dir": self.cache_dir
        }

class LiteLLMWithCache:
    def __init__(self, cache_dir="./prompt_cache"):
        self.cache = HierarchicalPromptCache(cache_dir)
        self.llm = litellm
        
        # Configure LiteLLM - adapt to your setup
        self.llm.set_verbose = True
        self.default_model = "gpt-4"
        
    def query_llm(self, prompt, model=None, use_cache=True, **kwargs):
        model = model or self.default_model
        
        if use_cache:
            cached = self.cache.get(prompt, model)
            if cached:
                return cached
        
        # Call LLM API
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        
        result = response.choices[0].message.content
        
        if use_cache:
            self.cache.set(prompt, model, result)
            
        return result

# Example usage
if __name__ == "__main__":
    llm = LiteLLMWithCache(cache_dir="./prompt_cache")
    
    # First query (will call API)
    prompt = "Explain Fibonacci retracement in trading"
    print("First call (API):")
    response = llm.query_llm(prompt, model="gpt-4")
    print(response[:200] + "...")
    
    # Second query (will use cache)
    print("\nSecond call (Cache):")
    cached_response = llm.query_llm(prompt, model="gpt-4")
    print(cached_response[:200] + "...")
    
    # Show cache stats
    print("\nCache Stats:", llm.cache.cache_stats())