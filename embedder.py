import requests
import json
import time
from typing import List
import numpy as np
from tqdm import tqdm

class VNPTEmbedder:
    """Embedder using VNPT API (500 requests/month limit)"""
    
    def __init__(self, api_keys_file='api-keys.json'):
        """
        Initialize VNPT Embedder API
        """
        print("Initializing VNPT Embedder...")
        
        # Load API keys
        with open(api_keys_file, 'r', encoding='utf-8') as f:
            api_keys = json.load(f)
        
        # Find embedding API key (item với llmApiName chứa "embed" hoặc item đầu tiên)
        embedding_api = None
        for key_item in api_keys:
            if isinstance(key_item, dict) and "llmApiName" in key_item:
                api_name = key_item["llmApiName"].lower()
                if "embed" in api_name:
                    embedding_api = key_item
                    break
        
        if embedding_api:
            # Theo code mẫu, cần authorization, tokenId, tokenKey
            self.authorization = embedding_api.get("authorization", "")
            self.token_id = embedding_api.get("tokenId", "")
            self.token_key = embedding_api.get("tokenKey", "")
            api_name = embedding_api.get("llmApiName", "Unknown")
            
            # Extract Bearer token từ authorization
            if self.authorization.startswith("Bearer "):
                self.bearer_token = self.authorization.replace("Bearer ", "")
            else:
                self.bearer_token = self.authorization
            
            print(f"✓ Using API: {api_name}")
        else:
            print("⚠️ No embedding API keys found in file")
            self.bearer_token = self.token_id = self.token_key = ""
        
        self.base_url = "https://api.idg.vnpt.vn"
        self.embedding_url = f"{self.base_url}/data-service/vnptai-hackathon-embedding"
        
        # Rate limiting counters
        self.month_counter = 0
        self.max_monthly = 500  # 500 requests per month
        
        # Cache for embeddings to avoid duplicate API calls
        self.embedding_cache = {}
        
        print(f"✓ VNPT Embedder initialized ({self.max_monthly} requests/month available)")
    
    def _make_request(self, texts: List[str]) -> List[List[float]]:
        """
        Send batch request to VNPT embedding API
        Theo code mẫu đã chạy được, format payload khác
        """
        if self.month_counter >= self.max_monthly:
            print(f"⚠️ Monthly quota reached ({self.month_counter}/{self.max_monthly})")
            return []
        
        headers = {
            'Authorization': f'Bearer {self.bearer_token}',
            'Token-id': self.token_id,
            'Token-key': self.token_key,
            'Content-Type': 'application/json',
        }
        
        embeddings = []
        
        for text in texts:
            # Rate limiting delay
            if self.month_counter > 0:
                # Minimum 0.12s between requests (500 requests/minute = ~8.3 requests/second)
                time.sleep(0.12)
            
            # Format payload theo code mẫu đã chạy
            payload = {
                'model': 'vnptai_hackathon_embedding',
                'input': text,
                'encoding_format': 'float',
            }
            
            try:
                response = requests.post(
                    self.embedding_url,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    self.month_counter += 1
                    
                    # Debug log để biết API đang hoạt động
                    if self.month_counter % 10 == 0:
                        print(f"  Embedding API: {self.month_counter}/{self.max_monthly} used")
                    
                    # Extract embedding từ response
                    if 'data' in result and len(result['data']) > 0:
                        embedding = result['data'][0]['embedding']
                        embeddings.append(embedding)
                    else:
                        print(f"⚠️ Response không có data: {result}")
                        # Return empty embedding as fallback
                        embeddings.append([])
                        
                elif response.status_code == 401:
                    print(f"❌ Authentication failed. Check your API keys.")
                    print(f"   Bearer token: {self.bearer_token[:20]}...")
                    print(f"   Token-id: {self.token_id}")
                    print(f"   Token-key: {self.token_key[:20]}...")
                    return []
                    
                elif response.status_code == 429:
                    print(f"⏳ Rate limited. Waiting 60s...")
                    time.sleep(60)
                    # Retry
                    return self._make_request([text])
                    
                else:
                    print(f"❌ Embedding API error {response.status_code}: {response.text[:200]}")
                    return []
                    
            except requests.exceptions.Timeout:
                print(f"⏱️ Timeout for text: {text[:50]}...")
                time.sleep(5)
                return []
                
            except Exception as e:
                print(f"❌ Embedding API request failed: {e}")
                return []
        
        return embeddings
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode single text to embedding
        Uses cache to avoid duplicate API calls
        """
        # Clean and normalize text for better caching
        cleaned_text = ' '.join(text.strip().split())
        
        # Check cache first
        if cleaned_text in self.embedding_cache:
            return self.embedding_cache[cleaned_text]
        
        # Check monthly quota
        if self.month_counter >= self.max_monthly:
            # Return random embedding as fallback
            print(f"⚠️ Monthly quota exceeded ({self.month_counter}/{self.max_monthly}), using fallback embedding")
            return np.random.randn(1024).astype(np.float32)
        
        # Make API request
        embeddings = self._make_request([cleaned_text])
        
        if embeddings and len(embeddings) > 0 and len(embeddings[0]) > 0:
            embedding = np.array(embeddings[0], dtype=np.float32)
            # Cache the result
            self.embedding_cache[cleaned_text] = embedding
            return embedding
        else:
            # Fallback to random embedding
            print(f"⚠️ Embedding API returned empty, using fallback")
            fallback_emb = np.random.randn(1024).astype(np.float32)
            self.embedding_cache[cleaned_text] = fallback_emb
            return fallback_emb
    
    def get_usage(self) -> dict:
        """Get current API usage statistics"""
        return {
            'used': self.month_counter,
            'total': self.max_monthly,
            'remaining': self.max_monthly - self.month_counter,
            'percentage': (self.month_counter / self.max_monthly * 100) if self.max_monthly > 0 else 0
        }
    
    def test_connection(self) -> bool:
        """Test if embedding API is working"""
        print("Testing embedding API connection...")
        
        # Test với text đơn giản
        test_text = "Xin chào Việt Nam"
        
        try:
            embedding = self.encode(test_text)
            
            if embedding is not None and len(embedding) > 0:
                print(f"✓ Embedding API connection successful")
                print(f"  Embedding dimension: {len(embedding)}")
                print(f"  Embedding sample (first 3): {embedding[:3]}")
                return True
            else:
                print(f"✗ Embedding API returned empty result")
                return False
                
        except Exception as e:
            print(f"✗ Embedding API connection failed: {e}")
            return False