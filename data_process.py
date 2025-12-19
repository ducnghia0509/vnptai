import json
import os
import time
import requests
import numpy as np
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime

class SimpleEmbeddingCreator:
    def __init__(self,
                 api_key: str,
                 token_id: str,
                 token_key: str,
                 max_workers: int = 3,
                 base_url: str = "https://api.idg.vnpt.vn"):
        
        self.api_key = api_key
        self.token_id = token_id
        self.token_key = token_key
        self.max_workers = max_workers
        self.base_url = base_url
        
        # Rate limiting settings
        self.request_lock = threading.Lock()
        self.requests_per_minute = 499
        self.requests_count = 0
        self.reset_time = time.time() + 60
        self.min_interval = 60.0 / self.requests_per_minute
        self.last_request_time = 0

    def _rate_limit(self):
        """Rate limiting cho API calls"""
        with self.request_lock:
            current_time = time.time()
            
            if current_time > self.reset_time:
                self.requests_count = 0
                self.reset_time = current_time + 60
                self.last_request_time = 0
            
            if self.requests_count >= self.requests_per_minute:
                sleep_time = self.reset_time - current_time
                time.sleep(sleep_time + 1)
                self.requests_count = 0
                self.reset_time = time.time() + 60
                self.last_request_time = time.time()
            
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_interval:
                time.sleep(self.min_interval - time_since_last)
            
            self.requests_count += 1
            self.last_request_time = time.time()

    def _get_embedding_single(self, text: str) -> List[float]:
        """Lấy embedding cho một text"""
        url = f"{self.base_url}/data-service/vnptai-hackathon-embedding"
        
        headers = {
            'Authorization': f'{self.api_key}',
            'Token-id': self.token_id,
            'Token-key': self.token_key,
            'Content-Type': 'application/json',
        }
        
        json_data = {
            'model': 'vnptai_hackathon_embedding',
            'input': text,
            'encoding_format': 'float',
        }
        
        self._rate_limit()
        
        try:
            response = requests.post(url, headers=headers, json=json_data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'data' in result and len(result['data']) > 0:
                    return result['data'][0]['embedding']
            return []
            
        except Exception as e:
            print(f"❌ Lỗi: {e}")
            return []

    def process_texts(self, texts: List[str]) -> List[List[float]]:
        """Xử lý nhiều texts song song"""
        embeddings = [None] * len(texts)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self._get_embedding_single, text): idx
                for idx, text in enumerate(texts)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    embedding = future.result()
                    embeddings[idx] = embedding
                except Exception as e:
                    print(f"❌ Lỗi trong thread: {e}")
                    embeddings[idx] = []
        
        return embeddings

def load_texts_from_jsonl(file_path: str, max_texts: int = None) -> List[Dict]:
    """Load texts từ file JSONL"""
    texts_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_texts and i >= max_texts:
                break
                
            try:
                item = json.loads(line)
                texts_data.append({
                    "text": item.get("text", ""),
                    "metadata": {k: v for k, v in item.items() if k != "text"}
                })
            except:
                continue
    
    return texts_data

def create_embeddings(input_files: List[str], 
                     output_dir: str,
                     api_key: str,
                     token_id: str,
                     token_key: str,
                     max_texts_per_file: int = None):
    """Tạo embeddings từ các file input"""
    
    # Tạo output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Khởi tạo embedding creator
    creator = SimpleEmbeddingCreator(api_key, token_id, token_key)
    
    total_processed = 0
    total_success = 0
    
    for file_idx, input_file in enumerate(input_files):
        print(f"\n📄 Processing file {file_idx + 1}/{len(input_files)}: {input_file}")
        
        # Load texts từ file
        texts_data = load_texts_from_jsonl(input_file, max_texts_per_file)
        
        if not texts_data:
            print(f"  ⚠️ No texts found in {input_file}")
            continue
        
        print(f"  Found {len(texts_data)} texts")
        
        # Lấy embeddings
        texts = [item["text"] for item in texts_data]
        embeddings = creator.process_texts(texts)
        
        # Lưu kết quả - giữ cấu trúc thư mục domain
        # Lấy domain name từ đường dẫn (thư mục cha của file)
        domain_name = os.path.basename(os.path.dirname(input_file))
        domain_output_dir = os.path.join(output_dir, domain_name)
        os.makedirs(domain_output_dir, exist_ok=True)
        
        output_file = os.path.join(domain_output_dir, f"embeddings_{os.path.basename(input_file)}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item, embedding in zip(texts_data, embeddings):
                if embedding:
                    result = {
                        "text": item["text"],
                        "embedding": embedding,
                        "metadata": item["metadata"],
                        "processed_at": datetime.now().isoformat()
                    }
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    total_success += 1
        
        total_processed += len(texts_data)
        
        print(f"  ✅ Saved {len([e for e in embeddings if e])}/{len(embeddings)} embeddings to {output_file}")
    
    print(f"\n{'='*60}")
    print(f"🎉 HOÀN THÀNH!")
    print(f"   Total processed: {total_processed}")
    print(f"   Total success: {total_success}")
    print(f"   Output directory: {output_dir}")

def main():
    # ========== CONFIGURATION ==========
    import json 
    with open("api-keys.json", "r", encoding="utf-8") as f:
        embed = json.load(f)[0]
    api = {
        "authorization": embed['authorization'],
        "tokenKey": embed['tokenKey'],
        "tokenId": embed['tokenId'] 
    }
    
    API_KEY = api["authorization"]
    TOKEN_ID = api["tokenId"]
    TOKEN_KEY = api["tokenKey"]
    
    # Tìm tất cả file JSONL trong thư mục filtered_data
    import glob
    INPUT_FILES = glob.glob("./filtered_data/*/*.jsonl")
    
    if not INPUT_FILES:
        print("❌ Không tìm thấy file JSONL nào trong ./filtered_data/*/*.jsonl")
        return
    
    print(f"📁 Found {len(INPUT_FILES)} JSONL files")
    
    # ========== RUN ==========
    create_embeddings(
        input_files=INPUT_FILES,
        output_dir="./embeddings",
        api_key=API_KEY,
        token_id=TOKEN_ID,
        token_key=TOKEN_KEY,
        max_texts_per_file=None  # Để None để xử lý tất cả, hoặc set số lượng giới hạn
    )

if __name__ == "__main__":
    main()