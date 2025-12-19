import os
import json
import csv
import time
import requests
from datetime import datetime
from tqdm import tqdm

# Import our modules
try:
    from embedder import VNPTEmbedder
    from rag_system import RAGSystem
    print("✓ Successfully imported RAG modules")
except ImportError as e:
    print(f"✗ Failed to import modules: {e}")
    print("Please ensure embedder.py and rag_system.py are in the same directory")
    exit(1)

# =========================================================
# CONFIGURATION
# =========================================================
CONFIG = {
    'api_keys_file': 'api-keys.json',
    'data_path': '/code/private_test.json',
    'output_path': 'submission.csv',
    'use_rag': True,
    'max_requests_per_hour': 60,
    'max_requests_per_day': 1000,
    'request_delay': 1.0,
    'rag_top_k': 3,
    'max_context_length': 1024,
    'max_retries': 3,
    'timeout': 30
}

# =========================================================
# LOAD API 
# =========================================================
print("Loading API keys...")
with open(CONFIG['api_keys_file'], 'r', encoding='utf-8') as f:
    api_keys = json.load(f)

# Get VNPT AI API keys (the small model)
llm_small = api_keys[1]  # According to your working code
AUTHORIZATION = llm_small["authorization"]
TOKEN_KEY = llm_small["tokenKey"]
TOKEN_ID = llm_small["tokenId"]
API_URL = "https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-small"

print("✓ VNPT AI API keys loaded")

# =========================================================
# INITIALIZE RAG SYSTEM
# =========================================================
rag_system = None
if CONFIG['use_rag']:
    print("\n" + "="*50)
    print("Initializing RAG system...")
    print("="*50)
    
    try:
        # Check if FAISS index exists
        if not os.path.exists("faiss_index.bin") or not os.path.exists("metadata.json"):
            print("⚠️ FAISS index not found. RAG will be disabled.")
            print("   Run 'python faiss_index.py' to build the index first.")
            CONFIG['use_rag'] = False
        else:
            rag_system = RAGSystem(CONFIG['api_keys_file'])
            
            # Test embedding connection first
            if rag_system.embedder:
                print("\nTesting embedding API connection...")
                if not rag_system.embedder.test_connection():
                    print("\n❌ Embedding API test failed.")
                    choice = input("Continue without RAG? (y/n): ").strip().lower()
                    if choice == 'y':
                        CONFIG['use_rag'] = False
                    else:
                        exit(1)
                else:
                    print("✓ Embedding API is working")
            
            if CONFIG['use_rag']:
                if rag_system.load_index():
                    print("\n✅ RAG system ready")
                    print(f"   Documents in index: {rag_system.index.ntotal}")
                    
                    # Show initial usage
                    if rag_system.embedder:
                        usage = rag_system.embedder.get_usage()
                        print(f"   Embedding API usage: {usage['used']}/{usage['total']}")
                else:
                    print("❌ Failed to load RAG index")
                    CONFIG['use_rag'] = False
                    
    except Exception as e:
        print(f"\n✗ RAG initialization failed: {e}")
        CONFIG['use_rag'] = False
else:
    print("ℹ️ RAG disabled by configuration")

print("\n" + "="*50)
print(f"Final Configuration:")
print(f"  RAG Enabled: {CONFIG['use_rag']}")
print(f"  VNPT AI Model: vnptai_hackathon_small")
print(f"  Max Requests: {CONFIG['max_requests_per_hour']}/hour, {CONFIG['max_requests_per_day']}/day")
print(f"  Output File: {CONFIG['output_path']}")
print("="*50 + "\n")

# =========================================================
# LOAD DATASET
# =========================================================
print(f"Loading dataset from {CONFIG['data_path']}...")
try:
    with open(CONFIG['data_path'], 'r', encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"✓ Loaded {len(dataset)} questions")
except Exception as e:
    print(f"✗ Failed to load dataset: {e}")
    exit(1)

# =========================================================
# API CALL FUNCTION WITH RETRY
# =========================================================
def call_api_with_retry(payload, max_retries=3):
    """Call VNPT AI API with retry logic"""
    headers = {
        "Authorization": AUTHORIZATION,
        "Token-id": TOKEN_ID,
        "Token-key": TOKEN_KEY,
        "Content-Type": "application/json",
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                API_URL, 
                headers=headers, 
                json=payload, 
                timeout=CONFIG['timeout']
            )

            if response.status_code == 200:
                return response.json()

            # Handle rate limiting
            if response.status_code == 429:
                wait = 2 ** attempt
                print(f"   ⚠️ Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
                
            print(f"   ❌ API error {response.status_code}: {response.text}")
            
            # For 401 errors, skip retry as it's an auth issue
            if response.status_code == 401:
                print("   ❌ Authentication failed. Check your API keys.")
                return None
                
            time.sleep(1)  # Small delay before retry

        except Exception as e:
            print(f"   ⚠️ Request error: {e}")
            time.sleep(1)
    
    return None

# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict_answer(question, choices):
    """Predict answer using RAG (if available) and VNPT AI"""
    
    # Step 1: Get RAG context
    context = ""
    if CONFIG['use_rag'] and rag_system and rag_system.embedder:
        try:
            print("   🔍 Retrieving relevant information from knowledge base...")
            
            # Generate embedding for the question
            query_embedding = rag_system.embedder.encode(question)
            
            # Retrieve similar documents
            results = rag_system.retrieve(query_embedding, k=CONFIG['rag_top_k'])
            
            if results:
                # Format context
                context_lines = []
                for i, res in enumerate(results, 1):
                    text = res['text']
                    score = res.get('score', 0)
                    
                    context_lines.append(f"[Source {i}, relevance: {1-score:.2f}] {text}")
                
                context = "\n".join(context_lines)
                print(f"   📚 Found {len(results)} relevant documents")
                
                # Show embedding usage
                if rag_system.embedder:
                    usage = rag_system.embedder.get_usage()
                    print(f"   📊 Embedding API usage: {usage['used']}/{usage['total']}")
            else:
                print("   ℹ️ No relevant documents found in knowledge base")
        
        except Exception as e:
            print(f"   ⚠️ RAG error: {e}")
    
    # Step 2: Prepare payload for VNPT AI
    if context:
        system_content = f"""
BẠN LÀ TRỢ LÝ AI TRẢ LỜI CÂU HỎI TRẮC NGHIỆM TIẾNG VIỆT.

THÔNG TIN THAM KHẢO:
{context}

NGUYÊN TẮC BẮT BUỘC:
- Chỉ sử dụng thông tin có trong THÔNG TIN THAM KHẢO.
- Không suy đoán, không bịa.
- Nếu ở câu hỏi nói về nội dung nhạy cảm như chiến tranh, chính sách chính trị, an toàn, quốc phòng, tranh chấp biển, lãnh thổ, chiến tranh, chủng tộc,...thì bạn không được trả lời và chọn đáp án gần nghĩa nhất với Không thể trả lời câu hỏi này
- Nếu câu hỏi là kiến thức tính toán phức tạp, bạn cần làm từng bước trong *suy nghĩ* rồi mới được đưa ra câu trả lời 
CÁCH XỬ LÝ:
1. Nếu có thể xác định ĐÁP ÁN ĐÚNG từ thông tin → chọn phương án tương ứng.
2. Nếu KHÔNG THỂ xác định đáp án vì:
   - Thiếu dữ liệu
   - Các phương án đều không được đề cập
   - Câu hỏi mơ hồ / vượt phạm vi
→ Tìm đáp án gần nghĩa với: Không thể trả lời câu hỏi này

YÊU CẦU ĐẦU RA:
- Nếu trả lời được → chỉ 1 ký tự: A, B, C, D, ...
- Tuyệt đối không thêm văn bản gì ngoài ký tự chữ cái viết hoa đã chọn.
"""

    else:
        system_content = """
BẠN LÀ HỆ THỐNG TRẢ LỜI CÂU HỎI TRẮC NGHIỆM.

NGUYÊN TẮC:
- Dựa trên kiến thức phổ thông và logic.
- Không bịa đặt thông tin chuyên ngành nếu không chắc chắn.

CÁCH LÀM:
1. Đọc kỹ câu hỏi và tất cả các phương án.
2. Nếu là câu hỏi logic / toán học, hãy suy nghĩ từng bước một CÁCH NỘI BỘ.
3. Chọn phương án đúng hoặc hợp lý nhất.

YÊU CẦU ĐẦU RA:
- CHỈ MỘT KÝ TỰ DUY NHẤT: A, B, C, D, E, ...
- KHÔNG giải thích.
- KHÔNG thêm bất kỳ văn bản nào khác.

VÍ DỤ ĐẦU RA HỢP LỆ:
C
"""
    
    user_content = f"Câu hỏi: {question}\n\nLựa chọn:\n{choices}"
    
    payload = {
        'model': "vnptai_hackathon_small",
        'messages': [
            {'role': 'system', 'content': system_content},
            {'role': 'user', 'content': user_content}
        ],
        'temperature': 0.1,
        'top_p': 0.9,
        'top_k': 20,
        'n': 1,
        'max_completion_tokens': 1,
    }
    
    # Step 3: Call API
    try:
        result = call_api_with_retry(payload, max_retries=CONFIG['max_retries'])
        
        if result and "choices" in result:
            answer = result["choices"][0]["message"]["content"].strip()
            
            # Debug: show raw response
            print(f"   🤖 Raw response: '{answer}'")
            
            # Clean the answer
            if answer:
                # Take first character and convert to uppercase
                clean_answer = answer[0].upper() if answer[0].isalpha() else ""
                
                # if clean_answer in ['A', 'B', 'C', 'D']:
                if clean_answer.isalpha() and clean_answer.isupper():
                    return clean_answer
                elif answer[0].isdigit():
                    num = int(answer[0])
                    if 1 <= num:
                        return chr(65 + num)  # 0->A, 1->B, etc. cần để ý
    
    except Exception as e:
        print(f"   ❌ API error: {e}")
        # Wait a bit longer on error
        time.sleep(2)
    
    # Fallback - return first option
    return "A"

# =========================================================
# LOAD EXISTING PROGRESS
# =========================================================
def load_existing_progress():
    """Load already processed questions"""
    processed = set()
    
    if os.path.exists(CONFIG['output_path']):
        with open(CONFIG['output_path'], 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)  # Skip header
                for row in reader:
                    if row and len(row) >= 1:
                        processed.add(row[0])
            except StopIteration:
                pass
    
    return processed, len(processed)

# =========================================================
# QUOTA MANAGEMENT
# =========================================================
class QuotaManager:
    def __init__(self, max_per_hour=60, max_per_day=1000):
        self.max_per_hour = max_per_hour
        self.max_per_day = max_per_day
        self.hourly_requests = []
        self.daily_requests = []
        self.total_requests = 0
        self.current_hour = datetime.now().hour
        
    def can_make_request(self):
        """Check if we can make a request"""
        now = time.time()
        
        # Reset if hour changed
        now_hour = datetime.now().hour
        if now_hour != self.current_hour:
            self.hourly_requests = []
            self.current_hour = now_hour
        
        # Remove old requests
        self.hourly_requests = [t for t in self.hourly_requests if now - t < 3600]
        self.daily_requests = [t for t in self.daily_requests if now - t < 86400]
        
        # Check limits
        if len(self.daily_requests) >= self.max_per_day:
            return False
        
        if len(self.hourly_requests) >= self.max_per_hour:
            return False
        
        return True
    
    def record_request(self):
        """Record that a request was made"""
        now = time.time()
        self.hourly_requests.append(now)
        self.daily_requests.append(now)
        self.total_requests += 1
    
    def wait_if_needed(self):
        """Wait if quota limits are reached"""
        if not self.can_make_request():
            # Check which limit is reached
            now = time.time()
            self.hourly_requests = [t for t in self.hourly_requests if now - t < 3600]
            self.daily_requests = [t for t in self.daily_requests if now - t < 86400]
            
            if len(self.hourly_requests) >= self.max_per_hour:
                # Calculate wait time for hourly limit
                oldest = min(self.hourly_requests)
                wait_time = 3600 - (now - oldest) + 5
                print(f"   ⏳ Hourly limit reached ({self.max_per_hour}/hour), waiting {wait_time/60:.1f} minutes...")
                time.sleep(wait_time)
            
            if len(self.daily_requests) >= self.max_per_day:
                print("❌ Daily limit reached (1000/day). Stopping.")
                return False
        
        return True
    
    def get_stats(self):
        """Get current statistics"""
        now = time.time()
        hourly_count = len([t for t in self.hourly_requests if now - t < 3600])
        daily_count = len([t for t in self.daily_requests if now - t < 86400])
        
        return {
            'hourly': hourly_count,
            'max_hourly': self.max_per_hour,
            'daily': daily_count,
            'max_daily': self.max_per_day,
            'total': self.total_requests
        }

quota_manager = QuotaManager(
    max_per_hour=CONFIG['max_requests_per_hour'],
    max_per_day=CONFIG['max_requests_per_day']
)

# =========================================================
# MAIN EXECUTION
# =========================================================
def main():
    print("\n" + "="*70)
    print("🚀 VNPT RAG + VNPT AI System")
    print("="*70)
    
    # Load progress
    processed_qids, processed_count = load_existing_progress()
    total = len(dataset)

    
    # Check if we've already processed everything
    if processed_count >= total:
        return
    
    # Open output files
    file_mode = 'a' if processed_count > 0 else 'w'
    with open(CONFIG['output_path'], file_mode, newline='', encoding='utf-8') as f, \
         open('submission_time.csv', file_mode, newline='', encoding='utf-8') as f_time:
        writer = csv.writer(f)
        writer_time = csv.writer(f_time)

        # Write header if new file
        if file_mode == 'w':
            writer.writerow(['qid', 'answer'])
            writer_time.writerow(['qid', 'answer', 'time'])

        # Process questions
        for idx, item in enumerate(tqdm(dataset, desc="Processing", unit="q")):
            qid = item['qid']

            # Skip if already processed
            if qid in processed_qids:
                continue

            question = item['question']
            choices = "\n".join(item['choices'])

            # Display current question
            print(f"\n[{idx+1}/{total}] QID: {qid}")
            if len(question) > 80:
                print(f"   ❓ {question[:80]}...")
            else:
                print(f"   ❓ {question}")

            # Check quota before making request
            if not quota_manager.wait_if_needed():
                break

            # Get prediction and measure time
            try:
                start_time = time.time()
                answer = predict_answer(question, choices)
                elapsed = time.time() - start_time

                # Record the API request
                quota_manager.record_request()

                # Write result to both files
                writer.writerow([qid, answer])
                writer_time.writerow([qid, answer, f"{elapsed:.4f}"])
                f.flush()  # Ensure immediate write
                f_time.flush()

                # Add to processed set
                processed_qids.add(qid)

            except Exception as e:
                print(f"   ❌ Error: {e}")
                # Write fallback answer
                writer.writerow([qid, 'A'])
                writer_time.writerow([qid, 'A', '0.0000'])
                f.flush()
                f_time.flush()
                processed_qids.add(qid)

            # Show quota stats
            stats = quota_manager.get_stats()
            print(f"   📊 Quota: {stats['hourly']}/{stats['max_hourly']} per hour, "
                  f"{stats['daily']}/{stats['max_daily']} per day")

            # Delay between requests
            time.sleep(CONFIG['request_delay'])
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 PROCESSING COMPLETE!")
    print("="*70)
    
    final_stats = quota_manager.get_stats()
    print(f"📊 Statistics:")
    print(f"   Total requests made: {final_stats['total']}")
    print(f"   Questions processed: {len(processed_qids)}/{total}")
    print(f"   Hourly usage: {final_stats['hourly']}/{final_stats['max_hourly']}")
    print(f"   Daily usage: {final_stats['daily']}/{final_stats['max_daily']}")
    
    if CONFIG['use_rag'] and rag_system and rag_system.embedder:
        embed_usage = rag_system.embedder.get_usage()
        print(f"   Embedding API usage: {embed_usage['used']}/{embed_usage['total']}")
    
    print(f"📁 Results saved to: {CONFIG['output_path']}")
    print("="*70)

# =========================================================
# RUN MAIN FUNCTION
# =========================================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user. Progress has been saved.")
        print(f"📁 You can resume by running the script again.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()