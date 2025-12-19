import json
import os
import re
from datasets import load_dataset
from tqdm import tqdm

def chunk_text(text, chunk_size=256, overlap=32):
    """
    Chia văn bản thành các chunk với overlap
    
    Args:
        text: Văn bản cần chia
        chunk_size: Kích thước mỗi chunk (số từ)
        overlap: Số từ overlap giữa các chunk
        
    Returns:
        List các chunk
    """
    # Tách văn bản thành từ
    words = text.split()
    
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        
        # Di chuyển với overlap, trừ khi đã đến cuối
        if end >= len(words):
            break
        start = end - overlap
    
    return chunks

def split_by_sentences(text, chunk_size=256, overlap=32):
    # Tách câu đơn giản (có thể cải thiện với NLP library)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return [text]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_length = len(sentence_words)
        
        # Nếu câu quá dài so với chunk_size, chia nhỏ câu đó
        if sentence_length > chunk_size:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Chia câu dài thành các chunk
            words = sentence.split()
            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[max(0, i - overlap):i + chunk_size]
                chunks.append(' '.join(chunk_words))
            continue
        
        # Thêm câu vào chunk hiện tại
        if current_length + sentence_length <= chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            # Lưu chunk hiện tại
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            # Bắt đầu chunk mới với overlap
            if overlap > 0 and current_chunk:
                # Lấy overlap từ chunk trước
                last_chunk_words = ' '.join(current_chunk).split()
                overlap_words = last_chunk_words[-overlap:] if len(last_chunk_words) >= overlap else last_chunk_words
                current_chunk = [' '.join(overlap_words), sentence]
                current_length = len(overlap_words) + sentence_length
            else:
                current_chunk = [sentence]
                current_length = sentence_length
    
    # Thêm chunk cuối cùng
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # Loại bỏ các chunk quá ngắn (dưới 10 từ)
    chunks = [chunk for chunk in chunks if len(chunk.split()) >= 10]
    
    return chunks

def download_and_filter_data():
    """Tải và lọc dữ liệu từ HuggingFace dataset"""
    
    # ========== CONFIGURATION ==========
    target_domains = [
        "Science",
        "Computers_and_Electronics",
        "Internet_and_Telecom",
        "Finance",
        "Law_and_Government",
        "Health",
        "Jobs_and_Education",
        "Travel_and_Transportation"
    ]
    
    # CHUNKING CONFIG
    CHUNK_CONFIGS = {
        "small": {"chunk_size": 256, "overlap": 32, "split_method": "sentences"},
        "medium": {"chunk_size": 512, "overlap": 64, "split_method": "sentences"},
        "large": {"chunk_size": 1024, "overlap": 128, "split_method": "sentences"},
    }
    
    target_domains_set = set(target_domains)
    
    # Giới hạn mỗi domain (trước khi chunking)
    MAX_SAMPLES_PER_DOMAIN = {
        "Science": 3000,
        "Computers_and_Electronics": 2500,
        "Business_and_Industrial": 1500,
        "Internet_and_Telecom": 3000,
        "Finance": 1500,
        "Law_and_Government": 1000,
        "Health": 1000,
        "Jobs_and_Education": 1000,
        "Travel_and_Transportation": 1500
    }
    
    BATCH_SIZE = 512  # Số samples mỗi file
    OUTPUT_DIR = "./filtered_data"
    
    # Chọn config chunking
    CHUNKING_CONFIG_NAME = "small"  # small/medium/large
    CHUNK_SIZE = CHUNK_CONFIGS[CHUNKING_CONFIG_NAME]["chunk_size"]
    OVERLAP = CHUNK_CONFIGS[CHUNKING_CONFIG_NAME]["overlap"]
    SPLIT_METHOD = CHUNK_CONFIGS[CHUNKING_CONFIG_NAME]["split_method"]
    
    # ========== CREATE OUTPUT DIRECTORY ==========
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Lưu config
    config = {
        "target_domains": target_domains,
        "max_samples_per_domain": MAX_SAMPLES_PER_DOMAIN,  # Dictionary với limit cho từng domain
        "chunking_config": CHUNKING_CONFIG_NAME,
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "split_method": SPLIT_METHOD
    }
    
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    # ========== INITIALIZE COUNTERS ==========
    domain_counter = {domain: 0 for domain in target_domains}
    domain_chunk_counter = {domain: 0 for domain in target_domains}  # Đếm chunks
    domain_data = {domain: [] for domain in target_domains}
    total_processed = 0
    total_chunks_created = 0
    
    # ========== LOAD DATASET ==========
    print("📥 Đang tải dataset từ HuggingFace...")
    ds = load_dataset("VTSNLP/vietnamese_curated_dataset", split="train", streaming=True)
    
    print(f"\n🎯 Target domains: {len(target_domains)}")
    print(f"📊 Max samples per domain:")
    for domain in target_domains:
        print(f"   • {domain}: {MAX_SAMPLES_PER_DOMAIN[domain]}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"\n🔪 Chunking Config: {CHUNKING_CONFIG_NAME}")
    print(f"   • Chunk size: {CHUNK_SIZE} từ")
    print(f"   • Overlap: {OVERLAP} từ")
    print(f"   • Split method: {SPLIT_METHOD}")
    
    # ========== PROCESS DATASET ==========
    print("\n🔄 Đang lọc, chunking và xử lý dữ liệu...")
    
    try:
        for item in tqdm(ds, desc="Processing", unit=" samples"):
            domain = item["domain"]
            
            # Chỉ lấy domains trong target và chưa đủ giới hạn
            if domain in target_domains_set and domain_counter[domain] < MAX_SAMPLES_PER_DOMAIN[domain]:
                domain_counter[domain] += 1
                total_processed += 1
                
                # Tính độ dài văn bản gốc
                original_length = len(item["text"].split())
                
                # Chunking văn bản
                if SPLIT_METHOD == "sentences":
                    chunks = split_by_sentences(item["text"], CHUNK_SIZE, OVERLAP)
                else:
                    chunks = chunk_text(item["text"], CHUNK_SIZE, OVERLAP)
                
                # Thêm từng chunk vào buffer
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_length = len(chunk.split())
                    domain_chunk_counter[domain] += 1
                    total_chunks_created += 1
                    
                    domain_data[domain].append({
                        "text": chunk,
                        "domain": domain,
                        "original_length": original_length,
                        "chunk_length": chunk_length,
                        "chunk_id": chunk_idx,
                        "total_chunks": len(chunks),
                        "original_id": item.get("id", total_processed),
                        "chunking_config": CHUNKING_CONFIG_NAME,
                        "chunk_size": CHUNK_SIZE,
                        "overlap": OVERLAP
                    })
                
                # Lưu batch khi đủ BATCH_SIZE chunks
                if len(domain_data[domain]) >= BATCH_SIZE:
                    save_batch(domain, domain_data[domain], OUTPUT_DIR)
                    domain_data[domain] = []  # Reset buffer
            
            # Dừng khi tất cả domains đã đủ
            if all(domain_counter[domain] >= MAX_SAMPLES_PER_DOMAIN[domain] for domain in target_domains):
                print("\n✅ Đã đủ samples cho tất cả domains!")
                break
                
    except KeyboardInterrupt:
        print("\n⏸️ Đã dừng bởi người dùng")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        raise
    
    # ========== SAVE REMAINING DATA ==========
    print("\n💾 Đang lưu các batches còn lại...")
    for domain in target_domains:
        if domain_data[domain]:
            save_batch(domain, domain_data[domain], OUTPUT_DIR)
    
    # ========== PRINT STATISTICS ==========
    print(f"\n{'='*60}")
    print("📊 THỐNG KÊ CHUNKING")
    print(f"{'='*60}")
    
    print(f"Tổng văn bản gốc đã xử lý: {total_processed:,}")
    print(f"Tổng chunks đã tạo: {total_chunks_created:,}")
    print(f"Tỷ lệ chunk/ văn bản: {total_chunks_created/total_processed:.2f}")
    
    print(f"\nPhân phối theo domain (văn bản gốc):")
    for domain in target_domains:
        count = domain_counter[domain]
        print(f"  {domain:30s}: {count:5d} samples")
    
    print(f"\nPhân phối chunks theo domain:")
    for domain in target_domains:
        count = domain_chunk_counter[domain]
        print(f"  {domain:30s}: {count:5d} chunks")
    
    # ========== SAVE FINAL STATISTICS ==========
    stats = {
        "total_original_processed": total_processed,
        "total_chunks_created": total_chunks_created,
        "avg_chunks_per_doc": total_chunks_created / total_processed if total_processed > 0 else 0,
        "max_per_domain": MAX_SAMPLES_PER_DOMAIN,  # Dictionary với limit riêng cho từng domain
        "target_domains": target_domains,
        "domain_distribution": domain_counter,
        "chunk_distribution": domain_chunk_counter,
        "chunking_config": {
            "name": CHUNKING_CONFIG_NAME,
            "chunk_size": CHUNK_SIZE,
            "overlap": OVERLAP,
            "split_method": SPLIT_METHOD
        }
    }
    
    stats_file = os.path.join(OUTPUT_DIR, "statistics.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Đã lưu thống kê vào: {stats_file}")
    print(f"✅ Đã lưu config vào: {os.path.join(OUTPUT_DIR, 'config.json')}")
    print(f"✅ Dữ liệu được lưu trong: {OUTPUT_DIR}/")

def save_batch(domain_name, batch_data, output_dir):
    """Lưu một batch dữ liệu"""
    
    # Tạo thư mục domain nếu chưa có
    domain_dir = os.path.join(output_dir, domain_name)
    os.makedirs(domain_dir, exist_ok=True)
    
    # Đếm số batch hiện có
    existing_files = [f for f in os.listdir(domain_dir) 
                     if f.startswith("batch_") and f.endswith('.jsonl')]
    batch_num = len(existing_files)
    
    # Tạo tên file
    filename = os.path.join(domain_dir, f"batch_{batch_num:03d}.jsonl")
    
    # Lưu dữ liệu
    with open(filename, "w", encoding="utf-8") as f:
        for item in batch_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # Tính thống kê batch
    avg_chunk_length = sum(item["chunk_length"] for item in batch_data) / len(batch_data)
    avg_original_length = sum(item["original_length"] for item in batch_data) / len(batch_data)
    
    print(f"  ✓ {domain_name}: Đã lưu {len(batch_data)} chunks vào {filename}")
    print(f"    • Avg chunk length: {avg_chunk_length:.1f} từ")
    print(f"    • Avg original length: {avg_original_length:.1f} từ")
    
    # Lưu thông tin batch
    summary = {
        "domain": domain_name,
        "batch_number": batch_num,
        "chunks": len(batch_data),
        "avg_chunk_length": avg_chunk_length,
        "avg_original_length": avg_original_length,
        "min_chunk_length": min(item["chunk_length"] for item in batch_data),
        "max_chunk_length": max(item["chunk_length"] for item in batch_data)
    }
    
    summary_file = os.path.join(domain_dir, f"batch_{batch_num:03d}_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

def check_and_create_filtered_data():
    """Kiểm tra và tạo dữ liệu nếu chưa có"""
    
    if not os.path.exists("./filtered_data"):
        print("📁 Thư mục filtered_data chưa tồn tại. Bắt đầu tạo dữ liệu...")
        download_and_filter_data()
    else:
        # Đếm số file JSONL trong filtered_data
        import glob
        jsonl_files = glob.glob("./filtered_data/*/*.jsonl")
        
        if len(jsonl_files) == 0:
            print("📁 Thư mục filtered_data tồn tại nhưng rỗng. Bắt đầu tạo dữ liệu...")
            download_and_filter_data()
        else:
            # Kiểm tra config hiện tại
            config_file = "./filtered_data/config.json"
            if os.path.exists(config_file):
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                print(f"✅ Thư mục filtered_data đã tồn tại với config:")
                print(f"   • Domains: {len(config['target_domains'])}")
                print(f"   • Chunk size: {config.get('chunk_size', 'N/A')}")
                print(f"   • Overlap: {config.get('overlap', 'N/A')}")
            else:
                print(f"✅ Thư mục filtered_data đã tồn tại với {len(jsonl_files)} files")
            
            print("   Bỏ qua bước tạo dữ liệu.")
            return True
    
    return False

if __name__ == "__main__":
    print("🚀 BẮT ĐẦU TẢI VÀ LỌC DỮ LIỆU VỚI CHUNKING")
    print("="*60)
    
    # Kiểm tra và tạo dữ liệu nếu cần
    data_exists = check_and_create_filtered_data()
    
    if data_exists:
        print("\n📊 Thống kê thư mục filtered_data:")
        import glob
        jsonl_files = glob.glob("./filtered_data/*/*.jsonl")
        domains = set([os.path.basename(os.path.dirname(f)) for f in jsonl_files])
        
        print(f"  • Số domains: {len(domains)}")
        print(f"  • Số files: {len(jsonl_files)}")
        print(f"  • Các domains: {', '.join(sorted(domains))}")
        
        # Đọc thống kê tổng
        stats_file = "./filtered_data/statistics.json"
        if os.path.exists(stats_file):
            with open(stats_file, "r", encoding="utf-8") as f:
                stats = json.load(f)
            print(f"  • Tổng văn bản gốc: {stats.get('total_original_processed', 'N/A')}")
            print(f"  • Tổng chunks: {stats.get('total_chunks_created', 'N/A')}")

import os
os._exit(0)