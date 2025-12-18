import json
import numpy as np
import faiss
import glob
import os

print("Building FAISS index from embeddings...")

# Find all embedding files
jsonl_files = glob.glob("embeddings/embeddings_batch_*.jsonl")
if not jsonl_files:
    # Try current directory
    jsonl_files = glob.glob("embeddings_batch_*.jsonl")

if not jsonl_files:
    print("❌ No embedding files found!")
    print("Please ensure embedding files are in 'embeddings/' directory or current directory.")
    exit(1)

print(f"Found {len(jsonl_files)} embedding files")

all_embeddings = []
all_metadata = []

# Load embeddings
for file_path in sorted(jsonl_files):
    print(f"Loading {os.path.basename(file_path)}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                
                # Get embedding
                embedding = np.array(data['embedding'], dtype=np.float32)
                all_embeddings.append(embedding)
                
                # Store metadata
                metadata = {
                    'text': data['text'],
                    'source_file': data.get('source_file', 'unknown'),
                    'domain': data.get('domain', 'unknown'),
                    'original_index': data.get('original_index', 0)
                }
                all_metadata.append(metadata)
                
            except Exception as e:
                print(f"  Skipping invalid line: {e}")
                continue

if not all_embeddings:
    print("❌ No embeddings loaded!")
    exit(1)

# Create numpy array
embeddings_array = np.array(all_embeddings)
print(f"✓ Loaded {len(embeddings_array)} embeddings")
print(f"  Dimension: {embeddings_array.shape[1]}")

# Create FAISS index
d = embeddings_array.shape[1]
index = faiss.IndexFlatL2(d)  # Simple L2 distance
index.add(embeddings_array)

# Save index
faiss.write_index(index, "faiss_index.bin")

# Save metadata
with open("metadata.json", 'w', encoding='utf-8') as f:
    json.dump(all_metadata, f, ensure_ascii=False, indent=2)

print(f"\n✅ FAISS index built successfully!")
print(f"   Index file: faiss_index.bin ({index.ntotal} vectors)")
print(f"   Metadata: metadata.json")