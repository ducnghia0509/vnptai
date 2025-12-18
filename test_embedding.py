#!/usr/bin/env python3
"""
Test script for VNPT Embedding API
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from embedder import VNPTEmbedder

def main():
    print("Testing VNPT Embedding API...")
    
    # Initialize embedder
    embedder = VNPTEmbedder('api-keys.json')
    
    # Test single encoding
    print("\n1. Testing single text encoding...")
    test_text = "Thủ đô của Việt Nam là Hà Nội"
    embedding = embedder.encode(test_text)
    
    if embedding is not None:
        print(f"✓ Single encoding successful")
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  Embedding dtype: {embedding.dtype}")
        print(f"  Embedding sample (first 5): {embedding[:5]}")
    else:
        print("✗ Single encoding failed")
    
    # Test batch encoding
    print("\n2. Testing batch encoding...")
    test_texts = [
        "Hà Nội là thủ đô của Việt Nam",
        "TP Hồ Chí Minh là thành phố lớn nhất",
        "Đà Nẵng là thành phố đáng sống"
    ]
    
    embeddings = embedder.encode_batch(test_texts, batch_size=2)
    
    if embeddings and len(embeddings) == len(test_texts):
        print(f"✓ Batch encoding successful")
        print(f"  Number of embeddings: {len(embeddings)}")
        for i, emb in enumerate(embeddings):
            print(f"  Text {i+1}: {test_texts[i][:30]}... -> shape: {emb.shape}")
    else:
        print("✗ Batch encoding failed")
    
    # Show usage
    print("\n3. API Usage:")
    usage = embedder.get_usage()
    print(f"  Used: {usage['used']}/{usage['total']}")
    print(f"  Remaining: {usage['remaining']}")
    print(f"  Percentage: {usage['percentage']:.1f}%")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    main()