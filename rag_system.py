import json
import numpy as np
import faiss
import os

class RAGSystem:
    """RAG system with FAISS index and VNPT embedder"""
    
    def __init__(self, api_keys_file='api-keys.json'):
        self.index = None
        self.metadata = []
        self.is_loaded = False
        
        # Initialize embedder
        try:
            # Import locally to avoid circular imports
            from embedder import VNPTEmbedder
            self.embedder = VNPTEmbedder(api_keys_file)
            print("✓ VNPT Embedder initialized")
        except ImportError as e:
            print(f"✗ Failed to import VNPTEmbedder: {e}")
            print("⚠️ Using random embeddings as fallback")
            self.embedder = None
    
    def load_index(self, index_path="faiss_index.bin", metadata_path="metadata.json"):
        """Load FAISS index with error handling"""
        try:
            if not os.path.exists(index_path):
                print(f"✗ Index file not found: {index_path}")
                return False
            
            if not os.path.exists(metadata_path):
                print(f"✗ Metadata file not found: {metadata_path}")
                return False
            
            print(f"Loading FAISS index from {index_path}...")
            self.index = faiss.read_index(index_path)
            
            print(f"Loading metadata from {metadata_path}...")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            self.is_loaded = True
            print(f"✓ FAISS index loaded: {self.index.ntotal} vectors")
            
            # Show embedding API usage
            if self.embedder:
                usage = self.embedder.get_usage()
                print(f"✓ Embedding API: {usage['used']}/{usage['total']} requests used")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading FAISS index: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def retrieve(self, query_embedding, k=3, threshold=1.5):
        """Retrieve top-k relevant documents"""
        if not self.is_loaded:
            print("⚠️ Index not loaded")
            return []
        
        # Reshape query if needed
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        try:
            # Search
            distances, indices = self.index.search(query_embedding, k)
            
            # Format results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx != -1 and idx < len(self.metadata) and dist < threshold:
                    results.append({
                        'text': self.metadata[idx]['text'],
                        'score': float(dist),
                        'domain': self.metadata[idx].get('domain', 'unknown'),
                        'source': self.metadata[idx].get('source_file', 'unknown')
                    })
            
            return results
            
        except Exception as e:
            print(f"✗ Error during retrieval: {e}")
            return []