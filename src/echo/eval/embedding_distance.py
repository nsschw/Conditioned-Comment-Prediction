from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import torch
import numpy as np

class EmbeddingDistance:
    """Enhanced embedding distance calculator using sentence-transformers."""
    
    def __init__(self, model_name="fredxlpy/LuxEmbedder"):
        """
        Initialize with a sentence transformer model.
        
        Args:
            model_name: Model identifier for SentenceTransformer
        """
        self.model = SentenceTransformer(model_name)
        
        
    def __call__(self, source: list[str], target: list[str]) -> dict:
        """
        Calculate semantic distance between source and target texts.
        
        Args:
            source: List of source/reference texts
            target: List of target/generated texts
            
        Returns:
            Dictionary with mean and standard deviation of distances
        """
        # Process in batches for memory efficiency
        batch_size = 256
        distances = []
        
        for i in range(0, len(source), batch_size):
            src_batch = source[i:i+batch_size]
            tgt_batch = target[i:i+batch_size]
            
            # Encode texts to embeddings
            src_embeddings = self.model.encode(src_batch)
            tgt_embeddings = self.model.encode(tgt_batch)
            
            # Calculate cosine similarity
            cos_similarities = cos_sim(src_embeddings, tgt_embeddings)
            cos_distance = 1 - torch.diagonal(cos_similarities).cpu().numpy()
            distances.extend(cos_distance)

        
        mean = float(np.mean(distances))
        std = float(np.std(distances))
        return {
            "mean_distance": mean,
            "std_distance": std
        }