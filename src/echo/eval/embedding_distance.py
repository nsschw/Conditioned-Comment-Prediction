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
        
        
    def __call__(self, source: list[str], target: list[str], return_mean=True):
        """
        Calculate semantic distance between source and target texts.
        
        Args:
            source: List of source/reference texts
            target: List of target/generated texts
            return_mean: Return mean distance (True) or all distances (False)
            
        Returns:
            Mean distance or list of individual distances
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

        
        if return_mean:
            return float(np.mean(distances))
        else:
            return [float(d) for d in distances]