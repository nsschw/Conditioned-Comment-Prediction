from pathlib import Path
import json

from echo.eval.bleu import BLEU
from echo.eval.embedding_distance import EmbeddingDistance


def evaluate(predictions_path: str):
    """Compute metrics on generated predictions."""
    predictions_path = Path(predictions_path)
    output_path = predictions_path.parent / "metrics.json"
    
    # Load predictions
    with open(predictions_path) as f:
        data = json.load(f)
    
    predictions = [item["prediction"] for item in data]
    references = [item["reference"] for item in data]
    
    # Compute metrics
    bleu = BLEU()
    emb_dist = EmbeddingDistance()
    
    metrics = {
        "bleu": bleu(predictions, references, return_mean=True),
        "embedding_distance": emb_dist(predictions, references, return_mean=True),
    }
    
    # Save metrics
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"{predictions_path.parent.name}:")
    print(f"  BLEU: {metrics['bleu']["bleu"]:.4f}")
    print(f"  Embedding Distance: {metrics['embedding_distance']:.4f}")
    
    return metrics