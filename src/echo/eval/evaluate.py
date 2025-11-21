from pathlib import Path
import json

from echo.eval.bleu import BLEU
from echo.eval.rouge import ROUGE
from echo.eval.embedding_distance import EmbeddingDistance


def evaluate(predictions_path: str):
    """Compute metrics on generated predictions."""
    predictions_path = Path(predictions_path)
    output_path = predictions_path.parent / "metrics.json"
    
    # Load predictions
    with open(predictions_path) as f:
        data = json.load(f)
    
    preds = data["generations"]
    predictions = [item["prediction"] for item in preds]
    references = [item["reference"] for item in preds]
    
    # Compute metrics
    bleu = BLEU()
    rouge = ROUGE()
    emb_dist = EmbeddingDistance()
    
    metrics = {
        "bleu": bleu(predictions, references),
        "rouge": rouge(predictions, references),
        "embedding_distance": emb_dist(predictions, references),
    }
    
    # Save metrics
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)


    return metrics