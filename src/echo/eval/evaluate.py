from pathlib import Path
import json
import gc
import torch

from echo.eval.bleu import BLEU
from echo.eval.rouge import ROUGE
from echo.eval.embedding_distance import EmbeddingDistance


def evaluate(predictions_path: str, save_metrics: bool = True) -> dict:
    """Compute metrics on generated predictions."""
    predictions_path = Path(predictions_path)

    # strip date out of path for output
    file_name = predictions_path.name
    date = file_name.strip("predictions_").strip(".json")
    output_path = predictions_path.parent / f"metrics_{date}.json"

    # check if output file already exists
    if output_path.exists():
        return json.load(open(output_path))
    
    # Load predictions
    with open(predictions_path) as f:
        data = json.load(f)
    
    preds = data["generations"]
    predictions = [item["prediction"] for item in preds]
    references = [item["reference"] for item in preds]

    # Clean predictions from thinking artifacts (Qwen)
    predictions = [pred.strip("<think>\n\n</think>") for pred in predictions]

    # Compute metrics
    bleu = BLEU()
    rouge = ROUGE()
    qwen3_emb_dist = EmbeddingDistance(model_name="Qwen/Qwen3-Embedding-8B")
    
    metrics = {
        "model_name": data["model_name"],
        "data_path": data["test_data_path"],
        "bleu": bleu(predictions, references),
        "rouge": rouge(predictions, references),
        "qwen3_embedding_distance": qwen3_emb_dist(predictions, references),
    }

    # Delete qwen3_emb_dist to free memory
    del qwen3_emb_dist
    torch.cuda.empty_cache()
    gc.collect()

    
    gemma_emb_dist = EmbeddingDistance(model_name="google/embeddinggemma-300m")
    lux_emb_dist = EmbeddingDistance()

    metrics.update({
        "gemma_embedding_distance": gemma_emb_dist(predictions, references),
        "luxembourgish_embedding_distance": lux_emb_dist(predictions, references),
    })
    
    # Save metrics
    if save_metrics:
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)


    return metrics