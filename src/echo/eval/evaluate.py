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
        #return rerun_bleu_and_rouge(output_path)
    
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
        "bleu": bleu(target=predictions, source=references),
        "rouge": rouge(target=predictions, source=references),
        "qwen3_embedding_distance": qwen3_emb_dist(target=predictions, source=references),
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



def rerun_bleu_and_rouge(metrics_path: Path) -> dict:
    """Recalculate BLEU and ROUGE for existing metrics file."""
    metrics_path = Path(metrics_path)
    
    # Reconstruct predictions filename from metrics filename
    # metrics_{date}.json -> predictions_{date}.json
    file_name = metrics_path.name
    date = file_name.replace("metrics_", "").replace(".json", "")
    predictions_path = metrics_path.parent / f"predictions_{date}.json"
    
    # Load original predictions to get text
    with open(predictions_path) as f:
        data = json.load(f)
        
    preds = data["generations"]
    predictions = [item["prediction"] for item in preds]
    references = [item["reference"] for item in preds]
    
    # Clean predictions from thinking artifacts (must match original logic)
    predictions = [pred.strip("<think>\n\n</think>") for pred in predictions]
    
    # Load existing metrics to preserve embeddings
    with open(metrics_path) as f:
        metrics = json.load(f)
        
    # Recompute text metrics
    bleu = BLEU()
    rouge = ROUGE()
    
    metrics["bleu"] = bleu(target=predictions, source=references)
    metrics["rouge"] = rouge(target=predictions, source=references)
    
    # Save updated metrics
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Updated BLEU/ROUGE for {metrics_path.name}")
    
    return metrics