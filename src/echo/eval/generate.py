from pathlib import Path
import json
import torch
from tqdm import tqdm

from echo.eval.model import Model


def generate(model_path: str, test_data_path: str):
    """Generate predictions for test data."""
    # Setup output path
    model_name = Path(model_path).parent.parent.name
    output_path = Path("../../outputs") / model_name / "predictions.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load model and data
    model = Model(model_path)
    with open(test_data_path) as f:
        test_data = json.load(f)
    
    # Generate
    prompts = [case["messages"] for case in test_data]
    preds = model.generate(prompts)

    # Save results
    results = []
    for case, pred in zip(test_data, preds):
        results.append({
            "prompt": case["messages"],
            "prediction": pred,
            "reference": case["human_comment"]
        })
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved {len(preds)} predictions to {output_path}")