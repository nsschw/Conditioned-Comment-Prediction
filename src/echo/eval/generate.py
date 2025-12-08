from pathlib import Path
import json
import datetime

from echo.eval.model import Model

def enforce_disabled_reasoning(messages):
    for message in messages:
        if message["role"] == "user":
            message["content"] += " /no_think"
    return messages
    

def generate(model_path: str, model_name: str, output_path: str, test_data_path: str, batch_size: int = 3, disable_reasoning: bool = False, **kwargs):
    """Generate predictions for test data."""
    # Setup output path
    output_path = Path(output_path) / model_name / f"predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load model and data
    model = Model(model_path, batch_size=batch_size)
    with open(test_data_path) as f:
        test_data = json.load(f)
    
    # Generate
    prompts = [case["messages"] for case in test_data]

    # Disable Reasoning
    if disable_reasoning:
        prompts = [enforce_disabled_reasoning(prompt) for prompt in prompts]


    preds = model.generate(prompts, **kwargs)

    # Save results
    results = []
    for case, pred in zip(test_data, preds):
        results.append({
            "prompt": case["messages"],
            "prediction": pred,
            "reference": case["human_comment"]
        })
    
    # merge kwargs and generations
    run = {"generation_args": kwargs, "generations": results, "model_name": model_name, "model_path": model_path, "test_data_path": test_data_path}

    with open(output_path, "w") as f:
        json.dump(run, f, indent=2)
    
    print(f"Saved {len(preds)} predictions to {output_path}")