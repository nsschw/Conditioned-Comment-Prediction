from echo.eval.generate import generate
import torch
import gc

models = []

test_data = "../../data/processed/lux_test_history.json"

for model_path in models:
    generate(model_path, test_data)
    
    # Clear CUDA memory between models
    torch.cuda.empty_cache()
    gc.collect()