from echo.eval.generate import generate
import torch
import gc

model_paths = []
model_names = []

test_data = "../../data/processed/lux_test_history.json"

deterministic_generation_kwargs = {
    "max_new_tokens": 200,
    "do_sample": False
    }

for model_name, model_path in zip(model_names, model_paths):
    generate(model_path=model_path, model_name=model_name, output_path="../../outputs", test_data_path=test_data, **deterministic_generation_kwargs)
    
    # Clear CUDA memory between models
    torch.cuda.empty_cache()
    gc.collect()