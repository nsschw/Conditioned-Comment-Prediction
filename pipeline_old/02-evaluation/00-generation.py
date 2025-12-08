from echo.eval.generate import generate
import torch
import gc

model_paths = [
    #"../../models/Llama-3.1-8B-Instruct-lux-history/20251121_114903/final",
    #"../../models/Ministral-8B-Instruct-2410-lux-history/20251121_135308/final",
    #"../../models/Qwen3-8B-lux-history/20251121_155617/final",
    #"meta-llama/Llama-3.1-8B-Instruct",
    #"mistralai/Ministral-8B-Instruct-2410",
    "Qwen/Qwen3-8B"
]
model_names = [
    #"Llama-3.1-8B-Instruct-lux-history",
    #"Ministral-8B-Instruct-2410-lux-history",
    #"Qwen3-8B-lux-history",
    #"Llama-3.1-8B-Instruct",
    #"Ministral-8B-Instruct-2410",
    "Qwen3-8B"
]

test_data = "../../data/processed/lux_test_history.json"

generation_kwargs = {
    "max_new_tokens": 200,
    "temperature": 0.75,
    }

for model_name, model_path in zip(model_names, model_paths):
    generate(model_path=model_path, model_name=model_name, output_path="../../outputs", test_data_path=test_data, disable_reasoning=True, **generation_kwargs)
    
    # Clear CUDA memory between models
    torch.cuda.empty_cache()
    gc.collect()