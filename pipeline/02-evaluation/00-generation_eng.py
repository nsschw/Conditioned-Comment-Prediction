from echo.eval.generate import generate
import torch
import gc

generation_kwargs = {
    "max_new_tokens": 500,
    "temperature": 0.75,
}

model_paths = [
    "../../models/Llama-3.1-8B-Instruct-eng-bio-history/20251121_114736/final",
    "../../models/Ministral-8B-Instruct-2410-eng-bio-history/20251121_143235/final",
    "../../models/Qwen3-8B-eng-bio-history/20251121_162359/final",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Ministral-8B-Instruct-2410",
    "Qwen/Qwen3-8B"
]
model_names = [
    "Llama-3.1-8B-Instruct-eng-bio-history",
    "Ministral-8B-Instruct-2410-eng-bio-history",
    "Qwen3-8B-eng-bio-history",
    "Llama-3.1-8B-Instruct",
    "Ministral-8B-Instruct-2410",
    "Qwen3-8B"
]

test_data = "../../data/processed/eng_test_bio_history.json"

for model_name, model_path in zip(model_names, model_paths):
    if model_name == "Qwen3-8B":
        generate(model_path=model_path, model_name=model_name, output_path="../../outputs", test_data_path=test_data, disable_reasoning=True, **generation_kwargs)
    else:
        generate(model_path=model_path, model_name=model_name, output_path="../../outputs", test_data_path=test_data, disable_reasoning=False, **generation_kwargs)
    
    torch.cuda.empty_cache()
    gc.collect()