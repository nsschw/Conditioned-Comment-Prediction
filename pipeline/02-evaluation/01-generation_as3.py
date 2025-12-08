from echo.eval.generate import generate
import torch
import gc

generation_kwargs = {
    "max_new_tokens": 500,
    "temperature": 0.75,
}

model_paths = [
    "../../models/Qwen3-0.6B-eng-bio-history/20251128_115038/final",
    "../../models/Qwen3-1.7B-eng-bio-history/20251128_120258/final",
    "../../models/Qwen3-4B-eng-bio-history/20251128_121941/final",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
]
model_names = [
    "Qwen3-0.6B-eng-bio-history",
    "Qwen3-1.7B-eng-bio-history",
    "Qwen3-4B-eng-bio-history",
    "Qwen3-0.6B",
    "Qwen3-1.7B",
    "Qwen3-4B",
]

test_data = "../../data/processed/eng_test_bio_history.json"

for model_name, model_path in zip(model_names, model_paths):
    generate(model_path=model_path, model_name=model_name, output_path="../../outputs", test_data_path=test_data, disable_reasoning=True, **generation_kwargs)
    torch.cuda.empty_cache()
    gc.collect()
