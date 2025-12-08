from echo.eval.generate import generate
import torch
import gc

generation_kwargs = {
    "max_new_tokens": 500,
    "temperature": 0.75,
}

model_path = "../../models/Llama-3.1-8B-Instruct-mix-bio-history/20251128_115753/final"
model_name = "Llama-3.1-8B-Instruct-mix-bio-history"

test_data_paths = [
    "../../data/processed/eng_test_bio_history.json",
    "../../data/processed/ger_test_bio_history.json",
    "../../data/processed/lux_test_bio_history.json",
]
             

for test_data_path in test_data_paths:
    generate(model_path=model_path, model_name=model_name, output_path="../../outputs", test_data_path=test_data_path, disable_reasoning=False, **generation_kwargs)
    torch.cuda.empty_cache()
    gc.collect()