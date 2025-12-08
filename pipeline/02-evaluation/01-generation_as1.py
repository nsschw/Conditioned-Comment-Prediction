from echo.eval.generate import generate
import torch
import gc

generation_kwargs = {
    "max_new_tokens": 500,
    "temperature": 0.75,
}


model_paths = [
    "../../models/Llama-3.1-8B-Instruct-eng-bio/20251125_093656/final",
    "meta-llama/Llama-3.1-8B-Instruct",
]
model_names = [
    "Llama-3.1-8B-Instruct-eng-bio",
    "Llama-3.1-8B-Instruct",
]
test_data = "../../data/processed/eng_test_bio.json"

for model_name, model_path in zip(model_names, model_paths):
    generate(model_path=model_path, model_name=model_name, output_path="../../outputs", test_data_path=test_data, disable_reasoning=False, **generation_kwargs)
    torch.cuda.empty_cache()
    gc.collect()


model_paths = [
    "../../models/Llama-3.1-8B-Instruct-eng-history/20251125_083546/final",
    "meta-llama/Llama-3.1-8B-Instruct",
]
model_names = [
    "Llama-3.1-8B-Instruct-eng-history",
    "Llama-3.1-8B-Instruct",
]
test_data = "../../data/processed/eng_test_history.json"

for model_name, model_path in zip(model_names, model_paths):
    generate(model_path=model_path, model_name=model_name, output_path="../../outputs", test_data_path=test_data, disable_reasoning=False, **generation_kwargs)
    torch.cuda.empty_cache()
    gc.collect()