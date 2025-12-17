from echo.eval.generate import generate
import torch
import gc

generation_kwargs = {
    "max_new_tokens": 500,
    "temperature": 0.75,
}

model_paths = [
    "../../models/Llama-3.1-8B-Instruct-eng-history/20251125_083546/final",
    "meta-llama/Llama-3.1-8B-Instruct",
]
model_names = [
    "Llama-3.1-8B-Instruct-eng-history-as4",
    "Llama-3.1-8B-Instruct-as4",
]

for model_name, model_path in zip(model_names, model_paths):

    for i in range(30):
        temp_test_data = f"../../data/processed/eng_test_history_as4_{i}.json"
        for _ in range(5):
            generate(model_path=model_path, model_name=model_name, output_path="../../outputs", test_data_path=temp_test_data, disable_reasoning=False, batch_size=16, **generation_kwargs)

    torch.cuda.empty_cache()
    gc.collect()