import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

def analyse_dataset(model_name: str, dataset_path: str):
    """
    Loads tokenizer/dataset, prints 1st-4th percentiles of token lengths.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    def get_token_length(example):
        return {"length": len(tokenizer.apply_chat_template(example["messages"], add_special_tokens=False, tokenize=True))}

    token_lengths = dataset.map(get_token_length)["length"]
    
    percentiles_to_calc = [25, 50, 75, 90, 95, 99, 100]
    results = np.percentile(token_lengths, percentiles_to_calc)
    
    print(f"Total examples: {len(token_lengths)}")
    print(f"Mean token length: {np.mean(token_lengths):.1f}")
    print(f"Std deviation: {np.std(token_lengths):.1f}")
    print(f"Min token length: {np.min(token_lengths)}")
    print(f"Max token length: {np.max(token_lengths)}")
    
    print("\n--- Token Length Analysis ---")
    for p, res in zip(percentiles_to_calc, results):
        print(f" {p}th percentile: {int(res)}")
    print("-----------------------------\n")