from echo.eval.evaluate import evaluate
from pathlib import Path
import pandas as pd
import gc
import torch
import tqdm
import json

outputs_dir = Path("../../outputs/")
results = []
for model_dir in tqdm.tqdm(outputs_dir.iterdir()):
    if model_dir.is_dir():

        if model_dir.name != "Llama-3.1-8B-Instruct-eng-history-as4" and model_dir.name != "Llama-3.1-8B-Instruct-as4":
            continue

        predictions_file = list(model_dir.glob("predictions_*.json"))
        for predictions_file in predictions_file:
            results.append(evaluate(str(predictions_file)))
            # Clear CUDA memory between models
            torch.cuda.empty_cache()
            gc.collect()


# Flatten the nested structure
flattened_results = []
for result in results:
    flat = {
        'model_name': result['model_name'],
        'data_path': result['data_path'],
        'bleu': result['bleu']['bleu'],
        'bleu_precision_1': result['bleu']['precisions'][0],
        'bleu_precision_2': result['bleu']['precisions'][1],
        'bleu_precision_3': result['bleu']['precisions'][2],
        'bleu_precision_4': result['bleu']['precisions'][3],
        'brevity_penalty': result['bleu']['brevity_penalty'],
        'length_ratio': result['bleu']['length_ratio'],
        'translation_length': result['bleu']['translation_length'],
        'reference_length': result['bleu']['reference_length'],
        'rouge1': result['rouge']['rouge1'],
        'rouge2': result['rouge']['rouge2'],
        'rougeL': result['rouge']['rougeL'],
        'rougeLsum': result['rouge']['rougeLsum'],
        'qwen3_embedding_distance': result['qwen3_embedding_distance'],
        'gemma_embedding_distance': result['gemma_embedding_distance'],
        'luxembourgish_embedding_distance': result['luxembourgish_embedding_distance']
    }
    flattened_results.append(flat)

# Convert to DataFrame and save
df = pd.DataFrame(flattened_results)

def history_size(data_path):
    if "as4_" in data_path:
        size_str = data_path.split("as4_")[-1].split(".json")[0]
        return int(size_str)
    else:
        return 0

def case_count(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return len(data)
    
df['history_length'] = df['data_path'].apply(history_size)
df["case_count"] = df['data_path'].apply(case_count)
df.to_csv('../../outputs/as4_results.csv', index=False)