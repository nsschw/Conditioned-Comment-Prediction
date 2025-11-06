from echo.utility.llm import LLM
import json
import dotenv
import random
import tqdm

def parse_history(history):
    output_str = ""
    for entry in reversed(history):
        if entry["role"] == "assistant":
            output_str += f"User: {entry['content']}\n\n"
    return output_str

ENV = dotenv.dotenv_values("../" * 2 + ".env")
PROFILER_LLM = LLM(api_key=ENV["HF_TOKEN"], model="deepseek-ai/DeepSeek-V3.2-Exp:novita")

with open("../../prompts/profiler.txt", "r") as f:
    PROFILER_PROMPT = f.read()


german_comments = json.load(open("../../data/intermediate/ger_30-shot_train.json", "r", encoding="utf-8"))
english_comments = json.load(open("../../data/intermediate/eng_30-shot_train.json", "r", encoding="utf-8"))
luxembourgish_comments = json.load(open("../../data/intermediate/lux_30-shot_train.json", "r", encoding="utf-8"))

random.seed(42)

german_comments = random.sample(german_comments, 3800)
luxembourgish_comments = random.sample(luxembourgish_comments, 3800)
english_comments = random.sample(english_comments, 3800)


for comment in tqdm.tqdm(luxembourgish_comments):
    history_str = parse_history(comment["messages"])
    prompt = PROFILER_PROMPT.format(content=history_str)
    response = PROFILER_LLM.generate([{"role": "user", "content": prompt}])
    comment["bio"] = response

    json.dump(luxembourgish_comments, open("../../data/intermediate/lux_30-shot_train_with_bios.json", "w", encoding="utf-8"))


for comment in tqdm.tqdm(german_comments):
    history_str = parse_history(comment["messages"])
    prompt = PROFILER_PROMPT.format(content=history_str)
    response = PROFILER_LLM.generate([{"role": "user", "content": prompt}])
    comment["bio"] = response

    json.dump(german_comments, open("../../data/intermediate/ger_30-shot_train_with_bios.json", "w", encoding="utf-8"))


for comment in tqdm.tqdm(english_comments):
    history_str = parse_history(comment["messages"])
    prompt = PROFILER_PROMPT.format(content=history_str)
    response = PROFILER_LLM.generate([{"role": "user", "content": prompt}])
    comment["bio"] = response

    json.dump(english_comments, open("../../data/intermediate/eng_30-shot_train_with_bios.json", "w", encoding="utf-8"))