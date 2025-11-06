import logging
import requests
import typing
import time

class LLM():

    def __init__(self,
                api_key: str,
                model: str = "Qwen/Qwen3-4B-Instruct-2507:nscale",
                url: str = "https://router.huggingface.co/v1/chat/completions"
                ):
        
        self.api_key: str = api_key
        self.model: str = model
        self.url: str = url

    def _query(self, payload):
        headers: dict = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(self.url, headers=headers, json=payload)
        return response.json()
    
    def generate(self, chat: list[dict], max_retries: int = 3) -> str:
        try:
            response: str = self._query(
                {
                    "messages": chat,
                    "model": self.model,
                }
            )["choices"][0]["message"]["content"]
        
        except Exception as e:
            logging.error(f"Failed to query LLM: {e}")
            if max_retries > 0:
                time.sleep(60)
                return self.generate(chat, max_retries - 1)
            raise RuntimeError("Failed to generate response from LLM after retries") from e
        
        return response