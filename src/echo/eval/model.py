from transformers import AutoModelForCausalLM,AutoTokenizer
import torch
from transformers import pipeline
from tqdm import tqdm
import pandas as pd
import json

class Model():
    """
    A class for loading and using a base language model without fine-tuning.
    This class provides a compatible interface with Model for evaluation.
    
    Attributes:
        model: The base language model.
        tokenizer: The tokenizer associated with the model.
    
    Example:
        >>> model = BaseModel("microsoft/Phi-4-mini-instruct")
        >>> response = model.generate([[{"role": "user", "content": "Hello, how are you?"}]])
    """

    def __init__(self, model_name: str, batch_size: int = 3):
        """
        Initialize a base model without fine-tuning.
        
        Args:
            model_name: The name or path of the pre-trained model.
        """
        # Load model and tokenizer directly
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            dtype = "auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        
        # Create a pipeline for generation
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, batch_size=batch_size)
        if self.tokenizer.pad_token_id is None:
            self.pipe.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        

    def generate(self, prompts, **kwargs) -> str:
        """
        Generate a text response for a list of chat messages.
        
        Args:
            prompts: A list of dictionaries representing a chat conversation,
                   where each dict has 'role' and 'content' keys.
        
        Returns:
            The generated text response as a string.
        """
        
        # Generate text using the pipeline
        outputs = self.pipe(
            prompts,
            return_full_text=False,
            **kwargs
        )
        generated_texts = [output[0]["generated_text"] for output in outputs]
        return generated_texts

    
    def push_to_hub(self, repo_name: str, organization: str = None):
        """
        Push the model to the Hugging Face Hub as a private repository.
            
        Args:
            repo_name: The name of the repository on the Hugging Face Hub.
            organization: Optional organization name for the repository.
        """
        self.model.push_to_hub(repo_name, organization=organization, private=False)
        self.tokenizer.push_to_hub(repo_name, organization=organization, private=False)