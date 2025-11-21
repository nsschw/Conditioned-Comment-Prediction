import evaluate

class ROUGE:
    """ROUGE score calculator using Hugging Face's evaluate library."""
    
    def __init__(self):
        """Initialize the ROUGE metric."""
        self.rouge = evaluate.load("rouge")
        
    def __call__(self, source: list[str], target: list[str]) -> dict:
        """
        Calculate the ROUGE score between predictions and references.
        
        Args:
            source: List of source/reference texts
            target: List of target/generated texts
        
        Returns:
            ROUGE score dictionary.
        """

        results = self.rouge.compute(predictions=target, references=source)
        return results