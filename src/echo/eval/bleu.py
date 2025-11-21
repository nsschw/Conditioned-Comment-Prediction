import evaluate

class BLEU:
    """BLEU score calculator using Hugging Face's evaluate library."""
    
    def __init__(self):
        """Initialize the BLEU metric."""
        self.bleu = evaluate.load("bleu")
        
    def __call__(self, source: list[str], target: list[str]) -> dict:
        """
        Calculate the BLEU score between predictions and references.
        
        Args:
            source: List of source/reference texts
            target: List of target/generated texts
        
        Returns:
            BLEU score dictionary.
        """

        results = self.bleu.compute(predictions=target, references=[[ref] for ref in source])
        return results