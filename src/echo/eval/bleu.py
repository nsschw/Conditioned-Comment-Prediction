import evaluate

class BLEU:
    """BLEU score calculator using Hugging Face's evaluate library."""
    
    def __init__(self):
        """Initialize the BLEU metric."""
        self.bleu = evaluate.load("bleu")
        
    def __call__(self, source: list[str], target: list[str], return_mean = True) -> float:
        """
        Calculate the BLEU score between predictions and references.
        
        Args:
            predictions: List of predicted sentences.
            references: List of reference sentences.
        
        Returns:
            BLEU score as a float.
        """

        if return_mean:
            results = self.bleu.compute(predictions=source, references=[[ref] for ref in target])
            return results
        else:
            scores = []
            for pred, ref in zip(source, target):
                result = self.bleu.compute(predictions=[pred], references=[[ref]])
                scores.append(result['bleu'])
            return scores
