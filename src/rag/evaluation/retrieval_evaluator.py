class RetrievalEvaluator:

    def __init__(
        self,
        threshold=0.40
    ):

        self.threshold = threshold 

    
    def evaluate(
        self,
        documents,
        score_key="reranked_score"
    ):

        if not documents:

            return {
                "relevant": False,
                "score": 0.0
            }
        
        scores = [
            document.get(
                score_key,
                0.0
            )
            for document in documents 
        ]

        best_score = max(scores)

        return {
            "relevant": (
                best_score >= self.threshold
            ),
            "score": best_score 
        }