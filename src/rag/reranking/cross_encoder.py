from sentence_transformers import CrossEncoder


class CrossEncoderReranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):

        self.model_name = model_name 

        self.model = CrossEncoder(
            model_name
        )

    def rerank(
        self,
        query,
        documents,
        top_k=3
    ):

        pairs = [
            [query, document["text"]]
            for document in documents 
        ]

        scores = self.model.predict(pairs)

        reranked = []

        for document, score in zip(
            documents, scores 
        ):

            result = document.copy()

            result["reranked_score"] = float(
                score 
            )

            reranked.append(result)
        
        reranked.sort(
            key=lambda x: x["reranked_score"],
            reverse=True  
        )

        return reranked[:top_k]
