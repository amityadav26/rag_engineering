from rank_bm25 import BM25Okapi

class BM25Retriever:

    def __init__(self, documents):
        self.documents = documents 

        self.tokenized_documents = [
            document["text"].lower().split()
            for document in documents 
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_documents
        )

    def retrieve(self, query, top_k=10):

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for index in ranked_indices[:top_k]:

            result = self.documents[index].copy()

            result["score"] = float(
                scores[index]
            )

            results.append(result)

        return results 