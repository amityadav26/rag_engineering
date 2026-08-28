from src.rag.retrieval.fusion import (
    reciprocal_rank_fusion
)

class HybridRetriever:

    def __init__(
        self,
        dense_retriever,
        sparse_retriever
    ):

        self.dense_retriever = (
            dense_retriever 
        )

        self.sparse_retriever = (
            sparse_retriever
        )


    def retrieve(
        self,
        query,
        top_k=10 
    ):

        dense_results = (
            self.dense_retriever.retrieve(
                query,
                top_k=top_k 
            )
        )

        sparse_results = (
            self.sparse_retriever.retrieve(
                query,
                top_k=top_k 
            )
        )

        return reciprocal_rank_fusion(

            rankings = [
                dense_results,
                sparse_results
            ],

            k=60,

            top_n=top_k 
        )