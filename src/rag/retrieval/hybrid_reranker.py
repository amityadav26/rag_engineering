from src.rag.retrieval.fusion import (
    reciprocal_rank_fusion 
)


class HybridRerankerRetriever:

    def __init__(
        self,
        dense_retriever,
        sparse_retriever,
        reranker 
    ):

        self.dense_retriever = (
            dense_retriever
        )

        self.sparse_retriever = (
            sparse_retriever
        )

        self.reranker = reranker