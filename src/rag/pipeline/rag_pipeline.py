
from src.rag.retrieval.fusion import (
    reciprocal_rank_fusion
)

class RAGPipeline:

    def __init__(
        self,
        router,
        dense_retriever,
        sparse_retriever=None,
        graph_retriever=None,
        reranker=None 
    ):

        self.router = router 

        self.dense_retriever = (
            dense_retriever
        )

        self.sparse_retriever = (
            sparse_retriever
        )

        self.graph_retriever = (
            graph_retriever
        )

        self.reranker = reranker 


    def retrieve(
        self,
        query,
        top_k=10
    ):

        strategy = (
            self.router.route(query)
        )

        if (
            strategy.value == "dense"
        ):

            results = (
                self.dense_retriever.retrieve(
                    query,
                    top_k=top_k 
                )
            )

        elif (
            strategy.value == "hybrid"
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

            results = (
                reciprocal_rank_fusion(
                    rankings=[
                        dense_results,
                        sparse_results 
                    ],
                    k=60,
                    top_n=top_k
                )
            )

        elif (
            strategy.value == "graph"
        ):

            results = (
                self.graph_retriever.retrieve(
                    query 
                )
            )

        else:

            raise ValueError(
                f"Unsupported strategy: {strategy}"
            )

        return {
            "strategy": strategy.value ,

            "results": results
        }