class MultiQueryRetriever:

    
    def __init__(
        self,
        dense_retriever,
        sparse_retriever,
        fusion_function
    ):

        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.fusion_function = fusion_function

    
    def retrieve(
        self,
        queries,
        top_k=10,
        final_top_k=10
    ):

        all_rankings = []

        for query in queries:

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

            hybrid_results = self.fusion_function(
                rankings=[
                    dense_results,
                    sparse_results
                ],
                k=60,
                top_n=final_top_k 
            )

            all_rankings.append(
                hybrid_results 
            )

        return self.fusion_function(
            rankings=all_rankings,
            k=60,
            top_n=final_top_k
        )