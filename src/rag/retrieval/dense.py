class DenseRetrieval:

    def __init__(
        self,
        embedding_model,
        vector_store
    ):

        self.embedding_model = embedding_model 
        self.vector_store = vector_store 

    
    def retrieve (
        self,
        query,
        top_k=10 
    ):

        query_vector = (
            self.embedding_model.embed_query(query)
        )

        results = self.vector_store.search(
            query_vector,
            top_k=top_k 
        )

        return results 