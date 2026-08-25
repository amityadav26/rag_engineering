class SelfRAG:

    def __init__(
        self,
        retriever,
        evaluator 
    ):

        self.retriever = retriever 
        self.evaluator = evaluator 

    
    def should_retrieve(
        self,
        query 
    ):

        # Simple first implementation
        # Retrieval is enabled by default 

        return True 


    def retrieve(
        self,
        query,
        top_k=10
    ):

        if not self.should_retrieve(query):

            return {
                "documents": [],
                "retrieved": False 
            }
        
        documents = self.retriever.retrieve(
            query,
            top_k=top_k 
        )

        evaluation = self.evaluator.evaluate(
            documents
        )

        return {
            "documents": documents,
            "retrieved": True,
            "relevant": evaluation["relevant"],
            "evaluation": evaluation
        }