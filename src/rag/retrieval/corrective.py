class CorrectiveRetriever:

    def __init__(
        self,
        retriever,
        evaluator,
        fallback_retriever = None 
    ):

        self.retriever = retriever 
        self.evaluator = evaluator 
        self.fallback_retriever = (
            fallback_retriever
        )


    def retrieve(
        self,
        query,
        top_k=10
    ):

        documents = self.retriever.retrieve(
            query,
            top_k=top_k 
        )

        evaluation = self.evaluator.evaluate(
            documents 
        )

        if evaluation["relevant"]:

            return {
                "douments": documents,
                "corrected": False,
                "evaluation": evaluation
            }

        if self.fallback_retriever:

            fallback_documents = (
                self.fallback_retriever.retrieve(
                    query,
                    top_k=top_k 
                )
            )

            return {
                "documents": fallback_documents,
                "corrected": True,
                "evaluation": evaluation 
            }

        return {
            "documents": documents,
            "corrected": False,
            "evaluation": evaluation
        }