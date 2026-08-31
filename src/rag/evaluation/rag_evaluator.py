class RAGEvaluator:

    def __init__(
        self,
        llm 
    ):

        self.llm = llm 

    
    def evaluate_faithfulness(
        self,
        question,
        context,
        answer
    ):

        prompt = f"""
        You are evaluating a RAG answer.

        Question:
        {question}

        Context:
        {context}

        Answer:
        {answer}

        Determine whether every important clain in the answer is
        supported by the context.

        Return a score from 0 to 1.
        0 = completely unsupported
        1 = completely supported
        """

        result = self.llm.invoke(
            prompt 
        )

        return result 