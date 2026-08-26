class DenseSearchTool:

    def __init__(self, retriever):
        self.retriever = retriever 

    def run (
        self,
        query,
        top_k=5
    ):

        return self.retriever.retrieve(
            query,
            top_k=top_k
        )


class BM25SearchTool:

    def __init__(self, retriever):
        self.retriever = retriever 

    def run(
        self,
        query,
        top_k=5
    ):

        return self.retriever.retrieve(
            query,
            top_k=top_k 
        )