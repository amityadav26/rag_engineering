from sentence_transformers import SentenceTransformer 

class EmbeddingModel:

    def __init__(
        self,
        model_name="all-MiniLM-L6-v2"
    ):

        self.model_name = model_name

        self.model = SentenceTransformer(model_name)

    
    def embed_documents(self, texts):

        return self.model.encode(
            texts,
            convert_to_numpy=True
            )

    
    def embed_query(self, query):

        return self.model.encode(
            query,
            convert_to_numpy=True 
            )
