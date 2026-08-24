class HyDE:

    def __init__(self, embedding_model):

        self.embedding_model = embedding_model 

    def embed_hyothetical_document(self, hypothetical_document):
        
        return self.embedding_model.embed_query(
            hypothetical_document
        )