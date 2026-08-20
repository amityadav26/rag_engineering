from qdrant_client import QdrantClient 
from qdrant_client.models import Filter, VectorParams, Distance, PointStruct


class QdrantVectorStore:

    def __init__(
        self,
        collection_name,
        vector_size=384,
        host="localhost",
        port=6333
    ):

        self.collection_name = collection_name

        self.client = QdrantClient(
            host=host,
            port=port 
        )

        self.vector_size = vector_size 

        self._create_collection()


    def _create_collection(self):

        collections = (
            self.client.get_collection().collections 
        )

        collection_names = [
            collection.name 
            for collection in collections 
        ]

        if self.collection_name not in collection_names:

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE 
                )
            )

    
    def add_documents(
        self,
        documents,
        embeddings
    ):

        points = []

        for document, embedding in zip(
            documents, embeddings
        ):
            points.append(
                PointStruct(
                    id=document["id"],
                    vector=embedding.tolist(),
                    payload={
                        "text": document["text"],
                        "metadata": document.get(
                            "metadata", {}
                        )
                    }
                )
            )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points 
        )


    def search(
        self, 
        query_vector,
        top_k=10 
    ):

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k
        ).points 

        return [
            {
                "id": result.id,
                "score": result.score,
                "text": payload["text"],
                "metadata": result.get("metadata", {})
            }
            for result in results 
        ]