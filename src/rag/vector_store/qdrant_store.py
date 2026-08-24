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
            self.client.get_collections().collections 
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

        for idx, (document, embedding) in enumerate(
            zip(documents, embeddings)
        ):

            payload = {
                "text": document["text"],
                "metadata": document.get(
                    "metadata",
                    {}
                ),
                "original_id": document["id"],

                "parent_id": document.get(
                    "parent_id"
                )
            }

            # Parent-child documents
            if "parent_id" in document:

                payload["parent_id"] = (
                    document["parent_id"]
                )

            points.append(
                PointStruct(
                    id=idx + 1,
                    vector=embedding.tolist(),
                    payload=payload
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
                "id": result.payload.get(
                    "original_id",
                    result.id
                ),

                "score": result.score,

                "text": result.payload["text"],
                
                "parent_id": result.payload.get("parent_id"),

                "metadata": result.payload.get("metadata", {})
            }
            for result in results 
        ]