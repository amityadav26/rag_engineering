import json

from src.rag.embeddings.embedding_model import (
    EmbeddingModel
)

from src.rag.vector_store.qdrant_store import (
    QdrantVectorStore
)

from src.rag.retrieval.dense import (
    DenseRetriever
)

from src.rag.retrieval.sparse import (
    BM25Retriever
)

from src.rag.retrieval.fusion import (
    reciprocal_rank_fusion
)


def load_documents(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def main():

    # -------------------------
    # 1. Load documents
    # -------------------------

    documents = load_documents(
        "data/documents.json"
    )

    # -------------------------
    # 2. Initialize embedding
    # -------------------------

    embedding_model = EmbeddingModel()

    # -------------------------
    # 3. Initialize Qdrant
    # -------------------------

    vector_store = QdrantVectorStore(
        collection_name="rag_documents",
        vector_size=384
    )

    # -------------------------
    # 4. Create embeddings
    # -------------------------

    texts = [
        document["text"]
        for document in documents
    ]

    embeddings = (
        embedding_model
        .embed_documents(texts)
    )

    # -------------------------
    # 5. Store in Qdrant
    # -------------------------

    vector_store.add_documents(
        documents,
        embeddings
    )

    # -------------------------
    # 6. Create retrievers
    # -------------------------

    dense_retriever = DenseRetriever(
        embedding_model,
        vector_store
    )

    sparse_retriever = BM25Retriever(
        documents
    )

    # -------------------------
    # 7. Query
    # -------------------------

    query = (
        "What does ERR_CONNECTION_RESET mean?"
    )

    # -------------------------
    # 8. Dense retrieval
    # -------------------------

    dense_results = (
        dense_retriever.retrieve(
            query,
            top_k=10
        )
    )

    # -------------------------
    # 9. Sparse retrieval
    # -------------------------

    sparse_results = (
        sparse_retriever.retrieve(
            query,
            top_k=10
        )
    )

    # -------------------------
    # 10. Hybrid retrieval
    # -------------------------

    hybrid_results = (
        reciprocal_rank_fusion(
            rankings=[
                dense_results,
                sparse_results
            ],
            k=60,
            top_n=5
        )
    )

    # -------------------------
    # 11. Print results
    # -------------------------

    print("\n===== DENSE =====")

    for result in dense_results:
        print(
            result["id"],
            result["score"],
            result["text"]
        )

    print("\n===== BM25 =====")

    for result in sparse_results:
        print(
            result["id"],
            result["score"],
            result["text"]
        )

    print("\n===== HYBRID / RRF =====")

    for result in hybrid_results:
        print(
            result["id"],
            result["rrf_score"],
            result["text"]
        )


if __name__ == "__main__":
    main()