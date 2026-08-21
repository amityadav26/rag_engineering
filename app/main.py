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

from src.rag.reranking.cross_encoder import (
    CrossEncoderReranker
)

from src.rag.query.multi_query import (
    MultiQueryRetriever
)


def load_documents(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def print_results(
    title,
    results,
    score_key=None
):

    print(f"\n===== {title} =====")

    for result in results:

        if score_key:

            print(
                result["id"],
                result.get(score_key),
                result["text"]
            )

        else:

            print(
                result["id"],
                result["text"]
            )


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
    # 5. Store documents
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
    # 7. Create Multi-Query
    # -------------------------

    multi_query_retriever = MultiQueryRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        fusion_function=reciprocal_rank_fusion
    )

    # -------------------------
    # 8. Create Reranker
    # -------------------------

    reranker = CrossEncoderReranker()

    # -------------------------
    # 9. Original Query
    # -------------------------

    query = (
        "What causes TCP connection resets?"
    )

    # -------------------------
    # 10. Manually generated
    #     Multi-Queries
    # -------------------------

    multi_queries = [

        "What causes ERR_CONNECTION_RESET?",

        "What causes TCP connection resets?",

        "How can ERR_CONNECTION_RESET be diagnosed?",

        "What server problems cause connection resets?"

    ]

    # =====================================================
    # BASELINE HYBRID PIPELINE
    # =====================================================

    # -------------------------
    # 11. Dense Retrieval
    # -------------------------

    dense_results = (
        dense_retriever.retrieve(
            query,
            top_k=10
        )
    )

    # -------------------------
    # 12. Sparse Retrieval
    # -------------------------

    sparse_results = (
        sparse_retriever.retrieve(
            query,
            top_k=10
        )
    )

    # -------------------------
    # 13. Hybrid / RRF
    # -------------------------

    hybrid_results = (
        reciprocal_rank_fusion(
            rankings=[
                dense_results,
                sparse_results
            ],
            k=60,
            top_n=10
        )
    )

    # -------------------------
    # 14. Neural Reranking
    # -------------------------

    final_results = reranker.rerank(
        query=query,
        documents=hybrid_results,
        top_k=3
    )

    # =====================================================
    # MULTI-QUERY PIPELINE
    # =====================================================

    # -------------------------
    # 15. Multi-Query Retrieval
    # -------------------------

    multi_query_results = (
        multi_query_retriever.retrieve(
            queries=multi_queries,
            top_k=10,
            final_top_k=10
        )
    )

    # -------------------------
    # 16. Multi-Query Reranking
    # -------------------------

    multi_query_final = (
        reranker.rerank(
            query=query,
            documents=multi_query_results,
            top_k=3
        )
    )

    # =====================================================
    # PRINT RESULTS
    # =====================================================

    # -------------------------
    # Dense
    # -------------------------

    print_results(
        "DENSE",
        dense_results,
        score_key="score"
    )

    # -------------------------
    # BM25
    # -------------------------

    print_results(
        "BM25 / SPARSE",
        sparse_results,
        score_key="score"
    )

    # -------------------------
    # Hybrid / RRF
    # -------------------------

    print_results(
        "HYBRID / RRF",
        hybrid_results,
        score_key="rrf_score"
    )

    # -------------------------
    # Baseline Reranker
    # -------------------------

    print_results(
        "HYBRID + CROSS ENCODER",
        final_results,
        score_key="reranked_score"
    )

    # -------------------------
    # Multi-Query
    # -------------------------

    print_results(
        "MULTI-QUERY",
        multi_query_results,
        score_key="rrf_score"
    )

    # -------------------------
    # Multi-Query + Reranker
    # -------------------------

    print_results(
        "MULTI-QUERY + CROSS ENCODER",
        multi_query_final,
        score_key="reranked_score"
    )


if __name__ == "__main__":
    main()