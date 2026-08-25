import json


# =========================================================
# Embeddings
# =========================================================

from src.rag.embeddings.embedding_model import (
    EmbeddingModel
)


# =========================================================
# Vector Store
# =========================================================

from src.rag.vector_store.qdrant_store import (
    QdrantVectorStore
)


# =========================================================
# Retrievers
# =========================================================

from src.rag.retrieval.dense import (
    DenseRetriever
)

from src.rag.retrieval.sparse import (
    BM25Retriever
)

from src.rag.retrieval.fusion import (
    reciprocal_rank_fusion
)


# =========================================================
# Reranking
# =========================================================

from src.rag.reranking.cross_encoder import (
    CrossEncoderReranker
)


# =========================================================
# Parent-Child Retrieval
# =========================================================

from src.rag.retrieval.parent_child import (
    ParentStore,
    ChildChunker,
    ParentChildRetriever
)


# =========================================================
# Contextual Chunking
# =========================================================

from src.rag.retrieval.contextual import (
    ContextualChunker
)


# =========================================================
# Context Compression
# =========================================================

from src.rag.retrieval.compression import (
    ContextualCompressor
)


# =========================================================
# Lesson 15 - Retrieval Evaluation
# =========================================================

from src.rag.evaluation.retrieval_evaluator import (
    RetrievalEvaluator
)


# =========================================================
# Lesson 15 - Corrective RAG
# =========================================================

from src.rag.retrieval.corrective import (
    CorrectiveRetriever
)


# =========================================================
# Lesson 15 - Self RAG
# =========================================================

from src.rag.retrieval.self_rag import (
    SelfRAG
)


# =========================================================
# Lesson 15 - Adaptive RAG Router
# =========================================================

from src.rag.retrieval.router import (
    RetrievalRouter
)


# =========================================================
# Utility Functions
# =========================================================

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

    print(
        f"\n===== {title} ====="
    )

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


# =========================================================
# MAIN
# =========================================================

def main():

    # =====================================================
    # 1. Load Parent Documents
    # =====================================================

    parent_documents = load_documents(
        "data/parent_documents.json"
    )


    print(
        "\nTotal Parent Documents:",
        len(parent_documents)
    )


    # =====================================================
    # 2. Create Parent Store
    # =====================================================

    parent_store = ParentStore(
        parent_documents
    )


    # =====================================================
    # 3. Create Child Chunker
    # =====================================================

    chunker = ChildChunker(
        chunk_size=50,
        chunk_overlap=10
    )


    # =====================================================
    # 4. Create Contextual Chunker
    # =====================================================

    contextual_chunker = ContextualChunker(
        chunker=chunker
    )


    # =====================================================
    # 5. Create Contextual Child Chunks
    # =====================================================

    all_children = []


    for document in parent_documents:

        children = contextual_chunker.split(
            document
        )

        all_children.extend(
            children
        )


    print(
        "Total Child Chunks:",
        len(all_children)
    )


    # =====================================================
    # Optional - Display First Document Chunks
    # =====================================================

    document = parent_documents[0]

    children = contextual_chunker.split(
        document
    )


    print(
        "\n====== CONTEXTUAL CHUNKS ======"
    )


    for child in children:

        print(
            "\nChild ID:",
            child["id"]
        )

        print(
            "Original:",
            child["original_text"]
        )

        print(
            "Contextual:",
            child["text"]
        )


    # =====================================================
    # 6. Initialize Embedding Model
    # =====================================================

    embedding_model = EmbeddingModel()


    # =====================================================
    # 7. Create Child Embeddings
    # =====================================================

    child_texts = [

        child["text"]

        for child in all_children
    ]


    child_embeddings = (

        embedding_model
        .embed_documents(
            child_texts
        )

    )


    # =====================================================
    # 8. Initialize Qdrant
    # =====================================================

    vector_store = QdrantVectorStore(

        collection_name=(
            "contextual_child_rag"
        ),

        vector_size=384
    )


    # =====================================================
    # 9. Store Child Vectors
    # =====================================================

    vector_store.add_documents(

        all_children,

        child_embeddings
    )


    # =====================================================
    # 10. Create Dense Retriever
    # =====================================================

    dense_retriever = DenseRetriever(

        embedding_model,

        vector_store
    )


    # =====================================================
    # 11. Create Sparse Retriever
    # =====================================================

    sparse_retriever = BM25Retriever(

        all_children
    )


    # =====================================================
    # 12. Create Cross Encoder
    # =====================================================

    reranker = CrossEncoderReranker()


    # =====================================================
    # 13. Create Parent-Child Retriever
    # =====================================================

    parent_child_retriever = (

        ParentChildRetriever(

            dense_retriever=(
                dense_retriever
            ),

            sparse_retriever=(
                sparse_retriever
            ),

            fusion_function=(
                reciprocal_rank_fusion
            ),

            reranker=reranker,

            parent_store=parent_store
        )
    )


    # =====================================================
    # 14. Contextual Compressor
    # =====================================================

    compressor = ContextualCompressor(
        max_sentences=3
    )


    # =====================================================
    # 15. Retrieval Evaluator
    # =====================================================

    evaluator = RetrievalEvaluator(
        threshold=0.40
    )


    # =====================================================
    # 16. Corrective RAG
    # =====================================================

    corrective_retriever = CorrectiveRetriever(

        retriever=dense_retriever,

        evaluator=evaluator,

        fallback_retriever=sparse_retriever
    )


    # =====================================================
    # 17. Self RAG
    # =====================================================

    self_rag = SelfRAG(

        retriever=dense_retriever,

        evaluator=evaluator
    )


    # =====================================================
    # 18. Adaptive RAG Router
    # =====================================================

    router = RetrievalRouter()


    # =====================================================
    # 19. Query
    # =====================================================

    query = (

        "How many days can employees "
        "work remotely?"
    )


    print(
        "\n========================================"
    )

    print(
        "QUERY:"
    )

    print(query)

    print(
        "========================================"
    )


    # =====================================================
    # 20. Main Parent-Child RAG Pipeline
    # =====================================================

    results = (

        parent_child_retriever.retrieve(

            query=query,

            dense_top_k=10,

            sparse_top_k=10,

            fusion_top_k=10,

            rerank_top_k=3
        )
    )


    # =====================================================
    # 21. Evaluate Cross-Encoder Retrieval
    # =====================================================

    evaluation = evaluator.evaluate(

        results["reranked"],

        score_key="reranked_score"
    )


    print(
        "\n===== RETRIEVAL EVALUATION ====="
    )


    print(
        "Relevant:",
        evaluation["relevant"]
    )


    print(
        "Best Score:",
        evaluation["score"]
    )


    # =====================================================
    # 22. Get Parent Documents
    # =====================================================

    parent_results = [

        result["parent"]

        for result in results["parents"]
    ]


    # =====================================================
    # 23. Compress Parent Context
    # =====================================================

    compressed_results = compressor.compress(

        query=query,

        documents=parent_results
    )


    # =====================================================
    # 24. Print Dense Results
    # =====================================================

    print_results(

        "DENSE CHILDREN",

        results["dense"],

        score_key="score"
    )


    # =====================================================
    # 25. Print BM25 Results
    # =====================================================

    print_results(

        "BM25 CHILDREN",

        results["sparse"],

        score_key="score"
    )


    # =====================================================
    # 26. Print RRF Results
    # =====================================================

    print_results(

        "HYBRID / RRF CHILDREN",

        results["hybrid"],

        score_key="rrf_score"
    )


    # =====================================================
    # 27. Print Cross Encoder Results
    # =====================================================

    print_results(

        "CROSS ENCODER RESULTS",

        results["reranked"],

        score_key="reranked_score"
    )


    # =====================================================
    # 28. Print Final Parent Documents
    # =====================================================

    print(
        "\n===== FINAL PARENT DOCUMENTS ====="
    )


    for result in results["parents"]:

        parent = result["parent"]

        child = result["matched_child"]


        print(
            "\n--------------------------------"
        )


        print(
            "PARENT ID:",
            parent["id"]
        )


        print(
            "MATCHED CHILD:",
            child["id"]
        )


        print(
            "RERANK SCORE:",
            child.get(
                "reranked_score"
            )
        )


        print(
            "\nFULL PARENT DOCUMENT:"
        )


        print(
            parent["text"]
        )


    # =====================================================
    # 29. Print Compressed Context
    # =====================================================

    print(
        "\n===== COMPRESSED CONTEXT ====="
    )


    for document in compressed_results:

        print(
            "\nParent:",
            document["id"]
        )


        print(
            document["compressed_text"]
        )


    # =====================================================
    # LESSON 15
    # CORRECTIVE RAG
    # =====================================================

    corrective_results = (

        corrective_retriever.retrieve(

            query=query,

            top_k=10
        )
    )


    print(
        "\n========================================"
    )

    print(
        "CORRECTIVE RAG"
    )

    print(
        "========================================"
    )


    print(
        "Corrected:",
        corrective_results["corrected"]
    )


    print(
        "Relevant:",
        corrective_results["evaluation"]["relevant"]
    )


    print(
        "Best Score:",
        corrective_results["evaluation"]["score"]
    )


    print(
        "\nCorrective Documents:"
    )


    for document in corrective_results["documents"]:

        print(

            document["id"],

            document.get("score"),

            document["text"]

        )


    # =====================================================
    # LESSON 15
    # SELF RAG
    # =====================================================

    self_rag_results = (

        self_rag.retrieve(

            query=query,

            top_k=10
        )
    )


    print(
        "\n========================================"
    )

    print(
        "SELF RAG"
    )

    print(
        "========================================"
    )


    print(
        "Retrieved:",
        self_rag_results["retrieved"]
    )


    print(
        "Relevant:",
        self_rag_results["relevant"]
    )


    print(
        "Best Score:",
        self_rag_results["evaluation"]["score"]
    )


    # =====================================================
    # LESSON 15
    # ADAPTIVE RAG ROUTER
    # =====================================================

    test_queries = [

        "Find the exact error code ERR_CONNECTION_RESET",

        "What is the remote work policy?",

        "What is the relationship between service A and service B?"

    ]


    print(
        "\n========================================"
    )

    print(
        "ADAPTIVE RAG ROUTER"
    )

    print(
        "========================================"
    )


    for test_query in test_queries:

        route = router.route(
            test_query
        )


        print(
            "\nQuery:",
            test_query
        )


        print(
            "Selected Route:",
            route
        )


    # =====================================================
    # LESSON 15 SUMMARY
    # =====================================================

    print(
        "\n========================================"
    )

    print(
        "LESSON 15 SUMMARY"
    )

    print(
        "========================================"
    )


    print(
        "\n[1] Retrieval Evaluation"
    )

    print(
        "Relevant:",
        evaluation["relevant"]
    )


    print(
        "\n[2] Corrective RAG"
    )

    print(
        "Corrected:",
        corrective_results["corrected"]
    )


    print(
        "\n[3] Self RAG"
    )

    print(
        "Retrieved:",
        self_rag_results["retrieved"]
    )


    print(
        "\n[4] Adaptive RAG"
    )

    print(
        "Router tested successfully"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()