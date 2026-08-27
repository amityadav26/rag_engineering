import json

from src.rag.pipeline.rag_pipeline import (
    RAGPipeline 
)

from src.rag.retrieval.router import (
    RetrievalRouter
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

from src.rag.embeddings.embedding_model import (
    EmbeddingModel
)

from src.rag.vector_store.qdrant_store import (
    QdrantVectorStore
)

from src.rag.reranking.cross_encoder import (
    CrossEncoderReranker
)

# =========== LOAD DOCUMENTS ============

def load_documents(path):

    with open(
        path, "r", encoding="utf-8"
    ) as files: 

        return json.load(files)

# ========= MAIN =======================

def main():

    # ========= load documents =================

    documents = load_documents(
        "data/documents.json"
    )

    print(
        "\nTotal documents:",
        len(documents)
    )

    # ========== Embedding Model ================

    embedding_model = EmbeddingModel()

    #  =========== Qdrant Vector Store ==========

    vector_store = QdrantVectorStore(
        collection_name = "rag_lesson18_documents",
        vector_size = 384 
    )

    # =========== Create documents embeddings ===========

    texts = [
        document["text"]
        for document in documents 
    ]

    embeddings = (
        embedding_model.embed_documents(texts)
    )

    # =========== Store documents in Qdrant ===========

    vector_store.add_documents(
        documents=documents,
        embeddings=embeddings
    )

    # ============== Dense Retriever ===================

    dense_retriever = DenseRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store
    )

    # =============== Sparse Retriever =================

    sparse_retriever = BM25Retriever(
        documents=documents 
    )

    # ============== Rreranker ==========================

    reranker = CrossEncoderReranker()

    # ============ Router ==============================

    router = RetrievalRouter()

    # ============== RAG Pipeline =====================

    pipeline = RAGPipeline(
        router=router,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        graph_retriever=None,
        reranker=reranker 
    )

    # =============== Test Queries =======================

    queries = [

        "What is the remote work policy?",

        "What causes ERR_CONNECTION_RESET?",

        "How error code TCP connection reset be diagnosed?",

        "How many days can employees work remotely?",

        "What server problems can cause connection failures?"

    ]

    # ==================== Run Pipeline ====================

    for query in queries:

        print(
            "\n\n==============================================="
        )

        print(
            "QUERY:"
        )

        print(query) 

        result = pipeline.retrieve(
            query=query,
            top_k=5 
        )

        # ================= Selected Strategy ===================

        print(
            "\nSELECTED STRATEGY:"
        )

        print(
            result["strategy"]
        )

        # ================== Retrieved Documents ================

        print(
            "\nRETRIEVED DOCUMENTS:"
        )

        for index, document in enumerate(
            result["results"],
            start=1
        ):

            print(
                f"\n-------- Result {index} ------------"
            )

            print(
                "ID:",
                document.get("id")
            )

            print(
                "Score:",
                document.get("score")
            )

            print(
                "Text:",
                document.get("text")
            )


# ============= ENTRY PRINT ===============

if __name__ == "__main__":

    main()