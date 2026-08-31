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

from src.rag.evaluation.retrieval_evaluator import (
    RetrievalEvaluator
)

from src.rag.evaluation.evaluate_retrievers import (
    RetrieverEvaluatorRunner,
    load_evalauation_dataset
)

from src.rag.retrieval.hybrid import (
    HybridRetriever
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

    query = "What causes TCP connection resets?"

    
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

    # # =============== Test Queries =======================



    evaluation_dataset = load_evalauation_dataset(
        "data/evaluation_dataset.json"
    )

    evaluator = RetrievalEvaluator()

    runner = RetrieverEvaluatorRunner(
        evaluator=evaluator
    )

    dense_metrics = runner.evaluate(
        retriever=dense_retriever,
        evaluation_dataset=evaluation_dataset,
        top_k=5 
    )

    print(
        "\n========== DENSE EVALUATION =========="
    )

    for metric, value in dense_metrics.items():

        print(
            metric,
            ":",
            round(value, 4)
        )

    sparse_retriever = BM25Retriever(
        documents=documents 
    )

    bm25_metrics = runner.evaluate(
        retriever=sparse_retriever,
        evaluation_dataset=evaluation_dataset,
        top_k=5 
    )

    print(
        "\n========= BM25 EVALUATION ==========="
    )

    for metric, value in bm25_metrics.items():

        print(
            metric,
            ":",
            round(value, 4)
        )

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever
    )

    hybrid_metrics = runner.evaluate(
        retriever=hybrid_retriever,
        evaluation_dataset=evaluation_dataset,
        top_k=5 
    )

    print(
        "\n========= HYBRID EVALUATION ==========="
    )

    for metric, value in hybrid_metrics.items():

        print(
            metric,
            ":",
            round(value, 4)
        )

    
    retrievers = {

        "Dense": dense_retriever,

        "BM25": sparse_retriever,

        "Hybrid": hybrid_retriever
    }

    comparison = runner.compare(
        retrievers=retrievers,
        evaluation_dataset=evaluation_dataset,
        top_k=5 
    )

    print (
        "\n========== RETRIEVER COMPARSION =========="
    )

    for retriever_name , metrics in comparison.items():

        print(
            f"\n{retriever_name}"
        )

        for metric, value in metrics.items():

            print(
                f"{metric}: {value:.4f}"
            )

# ============= ENTRY PRINT ===============

if __name__ == "__main__":

    main()