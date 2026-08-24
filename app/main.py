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


from src.rag.retrieval.parent_child import (
    ParentStore,
    ChildChunker,
    ParentChildRetriever
)

from src.rag.retrieval.contextual import (
    ContextualChunker 
)

from src.rag.retrieval.compression import (
    ContextualCompressor
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


def main():

    # =========================================
    # 1. Load Parent Documents
    # =========================================

    parent_documents = load_documents(
        "data/parent_documents.json"
    )


    # =========================================
    # 2. Create Parent Store
    # =========================================

    parent_store = ParentStore(
        parent_documents
    )


    # =========================================
    # 3. Create Child Chunks
    # =========================================

    chunker = ChildChunker(
        chunk_size=50,
        chunk_overlap=10
    )

    contextual_chunker = ContextualChunker(
        chunker=chunker
    )

    document = parent_documents[0]

    children = contextual_chunker.split(document)

    print("\n====== CONTEXTUAL CHUNKS =======")

    for child in children:

        print("\nChild ID:", child["id"])

        print("Original:", child["original_text"])

        print("Contextual:", child["text"])



    all_children = []


    for document in parent_documents:

        children = contextual_chunker.split(
            document
        )

        all_children.extend(
            children
        )


    print(
        "\nTotal Parent Documents:",
        len(parent_documents)
    )


    print(
        "Total Child Chunks:",
        len(all_children)
    )


    # =========================================
    # 4. Initialize Embedding Model
    # =========================================

    embedding_model = EmbeddingModel()


    # =========================================
    # 5. Create Child Embeddings
    # =========================================

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


    # =========================================
    # 6. Initialize Qdrant
    # =========================================

    vector_store = QdrantVectorStore(

        collection_name=(
            "contextual_child_rag"
        ),

        vector_size=384
    )


    # =========================================
    # 7. Store Child Vectors
    # =========================================

    vector_store.add_documents(

        all_children,

        child_embeddings
    )


    # =========================================
    # 8. Dense Retriever
    # =========================================

    dense_retriever = DenseRetriever(

        embedding_model,

        vector_store
    )


    # =========================================
    # 9. Sparse Retriever
    # =========================================

    sparse_retriever = BM25Retriever(

        all_children
    )


    # =========================================
    # 10. Cross Encoder
    # =========================================

    reranker = CrossEncoderReranker()


    # =========================================
    # 11. Parent-Child Retriever
    # =========================================

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

    compressor = ContextualCompressor(
        max_sentences=3
    )


    # =========================================
    # 12. Query
    # =========================================

    query = (

        "How many days can employees "
        "work remotely?"
    )


    print(
        "\n===== QUERY ====="
    )

    print(query)


    # =========================================
    # 13. Run Complete Pipeline
    # =========================================

    results = (

        parent_child_retriever.retrieve(

            query=query,

            dense_top_k=10,

            sparse_top_k=10,

            fusion_top_k=10,

            rerank_top_k=3
        )
    )

    parent_documents = [
        result["parent"]
        for result in results["parents"]
    ]

    compressed_results = compressor.compress(
        query=query,
        documents=parent_documents
    )

    # =========================================
    # 14. Dense Results
    # =========================================

    print_results(

        "DENSE CHILDREN",

        results["dense"],

        score_key="score"
    )


    # =========================================
    # 15. BM25 Results
    # =========================================

    print_results(

        "BM25 CHILDREN",

        results["sparse"],

        score_key="score"
    )


    # =========================================
    # 16. RRF Results
    # =========================================

    print_results(

        "HYBRID / RRF CHILDREN",

        results["hybrid"],

        score_key="rrf_score"
    )


    # =========================================
    # 17. Reranked Results
    # =========================================

    print_results(

        "CROSS ENCODER RESULTS",

        results["reranked"],

        score_key="reranked_score"
    )


    # =========================================
    # 18. Final Parent Documents
    # =========================================

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



        print(
            "\n========= COMPRESSED CONTEXT=================" 
        )

        for document in compressed_results:

            print(
                "\nParent:", document["id"]
            )

            print(document["compressed_text"])

if __name__ == "__main__":

    main()