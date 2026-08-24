import json

from src.rag.embeddings.embedding_model import (
    EmbeddingModel
)

from src.rag.vector_store.qdrant_store import (
    QdrantVectorStore
)

from src.rag.retrieval.parent_child import (
    ParentStore,
    ChildChunker,
    ParentChildRetriever
)


def load_documents(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def main():

    # =====================================================
    # 1. LOAD PARENT DOCUMENTS
    # =====================================================

    parent_documents = load_documents(
        "data/parent_documents.json"
    )

    print(
        f"Loaded parents: "
        f"{len(parent_documents)}"
    )


    # =====================================================
    # 2. CREATE PARENT STORE
    # =====================================================

    parent_store = ParentStore(
        parent_documents
    )

    print(
        "Parent store created."
    )


    # =====================================================
    # 3. CREATE CHILD CHUNKS
    # =====================================================

    chunker = ChildChunker(
        chunk_size=50,
        chunk_overlap=10
    )

    all_children = []

    for document in parent_documents:

        children = chunker.split(
            document
        )

        all_children.extend(
            children
        )


    print(
        f"Created children: "
        f"{len(all_children)}"
    )


    # =====================================================
    # 4. SHOW CHILD → PARENT RELATIONSHIP
    # =====================================================

    print(
        "\n===== CHILD → PARENT ====="
    )

    for child in all_children:

        print(
            f"\nChild: {child['id']}"
        )

        print(
            f"Parent: {child['parent_id']}"
        )

        print(
            f"Text: {child['text'][:100]}..."
        )


    # =====================================================
    # 5. INITIALIZE EMBEDDING MODEL
    # =====================================================

    embedding_model = EmbeddingModel()


    # =====================================================
    # 6. CREATE CHILD EMBEDDINGS
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


    print(
        f"\nCreated embeddings: "
        f"{len(child_embeddings)}"
    )


    # =====================================================
    # 7. INITIALIZE QDRANT
    # =====================================================

    vector_store = QdrantVectorStore(
        collection_name="parent_child_rag",
        vector_size=384
    )


    # =====================================================
    # 8. STORE CHILDREN IN QDRANT
    # =====================================================

    vector_store.add_documents(
        all_children,
        child_embeddings
    )

    print(
        "Children stored in Qdrant."
    )


    # =====================================================
    # 9. CREATE PARENT-CHILD RETRIEVER
    # =====================================================

    retriever = ParentChildRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        parent_store=parent_store
    )


    # =====================================================
    # 10. QUERY
    # =====================================================

    query = (
        "How many days can employees "
        "work remotely?"
    )

    print(
        f"\n===== QUERY =====\n"
        f"{query}"
    )


    # =====================================================
    # 11. RETRIEVE PARENT
    # =====================================================

    results = retriever.retrieve(
        query=query,
        top_k=1
    )


    # =====================================================
    # 12. PRINT PARENT RESULTS
    # =====================================================

    print(
        "\n===== PARENT-CHILD RESULTS ====="
    )

    for result in results:

        print(
            "\n-----------------------------"
        )

        print(
            "Parent ID:",
            result["id"]
        )

        print(
            "Title:",
            result["title"]
        )

        print(
            "Metadata:",
            result["metadata"]
        )

        print(
            "\nComplete Parent Document:"
        )

        print(
            result["text"]
        )


if __name__ == "__main__":
    main()