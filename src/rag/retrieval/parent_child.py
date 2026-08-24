class ParentStore:

    def __init__(self, documents):

        self.documents = {
            document["id"]: document
            for document in documents
        }

    def get(self, parent_id):

        return self.documents.get(parent_id)


class ChildChunker:

    def __init__(
        self,
        chunk_size=50,
        chunk_overlap=10
    ):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document):

        words = document["text"].split()

        chunks = []

        start = 0

        child_number = 1

        while start < len(words):

            end = start + self.chunk_size

            chunk_text = " ".join(
                words[start:end]
            )

            child = {

                "id": (
                    f'{document["id"]}'
                    f'_child_{child_number}'
                ),

                "text": chunk_text,

                "parent_id": document["id"],

                "metadata": document.get(
                    "metadata",
                    {}
                )
            }

            chunks.append(child)

            child_number += 1

            start = (
                end - self.chunk_overlap
            )

        return chunks


class ParentChildRetriever:

    def __init__(
        self,
        dense_retriever,
        sparse_retriever,
        fusion_function,
        reranker,
        parent_store
    ):

        self.dense_retriever = (
            dense_retriever
        )

        self.sparse_retriever = (
            sparse_retriever
        )

        self.fusion_function = (
            fusion_function
        )

        self.reranker = reranker

        self.parent_store = parent_store


    def retrieve(
        self,
        query,
        dense_top_k=10,
        sparse_top_k=10,
        fusion_top_k=10,
        rerank_top_k=3
    ):

        # =================================
        # 1. Dense Retrieval
        # =================================

        dense_results = (
            self.dense_retriever.retrieve(
                query,
                top_k=dense_top_k
            )
        )


        # =================================
        # 2. Sparse Retrieval
        # =================================

        sparse_results = (
            self.sparse_retriever.retrieve(
                query,
                top_k=sparse_top_k
            )
        )


        # =================================
        # 3. Hybrid Retrieval
        # =================================

        hybrid_results = (
            self.fusion_function(
                rankings=[
                    dense_results,
                    sparse_results
                ],
                k=60,
                top_n=fusion_top_k
            )
        )


        # =================================
        # 4. Cross Encoder Reranking
        # =================================

        reranked_results = (
            self.reranker.rerank(
                query=query,
                documents=hybrid_results,
                top_k=rerank_top_k
            )
        )


        # =================================
        # 5. Child → Parent
        # =================================

        parent_results = []

        seen_parents = set()

        for child in reranked_results:

            parent_id = child.get(
                "parent_id"
            )

            if not parent_id:
                continue

            if parent_id in seen_parents:
                continue

            parent = (
                self.parent_store.get(
                    parent_id
                )
            )

            if parent:

                parent_results.append(
                    {
                        "parent": parent,
                        "matched_child": child
                    }
                )

                seen_parents.add(
                    parent_id
                )


        # =================================
        # 6. Return all stages
        # =================================

        return {

            "dense": dense_results,

            "sparse": sparse_results,

            "hybrid": hybrid_results,

            "reranked": reranked_results,

            "parents": parent_results
        }