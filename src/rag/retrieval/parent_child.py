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

            start = end - self.chunk_overlap

        return chunks


class ParentChildRetriever:

    def __init__(
        self,
        embedding_model,
        vector_store,
        parent_store
    ):

        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.parent_store = parent_store

    def retrieve(
        self,
        query,
        top_k=5
    ):

        # Query → embedding
        query_vector = (
            self.embedding_model
            .embed_query(query)
        )

        # Search child vectors
        child_results = (
            self.vector_store.search(
                query_vector,
                top_k=top_k
            )
        )

        parents = []

        seen_parents = set()

        for child in child_results:

            parent_id = child.get(
                "parent_id"
            )

            if not parent_id:
                continue

            if parent_id in seen_parents:
                continue

            parent = (
                self.parent_store
                .get(parent_id)
            )

            if parent:

                parents.append(parent)

                seen_parents.add(
                    parent_id
                )

        return parents