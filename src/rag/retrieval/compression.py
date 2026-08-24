class ContextualCompressor:

    def __init__(
        self,
        max_sentences=3
    ):

        self.max_sentences = (
            max_sentences
        )


    def compress(
        self,
        query,
        documents
    ):

        query_terms = set(
            query.lower().split()
        )

        compressed_documents = []

        for document in documents:

            text = document["text"]

            sentences = [
                sentence.strip()
                for sentence in text.split(".")
                if sentence.strip()
            ]

            scored_sentences = []

            for sentence in sentences:

                sentence_terms = set(
                    sentence.lower().split()
                )

                overlap = len(
                    query_terms
                    & sentence_terms
                )

                scored_sentences.append(
                    (
                        overlap,
                        sentence
                    )
                )

            scored_sentences.sort(
                reverse=True
            )

            selected = [
                sentence
                for score, sentence
                in scored_sentences[
                    :self.max_sentences
                ]
            ]

            compressed = document.copy()

            compressed[
                "compressed_text"
            ] = ". ".join(
                selected
            )

            compressed_documents.append(
                compressed
            )

        return compressed_documents