class ContextualChunker:

    def __init__(self, chunker):

        self.chunker = chunker


    def create_context(self, parent):

        title = parent.get(
            "title",
            ""
        )

        metadata = parent.get(
            "metadata",
            {}
        )

        department = metadata.get(
            "department",
            ""
        )

        document_type = metadata.get(
            "document_type",
            ""
        )

        context_parts = []

        if title:
            context_parts.append(
                f"Title: {title}"
            )

        if department:
            context_parts.append(
                f"Department: {department}"
            )

        if document_type:
            context_parts.append(
                f"Type: {document_type}"
            )

        return " | ".join(
            context_parts
        )


    def split(self, parent):

        children = self.chunker.split(
            parent
        )

        context = self.create_context(
            parent
        )

        contextualized_children = []

        for child in children:

            new_child = child.copy()

            # Keep the original chunk
            new_child[
                "original_text"
            ] = child["text"]

            # Add parent context
            new_child[
                "text"
            ] = (
                f"{context}\n"
                f"{child['text']}"
            )

            contextualized_children.append(
                new_child
            )

        return contextualized_children