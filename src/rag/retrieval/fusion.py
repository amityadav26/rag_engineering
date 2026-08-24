def reciprocal_rank_fusion(
    rankings,
    k=60,
    top_n=10
):

    scores = {}

    documents = {}

    for ranking in rankings:

        for rank, document in enumerate(
            ranking,
            start=1
        ):

            document_id = document["id"]

            if document_id not in scores:

                scores[document_id] = 0

                documents[
                    document_id
                ] = document

            scores[document_id] += (
                1 / (k + rank)
            )


    fused_results = []

    for document_id, score in scores.items():

        document = documents[
            document_id
        ]

        fused_results.append(
            {

                "id": document["id"],

                "text": document["text"],

                "parent_id": document.get(
                    "parent_id"
                ),

                "metadata": document.get(
                    "metadata",
                    {}
                ),

                "rrf_score": score
            }
        )


    fused_results.sort(
        key=lambda document: document["rrf_score"],
        reverse=True
    )

    return fused_results[:top_n]