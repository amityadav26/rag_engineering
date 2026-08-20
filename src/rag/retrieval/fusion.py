def reciprocal_rank_fusion(
    rankings,
    k=60,
    top_n=10
):

    rrf_scores = {}
    documents = {}

    for ranking in rankings:

        for rank, result in enumerate(
            ranking,
            start = 1
        ):
            doc_id = result["id"]

            if doc_id not in documents:
                documents[doc_id] = result 

            score = 1/ (k + rank)

            rrf_scores[doc_id] = (
                rrf_scores.get(doc_id, 0) + score 
            )


    fused_result = []

    for doc_id, score in rrf_score.items():

        result = documents[doc_id].copy()

        result["rrf_score"] = score 

        fused_result.append(result) 

    fused_result.sort(
        key=lambda x: x["rrf_score"],
        reverse=True 
    )

    return fused_result[:top_n]