class RetrievalEvaluator:

    def hit_at_k(
        self,
        results,
        relevant_ids,
        k 
    ):

        retrieved_ids = [

            result["id"]

            for result in results[:k] 

        ]

        return int(
            any(
                doc_id in relevant_ids
                for doc_id in retrieved_ids 
            )
        )


    def recall_at_k(
        self,
        results,
        relevant_ids,
        k
    ):

        retrieved_ids = {

            result["id"]

            for result in results[:k]
        }

        relevant_ids = set(
            relevant_ids 
        )

        if not relevant_ids:
            return 0.0 

        return (
            len(
                retrieved_ids 
                & relevant_ids 
            )
            /
            len(relevant_ids)
        )


    def precision_at_k(
        self,
        results,
        relevant_ids,
        k 
    ):

        retrieved_ids = [

            result["id"]

            for result in results[:k]

        ]

        if not retrieved_ids:
            return 0.0

        relevant_count = sum(

            doc_id in relevant_ids 

            for doc_id in retrieved_ids 

        )

        return (
            relevant_count 
            /
            len(retrieved_ids)
        )

    
    def reciprocal_rank(
        self,
        results,
        relevant_ids 
    ):

        relevant_ids = set(
            relevant_ids 
        )

        for rank, result in enumerate(
            results,
            start=1
        ):

            if result["id"] in relevant_ids:

                return 1.0/rank 

        return 0.0