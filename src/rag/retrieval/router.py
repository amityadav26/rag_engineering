class RetrievalRouter:

    
    def route(
        self,
        query 
    ):

        query_lower = query.lower()

        if (
            "exact" in query_lower 
            or "error code" in query_lower 
        ):

            return "sparse" 

        if (
            "relationship" in query_lower 
            or "connected" in query_lower 
        ):

            return "graph" 

        
        return "hybrid"