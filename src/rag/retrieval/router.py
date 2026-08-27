from enum import Enum

class RetrievalStrategy(Enum):

    DENSE = "dense"

    SPARSE = "sparse"

    HYBRID = "hybrid"

    GRAPH = "graph"

    MULTI_QUERY = "multi_query"

    HYDE = "hyde"


class RetrievalRouter:

    def route(
        self,
        query 
    ):

        query_lower = (
            query.lower()
        )


        graph_keywords = [

            "depends on",
            "related to",
            "connected to",
            "owned by",
            "hosted on",
            "which server",
            "which system"
        ]

        exact_keywords = [
            
            "error",
            "exception",
            "error code",
            "status code"
        ]

        if any(
            keyword in query_lower
            for keyword in graph_keywords
        ):

            return RetrievalStrategy.GRAPH 


        if any(
            keyword in query_lower
            for keyword in exact_keywords
        ):

            return RetrievalStrategy.HYBRID 


        return RetrievalStrategy.DENSE