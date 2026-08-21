class QueryExpander:

    def __init__(self):
        pass

    def expand(self, query, terms):

        expanded_query = (
            query + " ".join(terms)
        )

        return expanded_query 