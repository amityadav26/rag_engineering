class GraphRetriever:

    def __init__(
        self,
        graph_store
    ):

        self.graph_store = graph_store


    def retrieve(
        self,
        entity,
        depth=1
    ):

        visited = set()

        current = [
            entity
        ]

        results = []

        for _ in range(depth):

            next_nodes = []

            for node in current:

                if node in visited:

                    continue

                visited.add(node)

                neighbors = (
                    self.graph_store
                    .get_neighbors(node)
                )

                for edge in neighbors:

                    results.append(edge)

                    if (
                        edge["source"] == node
                    ):

                        next_nodes.append(
                            edge["target"]
                        )

                    else:

                        next_nodes.append(
                            edge["source"]
                        )

            current = next_nodes

        return results