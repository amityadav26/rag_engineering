class Neo4jGraphRetriever:

    def __init__(
        self,
        graph_store
    ):

        self.graph_store = graph_store 


    def retrieve(
        self,
        entity,
        depth=2
    ):

        query = """
        MATCH path = 
        (start:Entity {name: $entity})
        -[*1..2]-
        (connected)

        RETURN path
        """

        with self.graph_store.driver.session() as session:

            result = session.run(
                query,
                entity=entity 
            )

            return [
                record["path"]
                for record in result 
            ]