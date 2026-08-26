from neo4j import GraphDatabase

class Neo4jGraphStore:

    def __init__(
        self,
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password"
    ):

        self.driver = GraphDatabase.driver(
            uri,
            auth=(
                username,
                password 
            )
        )

    
    def close(self):

        self.driver.close()

    
    def add_relationship(
        self,
        source,
        relation,
        target
    ):

        query = """
        MERGE (a:Entity {name: $source})

        MERGE (b:Entity {name: $target})

        MERGE (
            a
        )-[r:RELATED {type: $relation}]->(
            b
        )

        RETURN a, r, b
        """

        with self.driver.session() as session:

            session.run(
                query,
                source=source,
                relation=relation,
                target=target 
            )