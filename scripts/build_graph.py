from src.rag.graph.neo4j_store import (
    Neo4jGraphStore
)

graph = Neo4jGraphStore()

relationships = [

    (
        "Service A",
        "depends_on",
        "Database B"
    ),

    (
        "Service B",
        "hosted_on",
        "Server C"
    ),

    (
        "Service C",
        "located_in",
        "Region D"
    ),

    (
        "Service A",
        "owned_by",
        "Company X"
    ),

    (
        "Company X",
        "operates_in",
        "Technology"
    )
]

for (
    source,
    relation,
    target
) in relationships:

    graph.add_relationship(
        source=source,
        relation=relation,
        target=target
    )

graph.close()

print(
    "Graph created successfully."
)