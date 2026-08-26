from src.rag.graph.neo4j_store import (
    Neo4jGraphStore
)

from src.rag.graph.neo4j_retriever import (
    Neo4jGraphRetriever
)

graph = Neo4jGraphStore()

retriever = Neo4jGraphRetriever(
    graph 
)

results = retriever.retrieve(
    entity="Service A",
    depth=2
)

print(
    "\n======== GRAPH RETRIEVAL ==========="
)

for path in results:

    print(path)

graph.close()