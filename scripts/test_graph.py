from src.rag.graph.graph_store import (
    GraphStore
)

from src.rag.graph.entity_extractor import (
    EntityExtractor
)

from src.rag.graph.graph_retriever import (
    GraphRetriever
)


graph = GraphStore()

extractor = EntityExtractor()

relationships = extractor.extract()


for relationship in relationships:

    graph.add_node(
        relationship["source"]
    )

    graph.add_node(
        relationship["target"]
    )

    graph.add_edge(
        relationship["source"],
        relationship["relation"],
        relationship["target"]
    )


retriever = GraphRetriever(
    graph 
)

results = retriever.retrieve(
    entity="Service A",
    depth=2 
)

print(
    "\n========== GRAPH RESULTS =========="  
)

for result in results:

    print(

        result["source"],

        "---",

        result["relation"],

        "--->",

        result["target"]
    )