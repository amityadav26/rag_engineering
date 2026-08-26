from neo4j import GraphDatabase

# Clear all data from Neo4j
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password")
)

with driver.session() as session:
    # Delete all nodes and relationships
    result = session.run("MATCH (n) DETACH DELETE n")
    print("✓ Cleared all data from Neo4j")

driver.close()