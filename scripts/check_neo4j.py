from neo4j import GraphDatabase

# Try to connect to Neo4j
try:
    driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "password")
    )
    
    print("✓ Connected to Neo4j successfully")
    
    # Check if data exists
    with driver.session() as session:
        # Count nodes
        result = session.run("MATCH (n) RETURN count(n) as count")
        count = result.single()["count"]
        print(f"✓ Total nodes in database: {count}")
        
        # List all entities
        result = session.run("MATCH (n:Entity) RETURN n.name as name LIMIT 10")
        entities = [record["name"] for record in result]
        print(f"✓ Sample entities: {entities}")
        
        # Check if Service A exists
        result = session.run("MATCH (n:Entity {name: 'Service A'}) RETURN n")
        service_a = result.single()
        if service_a:
            print("✓ Service A found in database")
        else:
            print("✗ Service A NOT found in database")
    
    driver.close()
    
except Exception as e:
    print(f"✗ Connection failed: {e}")
    print("Make sure Neo4j is running at bolt://localhost:7687")
    print("Default credentials: username='neo4j', password='password'")