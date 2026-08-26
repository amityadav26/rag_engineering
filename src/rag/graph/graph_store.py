# Graph storage

class GraphStore:

    def __init__(self):

        self.nodes = {}
        self.edges = []

    
    def add_node(
        self,
        node_id,
        node_type=None,
        properties=None 
    ):

        self.nodes[node_id] = {
            "id": node_id,
            "type": node_type,
            "properties": properties or {}
        }

    
    def add_edge(
        self,
        source,
        relation,
        target
    ):

        self.edges.append({
            "source": source,
            "relation": relation,
            "target": target 
        })

    
    def get_neighbors(
        self,
        node_id
    ):

        results = []

        for edge in self.edges:

            if edge["source"] == node_id:

                results.append(edge)

            elif edge["target"] == node_id:

                results.append(edge)

        return results 