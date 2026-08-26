
# Foe now just understand the architecture that how it works later using langgraph we will be swap the simple routing with the llm then it becomes the agentic RAG

class RAGAgent:

    def __init__(
        self,
        dense_tool,
        bm25_tool 
    ):

        self.dense_tool = dense_tool 
        self.bm25_tool = bm25_tool 


    def decide_tool(
        self,
        query 
    ):

        query_lower = query.lower()

        if(
            "exact" in query_lower
            or "error code" in query_lower 
        ):

            return "bm25"

        return "dense"


    def run(
        self,
        query
    ):

        tool_name = self.decide_tool(
            query 
        )

        if tool_name == "bm25":

            documents = (
                self.bm25_tool.run(
                    query
                )
            )

        else:

            documents = (
                self.dense_tool.run(
                    query
                )
            )

        return {
            "tool": tool_name,
            "documents": documents
        }