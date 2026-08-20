import json 
from src.rag.retrieval.sparse import BM25Retriever


def load_documents(path):

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():

    documents = load_documents(
        "data/documents.json"
    )

    retriever = BM25Retriever(
        documents=documents 
    )

    query = "What does ERR_CONNECTION_RESET mean?"

    results = retriever.retrieve(
        query=query, top_k=2
    )

    for result in results:
        print(
            result["id"],
            result["score"],
            result["text"]  
        )


if __name__ == "__main__":
    main()