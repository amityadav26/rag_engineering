**We will be learning and implementing the different architectures.**

**Not only the Framework , we will learning without any abstractions.**

**Learning different techniques - chunking, splitters and making the embeedings.**







| Situation                                        | Preferred approach  |
| ------------------------------------------------ | ------------------- |
| Simple semantic question                         | Dense               |
| Exact technical terms                            | Sparse/BM25         |
| Semantic + exact terminology                     | Hybrid              |
| Important ranking accuracy                       | Reranker            |
| User's query is ambiguous                        | Query Expansion     |
| Different formulations may help                  | Multi-Query         |
| Query is short/poorly formulated                 | HyDE                |
| Small chunk needs larger context                 | Parent-Child        |
| Chunk lacks surrounding meaning                  | Contextual Chunking |
| Retrieved context is too large                   | Compression         |
| Retrieval quality is poor                        | Corrective RAG      |
| System should decide whether retrieval is needed | Self-RAG            |
| Different knowledge sources exist                | Routing             |
| Relationship/multi-hop question                  | Graph RAG           |
| Complex autonomous retrieval                     | Agentic RAG         |
