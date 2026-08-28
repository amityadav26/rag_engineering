import json

from src.rag.evaluation.retrieval_evaluator import (
    RetrievalEvaluator
)


def load_evalauation_dataset(
    path 
):

    with open(
        path,
        "r",
        encoding="utf-8" 
    ) as file:

        return json.load(file)

class RetrieverEvaluatorRunner:


    def __init__(
        self,
        evaluator 
    ):

        self.evaluator = evaluator 

    
    def evaluate(
        self,
        retriever,
        evaluation_dataset,
        top_k=5 
    ):

        hit_scores = []
        recall_scores = []
        precision_scores = []
        mrr_scores = []

        for item in evaluation_dataset:

            query = item["question"]
            relevant_ids = item["relevant_ids"]

            results = retriever.retrieve(
                query,
                top_k=top_k 
            )

            hit = self.evaluator.hit_at_k(
                results,
                relevant_ids,
                top_k 
            )

            recall = self.evaluator.recall_at_k(
                results,
                relevant_ids,
                top_k 
            )

            precision = self.evaluator.precision_at_k(
                results,
                relevant_ids,
                top_k 
            )

            mrr = self.evaluator.reciprocal_rank(
                results,
                relevant_ids,
            )

            hit_scores.append(hit)
            recall_scores.append(recall)
            precision_scores.append(precision)
            mrr_scores.append(mrr)

        
        total = len(
            evaluation_dataset 
        )

        return {
            "Hit@k":
            sum(hit_scores) / total,

            "Recall@k":
            sum(recall_scores) / total,

            "Percision@k":
            sum(precision_scores) / total,

            "MRR":
            sum(mrr_scores) / total 
        
        }
