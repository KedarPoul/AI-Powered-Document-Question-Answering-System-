import json
import time
import matplotlib.pyplot as plt
from collections import Counter

from utils import qa_chain, retriever


# -----------------------------
# Utility functions
# -----------------------------


def exact_match(pred, gt):
    return int(pred.strip().lower() == gt.strip().lower())


def f1_score(pred, gt):
    pred_tokens = pred.lower().split()
    gt_tokens = gt.lower().split()

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)

    return 2 * precision * recall / (precision + recall)


# -----------------------------
# Main evaluation
# -----------------------------


def evaluate_system(qa_chain, retriever, k=3):
    with open("evaluation/questions.json") as f:
        data = json.load(f)

    em_scores = []
    f1_scores = []
    retrieval_hits = 0
    hallucinations = 0
    response_times = []

    for item in data:
        question = item["question"]
        ground_truth = item["answer"]

        start = time.time()

        # Retrieve documents
        docs = retriever.invoke(question)
        retrieved_texts = [doc.page_content for doc in docs[:k]]

        # Check retrieval hit
        if any(ground_truth.lower() in text.lower() for text in retrieved_texts):
            retrieval_hits += 1

        # Generate answer
        predicted = qa_chain(question)

        end = time.time()
        response_times.append(end - start)

        # Metrics
        em = exact_match(predicted, ground_truth)
        f1 = f1_score(predicted, ground_truth)

        em_scores.append(em)
        f1_scores.append(f1)

        if f1 < 0.3:
            hallucinations += 1

    total = len(data)

    results = {
        "Exact Match": sum(em_scores) / total,
        "F1 Score": sum(f1_scores) / total,
        "Retrieval Accuracy@k": retrieval_hits / total,
        "Hallucination Rate": hallucinations / total,
        "Avg Response Time (s)": sum(response_times) / total,
    }

    return results


def visualize_results(results):
    names = list(results.keys())
    values = list(results.values())

    plt.figure(figsize=(10, 5))
    plt.bar(names, values)
    plt.xticks(rotation=30)
    plt.title("Document Q&A System Evaluation Metrics")
    plt.tight_layout()
    plt.show()


# Example usage
# qa_chain -> your QA function
# retriever -> vectorstore.as_retriever()

results = evaluate_system(qa_chain, retriever, k=3)
print(results)

visualize_results(results)
