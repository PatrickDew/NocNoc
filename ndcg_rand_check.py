import numpy as np
import random

def dcg_at_k(relevance_scores, k):
    """
    Compute the Discounted Cumulative Gain (DCG) at rank position k.
    """
    relevance_scores = np.array(relevance_scores)[:k]
    dcg = np.sum((2 ** relevance_scores - 1) / np.log2(np.arange(1, len(relevance_scores) + 1) + 1))
    return dcg

def ndcg_at_k(predicted_relevance, ideal_relevance, k):
    """
    Compute the Normalized Discounted Cumulative Gain (NDCG) at rank position k.
    """
    dcg = dcg_at_k(predicted_relevance, k)
    idcg = dcg_at_k(ideal_relevance, k)
    
    if idcg == 0:
        return 0.0
    return dcg / idcg

def compute_relevance_scores(predicted_ranking, ideal_ranking):
    """
    Compute the relevance scores for the predicted and ideal rankings.
    """
    predicted_relevance = []
    ideal_index = 0
    last_matched_position = 0

    for predicted_item in predicted_ranking:
        if ideal_index >= len(ideal_ranking):
            predicted_relevance.append(0)
            continue
        
        match_found = False
        
        while ideal_index < len(ideal_ranking):
            if ideal_ranking[ideal_index] == predicted_item:
                predicted_relevance.append(1)
                match_found = True
                last_matched_position = ideal_index
                ideal_index += 1
                break
            ideal_index += 1
        
        if not match_found:
            predicted_relevance.append(0)
            ideal_index = last_matched_position + 1
    
    ideal_relevance = [1] * len(predicted_ranking)
    
    return predicted_relevance, ideal_relevance

def generate_random_ranking(length, max_value):
    """
    Generate a random ranking of specified length with values up to max_value.
    """
    return random.sample(range(max_value), length)

def main():
    # Parameters for generating random rankings
    length = 5
    max_value = 10
    
    # Generate random rankings
    predicted_ranking = generate_random_ranking(length, max_value)
    ideal_ranking = generate_random_ranking(length, max_value)
    
    print(f"Predicted Ranking: {predicted_ranking}")
    print(f"Ideal Ranking: {ideal_ranking}")
    
    # Compute relevance scores
    predicted_relevance, ideal_relevance = compute_relevance_scores(predicted_ranking, ideal_ranking)
    
    k = len(predicted_ranking)
    
    # Compute NDCG
    ndcg_score = ndcg_at_k(predicted_relevance, ideal_relevance, k)
    
    print(f"Predicted Relevance: {predicted_relevance}")
    print(f"Ideal Relevance: {ideal_relevance}")
    print(f"NDCG score at k={k}: {ndcg_score:.4f}")

if __name__ == "__main__":
    main()
