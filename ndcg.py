import numpy as np

def dcg_at_k(relevance_scores, k):
    """
    Compute the Discounted Cumulative Gain (DCG) at rank position k.

    Parameters:
    relevance_scores (list or numpy array): A list of relevance scores for the ranked items.
    k (int): The rank position up to which to calculate DCG.

    Returns:
    float: The DCG score at position k. Higher values mean better ranking.
    """
    relevance_scores = np.array(relevance_scores)[:k]
    
    # DCG = Σ (2^rel_i - 1) / log2(i + 1)
    dcg = np.sum((2 ** relevance_scores - 1) / np.log2(np.arange(1, len(relevance_scores) + 1) + 1))
    return dcg

def ndcg_at_k(predicted_relevance, ideal_relevance, k):
    """
    Compute the Normalized Discounted Cumulative Gain (NDCG) at rank position k.

    Parameters:
    predicted_relevance (list or numpy array): Relevance scores of the predicted ranking.
    ideal_relevance (list or numpy array): Relevance scores of the ideal ranking.
    k (int): The rank position up to which to calculate NDCG.

    Returns:
    float: The NDCG score at position k, which is DCG normalized by the ideal DCG.
    """
    dcg = dcg_at_k(predicted_relevance, k)
    idcg = dcg_at_k(ideal_relevance, k)  # Ideal DCG (IDCG) is calculated from the ideal ranking
    
    # Normalize DCG by the ideal DCG
    if idcg == 0:
        return 0.0
    return dcg / idcg

def compute_relevance_scores(predicted_ranking, ideal_ranking):
    """
    Compute the relevance scores for the predicted and ideal rankings.

    Parameters:
    predicted_ranking (list): A list of predicted item IDs in the order they were predicted.
    ideal_ranking (list): A list of ideal item IDs in the order they should have been ranked.

    Returns:
    tuple: (predicted_relevance, ideal_relevance)
    """
    # Initialize variables
    predicted_relevance = []
    ideal_index = 0  # Pointer to track the position in the ideal ranking
    last_matched_position = 0  # Track the position of the last matched item

    # Process each item in the predicted ranking
    for predicted_item in predicted_ranking:
        if ideal_index >= len(ideal_ranking):
            # If all ideal items have been matched, the remaining predicted items get 0 relevance
            predicted_relevance.append(0)
            continue
        
        match_found = False
        
        # Check for a match in the ideal ranking
        while ideal_index < len(ideal_ranking):
            if ideal_ranking[ideal_index] == predicted_item:
                predicted_relevance.append(1)
                match_found = True
                last_matched_position = ideal_index  # Update the last matched position
                ideal_index += 1  # Move to the next position for the next search
                break
            ideal_index += 1
        
        if not match_found:
            predicted_relevance.append(0)
            # Reset ideal_index to the position of the last matched item for the next predicted item
            ideal_index = last_matched_position + 1
    
    # Ideal relevance is always 1 for items in the ideal ranking
    ideal_relevance = [1] * len(predicted_ranking)
    
    return predicted_relevance, ideal_relevance

def main():
    # Example rankings
    predicted_ranking = [1, 2, 6, 7, 4]
    ideal_ranking = [1, 8, 2, 3, 4]
    
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
