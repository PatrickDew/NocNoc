# NocNoc Hackathon 2024 - Ranking Evaluation Metrics

This repository contains a Python implementation of the Discounted Cumulative Gain (DCG) and Normalized Discounted Cumulative Gain (NDCG) metrics for evaluating the relevance of ranking systems. The solution also includes utility functions to compute relevance scores between predicted and ideal rankings and generate random rankings for testing purposes.

## Getting Started

### Prerequisites

Make sure you have [Python 3.x](https://www.python.org/downloads/) installed on your system.

### Installation

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/PatrickDew/NocNoc.git
    ```

2. Navigate to the project directory:
    ```bash
    cd NocNoc
    ```

3. Install the required dependency:
    ```bash
    pip install numpy
    ```

## Usage

To run the ranking evaluation script, execute the following commands:

1. Compute the relevance scores and evaluate NDCG:
    ```bash
    python ndcg.py
    ```

2. Generate and check random rankings:
    ```bash
    python ndcg_rand_check.py
    ```

## Example

Given the following input:

    Predicted Ranking: [3, 1, 7, 5, 2]
    Ideal Ranking: [1, 3, 7, 5, 2]
    Predicted Relevance: [0, 1, 1, 1, 1]
    Ideal Relevance: [1, 1, 1, 1, 1]
    

The NDCG score at `k=5` would be:

    
    NDCG score at k=5: 0.8562
    

## Functions Overview

- `dcg_at_k(relevance_scores, k)`: Computes the DCG score at a given rank `k`.
- `ndcg_at_k(predicted_relevance, ideal_relevance, k)`: Computes the NDCG score normalized by the ideal ranking.
- `compute_relevance_scores(predicted_ranking, ideal_ranking)`: Converts rankings into relevance scores.
- `generate_random_ranking(length, max_value)`: Generates random rankings for testing.

