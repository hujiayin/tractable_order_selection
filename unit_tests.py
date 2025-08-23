from select_k.Selection_Sum import weighted_median_linear
import numpy as np

def test_weighted_median():
    tuples = [
        {'sum': 1, 'other': 10},
        {'sum': 3, 'other': 30},
        {'sum': 2, 'other': 20},
        {'sum': 5, 'other': 20},
        {'sum': 4, 'other': 1},
    ]
    weights = np.array([1, 5, 5, 1, 1])
    total_weight = np.sum(weights)
    result = weighted_median_linear(tuples, weights, total_weight)
    print(result)  # Expected output: {'sum': 3, 'other': 30}

    tuples = [
        {'sum': 1, 'other': 10},
        {'sum': 3, 'other': 30},
        {'sum': 2, 'other': 20},
        {'sum': 5, 'other': 20},
        {'sum': 4, 'other': 1},
    ]
    weights = np.array([1, 1, 1, 10, 1])
    total_weight = np.sum(weights)
    result = weighted_median_linear(tuples, weights, total_weight)
    print(result)  # Expected output: {'sum': 5, 'other': 20}

if __name__ == "__main__":
    test_weighted_median()