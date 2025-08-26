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

    tuples = [
        {'a': 3, 'b': 3, 'c': 3, 'sum': 12, 'd': 3, 'e': 3},
        {'a': 4, 'b': 4, 'c': 4, 'sum': 16, 'd': 4, 'e': 4},
        {'a': 5, 'b': 5, 'c': 5, 'sum': 20, 'd': 5, 'e': 5},
        {'a': 6, 'b': 6, 'c': 6, 'sum': 24, 'd': 6, 'e': 6},
        {'a': 7, 'b': 7, 'c': 7, 'sum': 28, 'd': 7, 'e': 7},
        {'a': 8, 'b': 8, 'c': 8, 'sum': 32, 'd': 8, 'e': 8},
        {'a': 9, 'b': 9, 'c': 9, 'sum': 36, 'd': 9, 'e': 9}
    ]
    weights = np.array([1, 1, 1, 1, 1, 1, 1])
    total_weight = np.sum(weights)
    result = weighted_median_linear(tuples, weights, total_weight)
    print(result)  # Expected output: {'a': 6, 'b': 6, 'c': 6, 'sum': 24, 'd': 6, 'e': 6},

if __name__ == "__main__":
    test_weighted_median()