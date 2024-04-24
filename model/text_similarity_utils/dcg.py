# (C) Mathieu Blondel, November 2013
# License: BSD 3 clause

import numpy as np
import os

class nDCG:
    def __init__(self, dataset, n_queries, split, type, rank=25, relevance_methods=['spice']):
        self.rank = rank
        self.relevance_methods = relevance_methods
        # Please replace the absolute path to this project, be like: xxx/rehamot/outputs
        relevance_dir = os.path.join('/mnt/07b5edc6-c6ec-4c22-bd14-51ca54d8a7fc/YanSheng/rehamot/outputs', 'computed_relevances')
        relevance_filenames = [os.path.join(relevance_dir, '{}-{}-{}-{}.npy'.format(dataset, split, m, type))
                               for m in relevance_methods]
        self.relevances = [np.memmap(f, dtype=np.float32, mode='r') for f in relevance_filenames]
        for r in self.relevances:
            r.shape = (n_queries, -1)

    def compute_ndcg(self, query_id, sorted_indexes, retrieval='motion'):
        sorted_indexes_cut = sorted_indexes[:self.rank]
        # npts = self.relevances[0].shape[1] // 5
        if retrieval == 'motion':
            relevances = [r[query_id, :] for r in self.relevances]

        elif retrieval == 'sentence':
            return NotImplementedError

            query_base = npts * fold_index
            # sorted_indexes += npts * 5 * fold_index
            relevances = [r[fold_index * npts * 5 : (fold_index + 1) * npts * 5, query_base + query_id] for r in self.relevances]

        ndcg_scores = [ndcg_from_ranking(r, sorted_indexes_cut) for r in relevances]
        ndcgs = {k: v for k, v in zip(self.relevance_methods, ndcg_scores)}

        sorted_relevances = [r[sorted_indexes] for r in relevances]
        sorted_relevances = {k: v for k, v in zip(self.relevance_methods, sorted_relevances)}

        return ndcgs, sorted_relevances

    # def compute_dcg(self, query_id, sorted_img_indexes):
    #     sorted_img_indexes = sorted_img_indexes[:self.rank]
    #     dcg_score = dcg_from_ranking(self.relevances[query_id], sorted_img_indexes)
    #     return dcg_score



def ranking_precision_score(y_true, y_score, k=10):
    """Precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    # Divide by min(n_pos, k) such that the best achievable score is always 1.0.
    return float(n_relevant) / min(n_pos, k)


def average_precision_score(y_true, y_score, k=10):
    """Average precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    average precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1][:min(n_pos, k)]
    y_true = np.asarray(y_true)[order]

    score = 0
    for i in xrange(len(y_true)):
        if y_true[i] == pos_label:
            # Compute precision up to document i
            # i.e, percentage of relevant documents up to document i.
            prec = 0
            for j in xrange(0, i + 1):
                if y_true[j] == pos_label:
                    prec += 1.0
            prec /= (i + 1.0)
            score += prec

    if n_pos == 0:
        return 0

    return score / n_pos


def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best


# Alternative API.

def dcg_from_ranking(y_true, ranking, gains = "exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    ranking : array-like, shape = [k]
        Document indices, i.e.,
            ranking[0] is the index of top-ranked document,
            ranking[1] is the index of second-ranked document,
            ...
    k : int
        Rank.
    Returns
    -------
    DCG @k : float
    """
    y_true = np.asarray(y_true)
    ranking = np.asarray(ranking)
    rel = y_true[ranking]
    if gains == "exponential":
        gains = 2 ** rel - 1
    elif gains == "linear":
        gains = rel
    else:
        raise ValueError("Invalid gains option.")
    discounts = np.log2(np.arange(len(ranking)) + 2)
    return np.sum(gains / discounts)


def ndcg_from_ranking(y_true, ranking):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    ranking : array-like, shape = [k]
        Document indices, i.e.,
            ranking[0] is the index of top-ranked document,
            ranking[1] is the index of second-ranked document,
            ...
    k : int
        Rank.
    Returns
    -------
    NDCG @k : float
    """
    k = len(ranking)
    best_ranking = np.argsort(y_true)[::-1]
    best = dcg_from_ranking(y_true, best_ranking[:k])
    if best == 0:
        return 0
    return dcg_from_ranking(y_true, ranking) / best


if __name__ == '__main__':

    # Check that some rankings are better than others
    assert dcg_score([5, 3, 2], [2, 1, 0]) > dcg_score([4, 3, 2], [2, 1, 0])
    assert dcg_score([4, 3, 2], [2, 1, 0]) > dcg_score([1, 3, 2], [2, 1, 0])

    assert dcg_score([5, 3, 2], [2, 1, 0], k=2) > dcg_score([4, 3, 2], [2, 1, 0], k=2)
    assert dcg_score([4, 3, 2], [2, 1, 0], k=2) > dcg_score([1, 3, 2], [2, 1, 0], k=2)

    # Perfect rankings
    assert ndcg_score([5, 3, 2], [2, 1, 0]) == 1.0
    assert ndcg_score([2, 3, 5], [0, 1, 2]) == 1.0
    assert ndcg_from_ranking([5, 3, 2], [0, 1, 2]) == 1.0

    assert ndcg_score([5, 3, 2], [2, 1, 0], k=2) == 1.0
    assert ndcg_score([2, 3, 5], [0, 1, 2], k=2) == 1.0
    assert ndcg_from_ranking([5, 3, 2], [0, 1]) == 1.0

    # Check that sample order is irrelevant
    assert dcg_score([5, 3, 2], [2, 1, 0]) == dcg_score([2, 3, 5], [0, 1, 2])

    assert dcg_score([5, 3, 2], [2, 1, 0], k=2) == dcg_score([2, 3, 5], [0, 1, 2], k=2)

    # Check equivalence between two interfaces.
    assert dcg_score([5, 3, 2], [2, 1, 0]) == dcg_from_ranking([5, 3, 2], [0, 1, 2])
    assert dcg_score([1, 3, 2], [2, 1, 0]) == dcg_from_ranking([1, 3, 2], [0, 1, 2])
    assert dcg_score([1, 3, 2], [0, 2, 1]) == dcg_from_ranking([1, 3, 2], [1, 2, 0])
    assert ndcg_score([1, 3, 2], [2, 1, 0]) == ndcg_from_ranking([1, 3, 2], [0, 1, 2])

    assert dcg_score([5, 3, 2], [2, 1, 0], k=2) == dcg_from_ranking([5, 3, 2], [0, 1])
    assert dcg_score([1, 3, 2], [2, 1, 0], k=2) == dcg_from_ranking([1, 3, 2], [0, 1])
    assert dcg_score([1, 3, 2], [0, 2, 1], k=2) == dcg_from_ranking([1, 3, 2], [1, 2])
    assert ndcg_score([1, 3, 2], [2, 1, 0], k=2) == \
            ndcg_from_ranking([1, 3, 2], [0, 1])

    # Precision
    assert ranking_precision_score([1, 1, 0], [3, 2, 1], k=2) == 1.0
    assert ranking_precision_score([1, 1, 0], [1, 0, 0.5], k=2) == 0.5
    assert ranking_precision_score([1, 1, 0], [3, 2, 1], k=3) == \
            ranking_precision_score([1, 1, 0], [1, 0, 0.5], k=3)

    # Average precision
    from sklearn.metrics import average_precision_score as ap
    assert average_precision_score([1, 1, 0], [3, 2, 1]) == ap([1, 1, 0], [3, 2, 1])
    assert average_precision_score([1, 1, 0], [3, 1, 0]) == ap([1, 1, 0], [3, 1, 0])