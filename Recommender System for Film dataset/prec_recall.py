# Utility file implements Precision / Recall computation. Based on examples/precision_recall_at_k.py
# Note that non-standard test set methodology is used. Only ratings on the test set items are computed
# so it is not a true measure of these values.
# NOT FOR SERIOUS EXPERIMENTAL USE

from collections import defaultdict
from scipy import mean
from pandas import Series, DataFrame


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user."""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()

    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Original code returns 1 in these cases which is definitely WRONG.
        if n_rel_and_rec_k == 0:
            precisions[uid] = 0
            recalls[uid] = 0
        else:
            # Precision@K: Proportion of recommended items that are relevant
            # n_rec_k cannot be 0 if n_rel_and_rec_k > 0
            precisions[uid] = (1.0 * n_rel_and_rec_k) / n_rec_k

            # Recall@K: Proportion of relevant items that are recommended
            # n_rel cannot be 0 if n_rel_and_rec_k > 0
            recalls[uid] = (1.0 * n_rel_and_rec_k) / n_rel

    return precisions, recalls


def pr_eval(algo, data, cv, n=10, threshold=3.5):
    """Runs a precision / recall evaluation

    :param algo: Prediction algorithm
    :param data: Dataset object
    :param cv: CV object
    :param n: Size of list to evaluate
    :param threshold: Threshold for an item to be considered "good"
    :return: Pandas dataframe with the results by fold.

    Note there is a dramatic interaction between the threshold and the results if the
    threshold is near the mean rating for the data set. Items that can't be predicted
    with get the mean value as a score and if that is above the threshold, they will be
    consider good predictions by the evaluation measure.
    """
    results = {}
    fold_count = 0

    for trainset, testset in cv.split(data):
        fold_count += 1
        algo.fit(trainset)
        predictions = algo.test(testset)

        precisions, recalls = precision_recall_at_k(predictions, k=n, threshold=threshold)
        result_summary = [mean(list(metric.values())) for metric in [precisions, recalls]]
        results['Fold {}'.format(fold_count)] = Series(result_summary, ['Precision', 'Recall'])

    print("Precision-Recall")
    print(results)

    return DataFrame(results)
