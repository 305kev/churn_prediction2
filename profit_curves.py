import numpy as np
import matplotlib.pyplot as plt


def standard_confusion_matrix(y_true, y_pred):
    """Return the confusion matrix in standard order.

    INPUTS: y_pred, y_true: numpy arrays with binary classification.
    RETURNS: cm: 2 x 2 numpy array - the standard confusion matrix.
    """
    tp = np.sum((y_true == y_pred) & (y_true == 1))
    fp = np.sum((y_true != y_pred) & (y_true == 0))
    fn = np.sum((y_true != y_pred) & (y_true == 1))
    tn = np.sum((y_true == y_pred) & (y_true == 0))
    return np.array([[tp, fp], [fn, tn]])


def profit_curve(cost_benefit, predicted_probs, labels):
    """Return the profit curve for a given set of classifier results.

    INPUTS: cost_benefit: 2 x 2 numpy array - the cost-benefit matrix
            predicted probs: numpy array, the model predicted probabilities
            labels: numpy array, the true labels for the classification
    RETURNS: profits: numpy array, the profit curve values
             thresh: numpy array, the corresponding threshold values
    """
    # Do some sorting
    idx = predicted_probs.argsort()
    sort_probs = predicted_probs[idx]
    sort_labels = labels[idx]
    thresholds = np.append(sort_probs, 1)

    # Do some counting
    y_pred = np.ones(labels.shape[0])
    profit = np.zeros(thresholds.shape[0])
    for index, thresh in enumerate(thresholds):
        cm = standard_confusion_matrix(sort_labels, y_pred)
        profit[index] = np.sum(cm * cost_benefit) / labels.shape[0]
        if index < labels.shape[0]:
            y_pred[index] = 0

    return thresholds, profit


def plot_profit_curve(model, cost_benefit, X_train, X_test, y_train, y_test,
                      show=True):
    """Plot the profit curve for a given model and cost-benefit matrix.

    Return the values that correspond to the maximum profit.
    INPUTS: model - a SKLearn model instance
            cost_benefit - a 2 x 2 numpy array, the cost-benefit matrix
            X_Train, X_test - numpy arrays, the feature matrices
            y_train, y_test - numpy arrays, the target classification vectors
            plot - Boolean, whether to show the plot or not
    RETURN: max_profits - numpy array with max value of profits, and the
                          corresponding percentile and threshold
    """

    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    t, profits = profit_curve(cost_benefit, probs[:, 1], y_test)
    midx = np.argmax(profits)
    percentages = np.arange(0, 100, 100. / len(profits))
    if show:
        plt.plot(percentages, profits, label=model.__class__.__name__)
        plt.title("Profit Curve")
        plt.xlabel("Percentage of test instances (decreasing by score)")
        plt.ylabel("Profit")
    return np.array([profits[midx], t[midx], percentages[midx]])
