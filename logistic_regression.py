import numpy as np
import joblib


def sigmoid(z):
    op = 1/(1 + np.exp(-z))
    return op


def predict(X, weights):
    z = np.dot(X, weights)
    return sigmoid(z)


def gradient_descent(X, y, weights, learning_rate, num_epochs):
    m = len(y)
    for _ in range(num_epochs):
        y_pred = predict(X, weights)
        gradient = np.dot(X.T, (y_pred - y)) / m
        weights -= learning_rate * gradient
    return weights


def lr_calculate_metrics(y_true, y_pred, accuracy):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision_0 = tn / (tn + fn)
    recall_0 = tn / (tn + fp)
    f1_score_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0)

    precision_1 = tp / (tp + fp)
    recall_1 = tp / (tp + fn)
    f1_score_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1)

    macro_precision = (precision_0 + precision_1) / 2
    macro_recall = (recall_0 + recall_1) / 2
    macro_f1_score = (f1_score_0 + f1_score_1) / 2

    weighted_precision = (tp + tn) / (tp + tn + fp + fn)
    weighted_recall = (tp + tn) / (tp + tn + fp + fn)
    weighted_f1_score = (f1_score_0 * (tn + fn) + f1_score_1 * (tp + fp)) / (tp + tn + fp + fn)

    return {
        'accuracy': accuracy,
        'precision_0': precision_0,
        'precision_1': precision_1,
        'recall_0': recall_0,
        'recall_1': recall_1,
        'f1_score_0': f1_score_0,
        'f1_score_1': f1_score_1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1_score': macro_f1_score,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1_score': weighted_f1_score
    }


def lr_classification_report(metrics):
    print("Logistic Regression From Scratch Results:\n")
    print(f"{'Accuracy':<14} {metrics['accuracy']:.2f}\n")
    print(f"{'':<15}{'precision':<15}{'recall':<15}{'f1-score':<15}")
    print(f"{'class 0':<15}{metrics['precision_0']:.2f}{'':<15}{metrics['recall_0']:.2f}{'':<15}{metrics['f1_score_0']:.2f}")
    print(f"{'class 1':<15}{metrics['precision_1']:.2f}{'':<15}{metrics['recall_1']:.2f}{'':<15}{metrics['f1_score_1']:.2f}")
    print(f"{'macro avg':<15}{metrics['macro_precision']:.2f}{'':<15}{metrics['macro_recall']:.2f}{'':<15}{metrics['macro_f1_score']:.2f}")
    print(f"{'weighted avg':<15}{metrics['weighted_precision']:.2f}{'':<15}{metrics['weighted_recall']:.2f}{'':<15}{metrics['weighted_f1_score']:.2f}\n")


def logistic_regression(X_train, y_train, X_test, y_test):
    X_train_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_test_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]

    num_features = X_train_bias.shape[1]
    weights = np.zeros(num_features)

    learning_rate = 0.01
    num_epochs = 1000

    for _ in range(num_epochs):
        y_pred = predict(X_train_bias, weights)
        gradient = np.dot(X_train_bias.T, (y_pred - y_train)) / len(y_train)
        weights -= learning_rate * gradient

    y_pred = np.round(predict(X_test_bias, weights))

    accuracy = np.mean(y_test == y_pred)
    metrics = lr_calculate_metrics(y_test, y_pred, accuracy)

    return metrics, y_pred, weights


