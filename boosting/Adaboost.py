import pandas as pd
import numpy as np
from tree.descision_tree import DecisionTree

class AdaBoost:
    """
    base on decision tree
    """

    def __init__(self, n_estimators=20, max_depth=-1, min_bins_sample=3):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_bins_sample = min_bins_sample
        self.alphas = []
        self.estimators = []

    def fit(self, x_df, y):
        if isinstance(y, pd.DataFrame):
            label_column = list(y.columns)[0]
            y = y[label_column]

        weights = np.ones(len(x_df)) / len(x_df)
        combine_pred = pd.Series(np.zeros(y.shape))
        best_accuracy = 0.0
        for i in range(self.n_estimators):
            print('estimator {}--'.format(i+1), end='')
            estimator = DecisionTree(self.max_depth, self.min_bins_sample)
            estimator.fit(x_df, y, weights)
            pred = estimator.predict(x_df)
            erate = weights[pred != y].sum()
            alpha_i = np.log((1-erate)/erate)/2
            weights = weights * np.exp(alpha_i * (y!= pred))
            weights = weights / weights.sum()

            self.estimators.append(estimator)
            self.alphas.append(alpha_i)

            combine_pred = combine_pred + pred.map({0: -1, 1: 1}) * alpha_i #映射为了标签可结合性
            accuracy = np.sum((combine_pred >= 0) == y) / len(y)

            print('error rate: {}'.format(erate))
            print('current all estimators accuracy is {}, best accuracy is {} '.format(accuracy, best_accuracy))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                print('Improved!')
                print('-' * 30)
            elif accuracy <= best_accuracy and i < self.n_estimators:
                print('Not Improved!')
                print('-' * 30)
            else:
                print('Not Improved and n_estimator>{} ! stop training'.format(self.n_estimators))

        print('=' * 50)

    def predict(self, x_df):
        preds = np.zeros(len(x_df))
        for i, alpha in enumerate(self.alphas):
            preds += alpha * self.estimators[i].predict(x_df).map({0: -1, 1: 1})
        return (preds >= 0).astype(int)





