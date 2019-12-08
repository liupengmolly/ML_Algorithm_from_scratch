import numpy as np
import pandas as pd


class DecisionTree:
    """
    暂时先实现分类
    """
    def __init__(self, max_depth=-1, min_bins_sample=3):
        self.max_depth = max_depth
        self.min_bins_sample = min_bins_sample

        self.split_feature = None
        self.split_value = None
        self.left_node = None
        self.right_node = None
        self.label = None

    def split_score(self, values, y, value, weights):
        p0 = ((y == 0)*weights).sum()
        p1 = ((y == 1)*weights).sum()
        gini = 1 - p0**2 - p1**2

        left_y = y[values <= value]
        left_weights = weights[values <= value]
        left_weight = left_weights.sum()
        left_values_p0 = ((left_y == 0) * left_weights).sum() / left_weight
        left_values_p1 = ((left_y == 1) * left_weights).sum() / left_weight
        gini_left = left_weight * (1 - left_values_p0**2 - left_values_p1**2)

        right_y = y[values > value]
        right_weights = weights[values > value]
        right_weight = right_weights.sum()
        right_values_p0 = ((right_y == 0) * right_weights).sum() / right_weight
        right_values_p1 = ((right_y == 1) * right_weights).sum() / right_weight
        gini_right = right_weight * (1 - right_values_p0**2 - right_values_p1**2)

        return gini - gini_left - gini_right

    def find_best_split(self, x_df, y, weights):
        """

        :param x_df: dataframe, nxf, f表示特征的数量
        :param y: dataframe, nx1
        :param weights: dataframe, nx1
        :return:
        """
        max_score = 0
        split_feature = None
        split_value = None

        for feature in x_df.columns:
            values = x_df[feature]
            unique_values = pd.Series(list(set(x_df[feature].values)))
            for value in unique_values:
                score = self.split_score(values, y, value, weights)
                if score > max_score:
                    split_feature = feature
                    split_value = value
                    max_score = score

        return split_feature, split_value

    def fit(self, x_df, y, weights=None):
        if weights is None:
            weights = np.ones(len(y))/len(y)
        if isinstance(y, pd.DataFrame):
            label_column = list(y.columns)[0]
            y = y[label_column]

        sample_num = len(x_df)
        if self.max_depth == 0 or sample_num <= self.min_bins_sample:
            self.label = y.value_counts().index[0]
        else:
            self.split_feature, self.split_value = self.find_best_split(x_df, y, weights)

            if self.split_feature is not None:
                x_left = x_df[x_df[self.split_feature] <= self.split_value]
                x_right = x_df[x_df[self.split_feature] > self.split_value]
                y_left = y[x_df[self.split_feature] <= self.split_value]
                y_right = y[x_df[self.split_feature] > self.split_value]
                self.max_depth = (self.max_depth - 1) if self.max_depth>0 else -1
                self.left_node = DecisionTree(self.max_depth, self.min_bins_sample)
                self.right_node = DecisionTree(self.max_depth, self.min_bins_sample)
                self.left_node.fit(x_left, y_left)
                self.right_node.fit(x_right, y_right)
            else:
                self.label = y.value_counts().index[0]

    def predict_row(self, x_row):
        if self.label is not None:
            return self.label
        else:
            row_value = x_row[self.split_feature]
            if row_value > self.split_value:
                return self.right_node.predict_row(x_row)
            else:
                return self.left_node.predict_row(x_row)

    def predict(self, x_df):
        predict = []
        for i in range(len(x_df)):
            predict.append(self.predict_row(x_df.iloc[i]))

        return pd.Series(predict)





