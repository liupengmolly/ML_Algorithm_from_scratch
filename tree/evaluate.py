import pandas as pd
import numpy as np
import time
from tree.descision_tree import DecisionTree
from sklearn.model_selection import KFold

if __name__ == '__main__':
    df = pd.read_csv("../data/titanic/processed_train.csv")
    columns = list(df.columns)
    features = columns
    features.remove('Survived')

    kf = KFold(n_splits=5)
    accuracy = 0
    for trn_idx, val_idx in kf.split(df):
        train = df.iloc[trn_idx]
        valid = df.iloc[val_idx]

        x_train = train[features].reset_index(drop=True)
        y_train = train['Survived'].reset_index(drop=True)
        x_valid = valid[features].reset_index(drop=True)
        y_valid = valid['Survived'].reset_index(drop=True)

        clf = DecisionTree(max_depth=5)
        clf.fit(x_train, y_train)
        y_predict = clf.predict(x_valid)

        accuracy += np.sum(y_predict==y_valid)/(5*len(y_valid))
    print(accuracy)

