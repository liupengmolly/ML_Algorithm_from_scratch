import pandas as pd
import numpy as np
from boosting.Adaboost import AdaBoost
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    df = pd.read_csv("../data/titanic/processed_train.csv")
    columns = list(df.columns)
    features = columns
    features.remove('Survived')

    kf = KFold(n_splits=5)
    accuracy = 0
    for i,(trn_idx, val_idx) in enumerate(kf.split(df)):

        train = df.iloc[trn_idx]
        valid = df.iloc[val_idx]

        x_train = train[features].reset_index(drop=True)
        y_train = train['Survived'].reset_index(drop=True)
        x_valid = valid[features].reset_index(drop=True)
        y_valid = valid['Survived'].reset_index(drop=True)

        # clf = DecisionTreeClassifier(max_depth=8)
        # clf = AdaBoostClassifier(learning_rate=0.1)
        clf = AdaBoost(n_estimators=10, max_depth=5)
        clf.fit(x_train, y_train)
        y_predict = clf.predict(x_valid)

        accuracy += np.sum(y_predict==y_valid)/(5*len(y_valid))
        print(accuracy)
    print('=' * 30)
    print(accuracy)
