import pandas as pd
import sklearn.tree as tree
import sklearn.model_selection as model_selection
import sys

def main():
    print(sys.argv)

    df = pd.read_csv(sys.argv[1], error_bad_lines=False)
    decision_tree = tree.DecisionTreeClassifier()
    train_df, test_df = model_selection.train_test_split(df)

    Y_train = train_df[sys.argv[2]]
    X_train = train_df.drop(sys.argv[2], axis=1)

    Y_test = test_df[sys.argv[2]]
    X_test = test_df.drop(sys.argv[2], axis=1)

    decision_tree = decision_tree.fit(X_train, Y_train)

    print("Decision Tree Accuracy: " + str(decision_tree.score(X_test, Y_test)))


if __name__ == '__main__':
    main()
