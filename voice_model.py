import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import sys

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint as sp_randint
from random import uniform

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def main():
    df = pd.read_csv('Datasets/voice.csv', error_bad_lines=False)
    encoder = LabelEncoder()

    decision_tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=7)
    neural_network = MLPClassifier(activation='tanh', hidden_layer_sizes=(69,))
    boosted_tree = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=110, learning_rate=1)
    linear_svm = SVC(kernel='linear', C=100)
    rbf_svm = SVC(kernel='rbf', C=100, gamma=1)
    sigmoid_svm = SVC(kernel='sigmoid', C=0.1, gamma=0.1)
    five_knn = KNeighborsClassifier(n_neighbors=7, p=1)

    df["label"] = encoder.fit_transform(df["label"].astype(str))

    train_df, test_df = train_test_split(df, random_state=45)

    Y_train = train_df['label']
    encoder.fit(Y_train)
    Y_train = encoder.transform(Y_train)
    X_train = train_df.drop('label', axis=1)

    Y_test = test_df['label']
    X_test = test_df.drop('label', axis=1)

    '''
    svm = SVC(kernel='linear')

    print("Fitting SVM (Linear)...")
    start_time = time.time()
    svm = svm.fit(X_train, Y_train)
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    print("SVM Accuracy: " + str(svm.score(X_test, Y_test)))

    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    plot_learning_curve(svm, "SVM Learning Curve", X_train, Y_train, (0.1, 1.01), cv=cv, n_jobs=4)
    plt.savefig('Figures2/LinearSVM/svm.pdf', bbox_inches='tight')
    plt.clf()
    '''



    '''
    print("Fitting sigmoid svm...")
    start_time = time.time()

    train_results = []
    test_results = []
    max_acc = -1
    max_val = 0
    #decision_trees : m_d-4, m_s_l-7
    max_depths = [0.1, 1, 10, 100]

    for max_depth in max_depths:
        print("Fitting knn...")
        interim_time = time.time()
        knn = SVC(kernel='sigmoid', C=0.1, gamma=max_depth)
        knn = knn.fit(X_train, Y_train)
        test_score = knn.score(X_test, Y_test)
        train_score = knn.score(X_train, Y_train)

        if test_score > max_acc:
            max_acc = test_score
            max_val = max_depth

        test_results.append(test_score)
        train_results.append(train_score)
        elapsed_time = time.time() - interim_time
        print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    from matplotlib.legend_handler import HandlerLine2D
    line1, = plt.plot(max_depths, train_results, 'b', label="Train Accuracy")
    line2, = plt.plot(max_depths, test_results, 'r', label="Test Accuracy")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel("Accuracy score")
    plt.xlabel("Gamma")
    plt.savefig('Figures2/SigmoidSVM/gamma_curve.pdf', bbox_inches='tight')
    plt.clf()
    print("MAX VALUE: " + str(max_val))
    print("MAX ACCURACY: " + str(max_acc))

    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    '''




    '''
    print("Fitting decision trees...")
    decision_tree = decision_tree.fit(X_train, Y_train)
    print("Fitting boosted decision tree...")
    boosted_tree = boosted_tree.fit(X_train, Y_train)

    print("Fitting neural network...")
    neural_network = neural_network.fit(X_train, Y_train)

    print("Fitting support vector machine (linear)...")
    linear_svm = linear_svm.fit(X_train, Y_train)
    #print("Fitting support vector machine (polynomial)...")
    #poly_svm = poly_svm.fit(X_train, Y_train)
    print("Fitting support vector machine (sigmoid)...")
    sigmoid_svm = sigmoid_svm.fit(X_train, Y_train)

    print("Fitting k-nearest neighbors (5)...")
    five_knn = five_knn.fit(X_train, Y_train)
    print("Fitting k-nearest neighbors (10)...")
    ten_knn = ten_knn.fit(X_train, Y_train)
    print("Fitting k-nearest neighbors (20)...")
    twenty_knn = twenty_knn.fit(X_train, Y_train)

    print("Decision Tree Accuracy: " + str(decision_tree.score(X_test, Y_test)))
    print("Boosted Decision Tree Accuracy: " + str(boosted_tree.score(X_test, Y_test)))

    print("Neural Network (MLP) Accuracy: " + str(neural_network.score(X_test, Y_test)))

    print("Support Vector Machine (Linear) Accuracy: " + str(linear_svm.score(X_test, Y_test)))
    #print("Support Vector Machine (Polynomial) Accuracy: " + str(poly_svm.score(X_test, Y_test)))
    print("Support Vector Machine (Sigmoid) Accuracy: " + str(sigmoid_svm.score(X_test, Y_test)))

    print("K-Nearest Neighbors (5) Accuracy: " + str(five_knn.score(X_test, Y_test)))
    print("K-Nearest Neighbors (10) Accuracy: " + str(ten_knn.score(X_test, Y_test)))
    print("K-Nearest Neighbors (20) Accuracy: " + str(twenty_knn.score(X_test, Y_test)))
    '''

    print("Fitting decision trees...")
    start_time = time.time()
    decision_tree = decision_tree.fit(X_train, Y_train)
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    print("Fitting boosted decision tree...")
    start_time = time.time()
    boosted_tree = boosted_tree.fit(X_train, Y_train)
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    print("Fitting neural network...")
    start_time = time.time()
    neural_network = neural_network.fit(X_train, Y_train)
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    print("Fitting support vector machine (sigmoid)...")
    start_time = time.time()
    sigmoid_svm = sigmoid_svm.fit(X_train, Y_train)
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    print("Fitting support vector machine (rbf)...")
    start_time = time.time()
    rbf_svm = rbf_svm.fit(X_train, Y_train)
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    print("Fitting support vector machine (linear)...")
    start_time = time.time()
    sigmoid_svm = linear_svm.fit(X_train, Y_train)
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    print("Fitting k-nearest neighbors (7)...")
    start_time = time.time()
    five_knn = five_knn.fit(X_train, Y_train)
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    print("Decision Tree Accuracy: " + str(decision_tree.score(X_test, Y_test)))
    print("Boosted Decision Tree Accuracy: " + str(boosted_tree.score(X_test, Y_test)))

    print("Neural Network (MLP) Accuracy: " + str(neural_network.score(X_test, Y_test)))

    print("Support Vector Machine (RBF) Accuracy: " + str(rbf_svm.score(X_test, Y_test)))
    print("Support Vector Machine (Linear) Accuracy: " + str(linear_svm.score(X_test, Y_test)))
    print("Support Vector Machine (Sigmoid) Accuracy: " + str(sigmoid_svm.score(X_test, Y_test)))

    print("K-Nearest Neighbors (7) Accuracy: " + str(five_knn.score(X_test, Y_test)))

    cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    plot_learning_curve(neural_network, "Neural Network (MLP) Learning Curve", X_train, Y_train, (0.5, 1.01), cv=cv, n_jobs=4)
    plt.savefig('Figures2/neural_network_ideal.pdf', bbox_inches='tight')

    cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    plot_learning_curve(decision_tree, "Decision Tree Learning Curve", X_train, Y_train, (0.5, 1.01), cv=cv, n_jobs=4)
    plt.savefig('Figures2/decision_tree_ideal.pdf', bbox_inches='tight')

    cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    plot_learning_curve(boosted_tree, "Decision Tree (Boosted) Learning Curve", X_train, Y_train, (0.5, 1.01), cv=cv, n_jobs=4)
    plt.savefig('Figures2/decision_tree_boosted_ideal.pdf', bbox_inches='tight')

    cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    plot_learning_curve(five_knn, "K-NN (7) Learning Curve", X_train, Y_train, (0.5, 1.01), cv=cv, n_jobs=4)
    plt.savefig('Figures2/knn5_ideal.pdf', bbox_inches='tight')

    cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    plot_learning_curve(rbf_svm, "SVM (RBF) Learning Curve", X_train, Y_train, (0.5, 1.01), cv=cv, n_jobs=4)
    plt.savefig('Figures2/rbfsvm_ideal.pdf', bbox_inches='tight')

    cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    plot_learning_curve(linear_svm, "SVM (Linear) Learning Curve", X_train, Y_train, (0.5, 1.01), cv=cv, n_jobs=4)
    plt.savefig('Figures2/linearsvm_ideal.pdf', bbox_inches='tight')

    cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    plot_learning_curve(sigmoid_svm, "SVM (Sigmoid) Learning Curve", X_train, Y_train, (0.5, 1.01), cv=cv, n_jobs=4)
    plt.savefig('Figures2/sigmoidsvm_ideal.pdf', bbox_inches='tight')

if __name__ == '__main__':
    main()
