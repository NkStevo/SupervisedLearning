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
    df = pd.read_csv("Datasets/weatherAUS.csv", error_bad_lines=False)

    encoder = LabelEncoder()
    imputer = Imputer(missing_values="NA")

    decision_tree = DecisionTreeClassifier(max_depth=6)
    neural_network = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100, 100, 100, 100, 100, 100), tol=0.00001)
    boosted_tree = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=275, learning_rate=1)
    sigmoid_svm = SVC(kernel='sigmoid')
    rbf_svm = SVC(kernel='rbf', C=0.1)
    five_knn = KNeighborsClassifier(n_neighbors=5, p=1)

    df.dropna()
    df.drop('Date', axis=1, inplace=True)
    df.drop('RISK_MM', axis=1, inplace=True)

    df = df[df["Location"] != "NA"]
    df = df[df["WindGustDir"] != "NA"]
    df = df[df["WindDir9am"] != "NA"]
    df = df[df["WindDir3pm"] != "NA"]
    df = df[df["RainToday"] != "NA"]
    df = df[df["RainTomorrow"] != "NA"]

    df["Location"] = encoder.fit_transform(df["Location"].astype(str))
    df["WindGustDir"] = encoder.fit_transform(df["WindGustDir"].astype(str))
    df["WindDir9am"] = encoder.fit_transform(df["WindDir9am"].astype(str))
    df["WindDir3pm"] = encoder.fit_transform(df["WindDir3pm"].astype(str))
    df["RainToday"] = encoder.fit_transform(df["RainToday"].astype(str))
    df["RainTomorrow"] = encoder.fit_transform(df["RainTomorrow"].astype(str))

    df = df[np.isfinite(df['Evaporation'])]
    df = df[np.isfinite(df['Sunshine'])]
    df = df[np.isfinite(df['WindGustSpeed'])]
    df = df[np.isfinite(df['WindSpeed9am'])]
    df = df[np.isfinite(df['WindSpeed3pm'])]
    df = df[np.isfinite(df['Humidity9am'])]
    df = df[np.isfinite(df['Humidity3pm'])]
    df = df[np.isfinite(df['Pressure9am'])]
    df = df[np.isfinite(df['Pressure3pm'])]
    df = df[np.isfinite(df['Cloud9am'])]
    df = df[np.isfinite(df['Cloud3pm'])]
    df = df[np.isfinite(df['Temp9am'])]
    df = df[np.isfinite(df['Temp3pm'])]
    df = df[np.isfinite(df['MinTemp'])]
    df = df[np.isfinite(df['MaxTemp'])]
    df = df[np.isfinite(df['Rainfall'])]

    df = df.reset_index()

    train_df, test_df = train_test_split(df, random_state=45)

    Y_train = train_df["RainTomorrow"]
    X_train = train_df.drop("RainTomorrow", axis=1)

    Y_test = test_df["RainTomorrow"]
    X_test = test_df.drop("RainTomorrow", axis=1)


    '''
    print("Fitting boosted decision tree...")
    start_time = time.time()
    train_results = []
    test_results = []
    max_acc = -1
    max_val = 0

    max_depths = range(1, 11)

    for max_depth in max_depths:
        knn = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=178, learning_rate=0.9)
        knn = knn.fit(X_train, Y_train)
        test_score = knn.score(X_test, Y_test)
        train_score = knn.score(X_train, Y_train)

        if test_score > max_acc:
            max_acc = test_score
            max_val = max_depth

        test_results.append(test_score)
        train_results.append(train_score)

    from matplotlib.legend_handler import HandlerLine2D
    line1, = plt.plot(max_depths, train_results, 'b', label="Train Accuracy")
    line2, = plt.plot(max_depths, test_results, 'r', label="Test Accuracy")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel("Accuracy score")
    plt.xlabel("Max Depth")
    plt.savefig('Figures/BoostedTree/boosted_tree_max_depth.pdf', bbox_inches='tight')
    plt.clf()
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    print("MAX VALUE: " + str(max_val))
    print("MAX ACCURACY: " + str(max_acc))
    '''
    '''
    print("Fitting sigmoid svm...")
    start_time = time.time()
    #cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    #neural_network = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100, 100, 100, 100, 100, 100), max_iter=200, learning_rate='adaptive', tol=0.00001, verbose=1)

    #plot_learning_curve(neural_network, "Neural Network (MLP) Learning Curve", X_train, Y_train, (0.1, 1.01), cv=cv, n_jobs=4)
    #plt.savefig('Figures/NeuralNetwork/neural_network_10_layers.pdf', bbox_inches='tight')
    #neural_network.fit(X_train, Y_train)

    #print("Neural Network (MLP) Accuracy: " + str(neural_network.score(X_test, Y_test)))


    train_results = []
    test_results = []
    max_acc = -1
    max_val = 0

    max_depths = [0.1, 1, 10, 100, 1000]

    for max_depth in max_depths:
        print("Fitting interim svm...")
        interim_time = time.time()
        knn = SVC(kernel='rbf', C=max_depth)
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
    plt.xlabel("C")
    plt.savefig('Figures/RBF_SVM/c_curve.pdf', bbox_inches='tight')
    plt.clf()
    print("MAX VALUE: " + str(max_val))
    print("MAX ACCURACY: " + str(max_acc))



    #plt.plot(neural_network.loss_curve_, label="MLP Loss Curve")
    #plt.savefig('Figures/NeuralNetwork/neural_network_loss_curve_10_layer.pdf', bbox_inches='tight')
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    '''

    '''
    # specify parameters and distributions to sample from
    param_dist = {'hidden_layer_sizes': [(sp_randint.rvs(100,600,1),sp_randint.rvs(100,600,1),), (sp_randint.rvs(100,600,1),)],\
                    "solver": ["adam", "sgd", "lbfgs"],\
                    "activation": ["identity", "logistic", "tanh", "relu"],\
                    "min_samples_split": sp_randint(2, 11),\
                    'alpha': uniform(0.0001, 0.9),\
                    'learning_rate': ['constant','adaptive']}

    n_iter_search = 20
    random_search = RandomizedSearchCV(neural_network, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=5)

    start = time.time()
    random_search.fit(X_train, Y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time.time() - start), n_iter_search))
    report(random_search.cv_results_)
    '''

    '''
    print("Plotting neural network...")
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    plot_learning_curve(neural_network, "Neural Network (MLP) Learning Curve", X_train, Y_train, (0.5, 1.01), cv=cv, n_jobs=4)
    plt.savefig('Figures/NeuralNetwork/neural_network_logistic.pdf', bbox_inches='tight')
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
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

    print("Fitting k-nearest neighbors (5)...")
    start_time = time.time()
    five_knn = five_knn.fit(X_train, Y_train)
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    print("Decision Tree Accuracy: " + str(decision_tree.score(X_test, Y_test)))
    print("Boosted Decision Tree Accuracy: " + str(boosted_tree.score(X_test, Y_test)))

    print("Neural Network (MLP) Accuracy: " + str(neural_network.score(X_test, Y_test)))

    print("Support Vector Machine (RBF) Accuracy: " + str(rbf_svm.score(X_test, Y_test)))
    print("Support Vector Machine (Sigmoid) Accuracy: " + str(sigmoid_svm.score(X_test, Y_test)))

    print("K-Nearest Neighbors (5) Accuracy: " + str(five_knn.score(X_test, Y_test)))

    cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    plot_learning_curve(neural_network, "Neural Network (MLP) Learning Curve", X_train, Y_train, (0.5, 1.01), cv=cv, n_jobs=4)
    plt.savefig('Figures/neural_network_ideal.pdf', bbox_inches='tight')

    cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    plot_learning_curve(decision_tree, "Decision Tree Learning Curve", X_train, Y_train, (0.5, 1.01), cv=cv, n_jobs=4)
    plt.savefig('Figures/decision_tree_ideal.pdf', bbox_inches='tight')

    cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    plot_learning_curve(boosted_tree, "Decision Tree (Boosted) Learning Curve", X_train, Y_train, (0.5, 1.01), cv=cv, n_jobs=4)
    plt.savefig('Figures/decision_tree_boosted_ideal.pdf', bbox_inches='tight')

    cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    plot_learning_curve(five_knn, "K-NN (5) Learning Curve", X_train, Y_train, (0.5, 1.01), cv=cv, n_jobs=4)
    plt.savefig('Figures/knn5_ideal.pdf', bbox_inches='tight')

    cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    plot_learning_curve(rbf_svm, "SVM (RBF) Learning Curve", X_train, Y_train, (0.5, 1.01), cv=cv, n_jobs=4)
    plt.savefig('Figures/rbfsvm_ideal.pdf', bbox_inches='tight')

    cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    plot_learning_curve(sigmoid_svm, "SVM (Sigmoid) Learning Curve", X_train, Y_train, (0.5, 1.01), cv=cv, n_jobs=4)
    plt.savefig('Figures/sigmoidsvm_ideal.pdf', bbox_inches='tight')

    """
    train_results = []
    test_results = []

    max_depths = np.logspace(-3, 2, 6)

    for max_depth in max_depths:
        knn = SVC(kernel='sigmoid', C=max_depth)
        knn = knn.fit(X_train, Y_train)
        test_score = knn.score(X_test, Y_test)
        train_score = knn.score(X_train, Y_train)

        test_results.append(test_score)
        train_results.append(train_score)

    from matplotlib.legend_handler import HandlerLine2D
    line1, = plt.plot(max_depths, train_results, 'b', label="Train Accuracy")
    line2, = plt.plot(max_depths, test_results, 'r', label="Test Accuracy")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel("Accuracy score")
    plt.xlabel("C")
    plt.savefig('Figures/SigmoidSVM/svm_c.pdf', bbox_inches='tight')
    plt.clf()


    train_results = []
    test_results = []

    max_depths = np.logspace(-3, 2, 6)

    for max_depth in max_depths:
        knn = SVC(kernel='sigmoid', gamma=max_depth)
        knn = knn.fit(X_train, Y_train)
        test_score = knn.score(X_test, Y_test)
        train_score = knn.score(X_train, Y_train)

        test_results.append(test_score)
        train_results.append(train_score)

    from matplotlib.legend_handler import HandlerLine2D
    line1, = plt.plot(max_depths, train_results, 'b', label="Train Accuracy")
    line2, = plt.plot(max_depths, test_results, 'r', label="Test Accuracy")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel("Accuracy score")
    plt.xlabel("gamma")
    plt.savefig('Figures/SigmoidSVM/svm_gamma.pdf', bbox_inches='tight')
    plt.clf()


    '''
    AAAAAAAAAAAGGGGHHH
    '''

    avg_time = 0
    avg_accuracy = 0

    #decision_tree = DecisionTreeClassifier(class_weight='balanced')

    sigmoid_svm = SVC(kernel='rbf')

    for x in range(5):
        print("Fitting support vector machine (sigmoid)...")
        start_time = time.time()
        sigmoid_svm = sigmoid_svm.fit(X_train, Y_train)
        elapsed_time = time.time() - start_time
        print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        score = sigmoid_svm.score(X_test, Y_test)
        print("SVM (RBF) Accuracy: " + str(score))

        avg_time += elapsed_time
        avg_accuracy += score

    avg_time = avg_time / 5
    avg_accuracy = avg_accuracy / 5

    print("\nAverage Elapsed Time: " + str(avg_time))
    print("Average Accuracy: " + str(avg_accuracy))


    avg_time = 0
    avg_accuracy = 0

    print("Plotting svms...")
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    plot_learning_curve(sigmoid_svm, "SVM (RBF) Learning Curve", X_train, Y_train, (0.5, 1.01), cv=cv, n_jobs=4)
    plt.savefig('Figures/RBF_SVM/svm.pdf', bbox_inches='tight')
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    plt.clf()


    sigmoid_svm = SVC(kernel='rbf', shrinking=False)

    for x in range(5):
        print("Fitting support vector machine (rbf)...")
        start_time = time.time()
        sigmoid_svm = sigmoid_svm.fit(X_train, Y_train)
        elapsed_time = time.time() - start_time
        print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        score = sigmoid_svm.score(X_test, Y_test)
        print("SVM (RBF) Accuracy: " + str(score))

        avg_time += elapsed_time
        avg_accuracy += score

    avg_time = avg_time / 5
    avg_accuracy = avg_accuracy / 5

    print("\nAverage Elapsed Time: " + str(avg_time))
    print("Average Accuracy: " + str(avg_accuracy))

    avg_time = 0
    avg_accuracy = 0

    print("Plotting svms...")
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    plot_learning_curve(sigmoid_svm, "SVM (RBF) Learning Curve", X_train, Y_train, (0.5, 1.01), cv=cv, n_jobs=4)
    plt.savefig('Figures/RBF_SVM/svm_shrinking_false.pdf', bbox_inches='tight')
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    plt.clf()


    sigmoid_svm = SVC(kernel='rbf', decision_function_shape='ovo')

    for x in range(5):
        print("Fitting support vector machine (rbf)...")
        start_time = time.time()
        sigmoid_svm = sigmoid_svm.fit(X_train, Y_train)
        elapsed_time = time.time() - start_time
        print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        score = sigmoid_svm.score(X_test, Y_test)
        print("SVM (RBF) Accuracy: " + str(score))

        avg_time += elapsed_time
        avg_accuracy += score

    avg_time = avg_time / 5
    avg_accuracy = avg_accuracy / 5

    print("\nAverage Elapsed Time: " + str(avg_time))
    print("Average Accuracy: " + str(avg_accuracy))

    avg_time = 0
    avg_accuracy = 0

    print("Plotting svms...")
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    plot_learning_curve(sigmoid_svm, "SVM (RBF) Learning Curve", X_train, Y_train, (0.5, 1.01), cv=cv, n_jobs=4)
    plt.savefig('Figures/RBF_SVM/svm_ovo.pdf', bbox_inches='tight')
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    plt.clf()

    train_results = []
    test_results = []

    max_depths = np.logspace(-3, 2, 6)

    for max_depth in max_depths:
        knn = SVC(kernel='rbf', C=max_depth)
        knn = knn.fit(X_train, Y_train)
        test_score = knn.score(X_test, Y_test)
        train_score = knn.score(X_train, Y_train)

        test_results.append(test_score)
        train_results.append(train_score)

    from matplotlib.legend_handler import HandlerLine2D
    line1, = plt.plot(max_depths, train_results, 'b', label="Train Accuracy")
    line2, = plt.plot(max_depths, test_results, 'r', label="Test Accuracy")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel("Accuracy score")
    plt.xlabel("C")
    plt.savefig('Figures/RBF_SVM/svm_c.pdf', bbox_inches='tight')
    plt.clf()


    train_results = []
    test_results = []

    max_depths = np.logspace(-3, 2, 6)

    for max_depth in max_depths:
        knn = SVC(kernel='rbf', gamma=max_depth)
        knn = knn.fit(X_train, Y_train)
        test_score = knn.score(X_test, Y_test)
        train_score = knn.score(X_train, Y_train)

        test_results.append(test_score)
        train_results.append(train_score)

    from matplotlib.legend_handler import HandlerLine2D
    line1, = plt.plot(max_depths, train_results, 'b', label="Train Accuracy")
    line2, = plt.plot(max_depths, test_results, 'r', label="Test Accuracy")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel("Accuracy score")
    plt.xlabel("gamma")
    plt.savefig('Figures/RBF_SVM/svm_gamma.pdf', bbox_inches='tight')
    plt.clf()
    """
    '''
    print("Plotting decision trees...")
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    plot_learning_curve(neural_network, "Decision Tree Learning Curve", X_train, Y_train, (0.5, 1.01), cv=cv, n_jobs=4)
    plt.savefig('Figures/DecisionTree/decision_tree_class_weight_balanced.pdf', bbox_inches='tight')
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    '''

if __name__ == '__main__':
    main()
