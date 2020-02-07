from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier

from functions import loadDataset, scaleData, plotLearningCurve

if __name__ == '__main__':

    # Caricamento del dataset

    X_train, y_train, X_test, y_test = loadDataset()

    print("Train set: Rows: %d, Columns: %d" % (X_train.shape[0], X_train.shape[1]))
    print("Test set: Rows: %d, Columns: %d" % (X_test.shape[0], X_test.shape[1]))

    # Processazione preliminare dei dati

    X_train_std, X_test_std = scaleData(X_train, X_test)

    # Istanze dei modelli

    decisionTree = DecisionTreeClassifier()
    perceptron = Perceptron()

    # Disegno delle curve di apprendimento

    iter = 25

    plotLearningCurve(decisionTree, X_train_std, y_train, X_test_std, y_test, iter)
    plotLearningCurve(perceptron, X_train_std, y_train, X_test_std, y_test, iter)

