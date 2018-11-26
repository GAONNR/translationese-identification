from sklearn import svm, model_selection
from sklearn.metrics import recall_score, precision_score, f1_score
from chunknizer import getXy
import random
import sys


def trainSVM(X_train, X_test, y_train, y_test):
    clf = svm.SVC(kernel='linear', C=100000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test)
    recall = recall_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    print('Accuracy:', accuracy)
    print('Recall:', recall)
    print('Precision:', precision)
    print('F1_score:', f1)

    return


if __name__ == '__main__':
    if 'europarl' in sys.argv:
        X, y = getXy('europarl')
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y, test_size=0.2)
        trainSVM(X_train, X_test, y_train, y_test)

    if 'literature' in sys.argv:
        if 'cross' in sys.argv:
            X_train, y_train = getXy('europarl')
            X_test, y_test = getXy('literature')
            trainSVM(X_train, X_test, y_train, y_test)
        elif 'mix' in sys.argv:
            Xe, ye = getXy('europarl')
            Xl, yl = getXy('literature')
            Xe = Xe[:len(Xl)]
            ye = ye[:len(Xl)]
            mixedFeatures = list(zip(Xe + Xl, ye + yl))
            random.shuffle(mixedFeatures)
            X, y = zip(*mixedFeatures)
            X_train, X_test, y_train, y_test = model_selection.train_test_split(
                X, y, test_size=0.2)
            trainSVM(X_train, X_test, y_train, y_test)
        else:
            X, y = getXy('literature')
            X_train, X_test, y_train, y_test = model_selection.train_test_split(
                X, y, test_size=0.2)
            trainSVM(X_train, X_test, y_train, y_test)
