from sklearn import svm, model_selection
from sklearn.metrics import recall_score, precision_score, f1_score
import spacy
import sys
import csv


def getXy(corpusName):
    fFeatures = open('features/%s/features.csv' % corpusName, 'r')
    featuresReader = csv.reader(fFeatures)

    column = True
    featuresNum = 0
    X = list()
    y = list()
    for line in featuresReader:
        if column:
            featuresNum = len(line) - 1
            column = False
            continue
        X.append(list(map(lambda x: float(x), line[:featuresNum])))
        y.append(int(line[-1]))

    fFeatures.close()
    return X, y


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
        else:
            X, y = getXy('literature')
            X_train, X_test, y_train, y_test = model_selection.train_test_split(
                X, y, test_size=0.2)
            trainSVM(X_train, X_test, y_train, y_test)
