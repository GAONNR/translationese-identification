from sklearn import svm, model_selection
from sklearn.metrics import recall_score, precision_score, f1_score
from functionwords import functionWords
import random
import spacy
import sys
import csv

europarlInfo = {
    'dat': 'corpus/europarl-v7.EN-FR/europarl-v7.fr-en.dat',
    'tok': 'corpus/europarl-v7.EN-FR/europarl-v7.fr-en.en.aligned.tok'
}


def calcStopWords(toks, tokDict):
    for tok in toks:
        if tok.text not in functionWords:
            continue

        if tok.text in tokDict:
            tokDict[tok.text] += 1
        else:
            tokDict[tok.text] = 1


def chunkEuroparl(nlp):
    enLines = 0
    enToks = 0
    tmpEnChunk = ''
    tmpEnToks = 0
    tmpEnIdx = 0
    enDict = dict()
    fEnChunks = open('chunks/europarl/en.csv', 'w')
    enWriter = csv.writer(fEnChunks)

    frLines = 0
    frToks = 0
    tmpFrChunk = ''
    tmpFrToks = 0
    tmpFrIdx = 0
    frDict = dict()
    fFrChunks = open('chunks/europarl/fr.csv', 'w')
    frWriter = csv.writer(fFrChunks)

    with open(europarlInfo['dat'], 'r') as corpusDat, \
            open(europarlInfo['tok'], 'r') as corpusTok:
        for datLine, tokLine in zip(corpusDat, corpusTok):
            if '\"EN\"' in datLine:
                lineToks = nlp(tokLine)
                enLines += 1
                enToks += len(lineToks)
                calcStopWords(lineToks, enDict)

                tmpEnChunk += tokLine
                tmpEnToks += len(lineToks)

                if tmpEnToks >= 2000:
                    enWriter.writerow([tmpEnIdx, tmpEnChunk, tmpEnToks])
                    tmpEnChunk = ''
                    tmpEnToks = 0
                    tmpEnIdx += 1

                    if tmpEnIdx % 100 == 0:
                        print('En chunk number %d has written' % tmpEnIdx)

            elif '\"FR\"' in datLine:
                lineToks = nlp(tokLine)
                frLines += 1
                frToks += len(lineToks)
                calcStopWords(lineToks, frDict)

                tmpFrChunk += tokLine
                tmpFrToks += len(lineToks)

                if tmpFrToks >= 2000:
                    frWriter.writerow([tmpFrIdx, tmpFrChunk, tmpFrToks])
                    tmpFrChunk = ''
                    tmpFrToks = 0
                    tmpFrIdx += 1

                    if tmpFrIdx % 100 == 0:
                        print('Fr chunk number %d has written' % tmpFrIdx)

        enWriter.writerow([tmpEnIdx, tmpEnChunk, tmpEnToks])
        frWriter.writerow([tmpFrIdx, tmpFrChunk, tmpFrToks])

        print('Europarl', 'English Lines:', enLines)
        print('Europarl', 'English Tokens:', enToks)
        print('Europarl', 'French Lines:', frLines)
        print('Europarl', 'French Tokens:', frToks)

        print('==========')
        print('English Stopwords Dict')
        print(enDict)

        print('==========')
        print('French Stopwords Dict')
        print(frDict)

    fEnChunks.close()
    fFrChunks.close()
    return


def csvToTotalChunks(totalChunks, csvReader, val, num):
    cnt = 0
    for no, chunk, chunkNum in csvReader:
        totalChunks.append([chunk.lower(), val, chunkNum])
        cnt += 1

        if cnt >= num:
            break
    return


def getStopWordsStat(nlp, chunk, chunkNum):
    tokenizedChunk = nlp(chunk)
    stopwordsStat = dict()

    for token in tokenizedChunk:
        if token.is_stop:
            if token.text in stopwordsStat:
                stopwordsStat[token.text] += 1
            else:
                stopwordsStat[token.text] = 1

    for key in stopwordsStat.keys():
        stopwordsStat[key] /= int(chunkNum)

    return stopwordsStat


def getFeaturesOfEuroparlSVM(nlp):
    fEnChunks = open('chunks/europarl/en.csv', 'r')
    fFrChunks = open('chunks/europarl/fr.csv', 'r')

    enReader = csv.reader(fEnChunks)
    frReader = csv.reader(fFrChunks)

    totalChunks = list()

    csvToTotalChunks(totalChunks, enReader, False, 1500)
    csvToTotalChunks(totalChunks, frReader, True, 1500)
    random.shuffle(totalChunks)

    print('Shuffled Chunks')

    fEnChunks.close()
    fFrChunks.close()

    stopwordsSet = set()
    stopwordsStats = list()
    for chunk, val, chunkNum in totalChunks:
        stopwordsStat = getStopWordsStat(nlp, chunk, chunkNum)
        stopwordsStats.append([stopwordsStat, val])
        stopwordsSet.update(list(stopwordsStat.keys()))
    stopwordsList = list(stopwordsSet)

    print('Marking Function Words Complete')

    X = [[0 for _ in range(len(stopwordsList))]
         for _ in range(len(totalChunks))]
    y = [0 for _ in range(len(totalChunks))]

    print('Writing to csv File....')

    for i in range(len(totalChunks)):
        y[i] = 1 if stopwordsStats[i][1] else 0
        for j in range(len(stopwordsList)):
            stopwordsStat = stopwordsStats[i][0]
            if stopwordsList[j] in stopwordsStat:
                X[i][j] = stopwordsStat[stopwordsList[j]]

    fEuroparlFeatures = open('features/europarl/features.csv', 'w')
    featureWriter = csv.writer(fEuroparlFeatures)
    featureWriter.writerow(stopwordsList + ['val'])
    for i in range(len(totalChunks)):
        featureWriter.writerow(X[i] + [y[i]])
    fEuroparlFeatures.close()

    return


def trainEuroparlSVM(nlp):
    fFeatures = open('features/europarl/features.csv', 'r')
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

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2)

    for kernel in ['linear']:
        print('Training by %s kernel' % kernel)
        clf = svm.SVC(kernel=kernel, C=100000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = clf.score(X_test, y_test)
        recall = recall_score(y_test, y_pred, average="macro")
        precision = precision_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")
        print(kernel, 'accuracy:', accuracy)
        print(kernel, 'recall:', recall)
        print(kernel, 'precision:', precision)
        print(kernel, 'f1_score:', f1)


if __name__ == '__main__':
    nlp = spacy.load('en')

    if 'europarl' in sys.argv:
        if 'chunknize' in sys.argv:
            chunkEuroparl(nlp)
        if 'svm' in sys.argv:
            if 'features' in sys.argv:
                getFeaturesOfEuroparlSVM(nlp)
            if 'train' in sys.argv:
                trainEuroparlSVM(nlp)
