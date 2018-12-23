from function_words import function_words as functionWords
import random
import spacy
import sys
import csv

europarlInfo = {
    'dat': 'corpus/europarl-v7.EN-DE/europarl-v7.de-en.dat',
    'tok': 'corpus/europarl-v7.EN-DE/europarl-v7.de-en.en.aligned.tok'
}

featuresLimit = {
    'europarl': 1500,
    'literature': 400
}


def chunkEuroparl(nlp):
    print('chunkEuroparl')
    enLines = 0
    enToks = 0
    tmpEnChunk = ''
    tmpEnToks = 0
    tmpEnIdx = 0
    fEnChunks = open('chunks/europarl/en-1.csv', 'w')
    enWriter = csv.writer(fEnChunks)

    frLines = 0
    frToks = 0
    tmpFrChunk = ''
    tmpFrToks = 0
    tmpFrIdx = 0
    fFrChunks = open('chunks/europarl/de.csv', 'w')
    frWriter = csv.writer(fFrChunks)

    with open(europarlInfo['dat'], 'r') as corpusDat, \
            open(europarlInfo['tok'], 'r') as corpusTok:
        for datLine, tokLine in zip(corpusDat, corpusTok):
            if '\"EN\"' in datLine:
                lineToks = tokLine.strip().split(' ')
                enLines += 1
                enToks += len(lineToks)

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
                lineToks = tokLine.strip().split(' ')
                frLines += 1
                frToks += len(lineToks)

                tmpFrChunk += tokLine
                tmpFrToks += len(lineToks)

                if tmpFrToks >= 2000:
                    frWriter.writerow([tmpFrIdx, tmpFrChunk, tmpFrToks])
                    tmpFrChunk = ''
                    tmpFrToks = 0
                    tmpFrIdx += 1

                    if tmpFrIdx % 100 == 0:
                        print('De chunk number %d has written' % tmpFrIdx)

        enWriter.writerow([tmpEnIdx, tmpEnChunk, tmpEnToks])
        frWriter.writerow([tmpFrIdx, tmpFrChunk, tmpFrToks])

        print('Europarl', 'English Lines:', enLines)
        print('Europarl', 'English Tokens:', enToks)
        print('Europarl', 'German Lines:', frLines)
        print('Europarl', 'German Tokens:', frToks)

    fEnChunks.close()
    fFrChunks.close()
    return


def chunkLiterature(nlp):
    print('chunkLiterature')
    enLines = 0
    enToks = 0
    tmpEnChunk = ''
    tmpEnToks = 0
    tmpEnIdx = 0
    fEnChunks = open('chunks/literature/en-1.csv', 'w')
    enWriter = csv.writer(fEnChunks)

    frLines = 0
    frToks = 0
    tmpFrChunk = ''
    tmpFrToks = 0
    tmpFrIdx = 0
    fFrChunks = open('chunks/literature/de.csv', 'w')
    frWriter = csv.writer(fFrChunks)

    with open('corpus/literature.EN-DE/literature.dat', 'r') as fLiteratureDat:
        for datLine in fLiteratureDat:
            if len(datLine.split(' ')) < 2:
                continue
            isTranslated, title = datLine.split(' ')
            if (isTranslated == 'S' and '.en.' in title):
                title = title.strip()
                with open('corpus/literature.EN-DE/books/%s' % title) as book:
                    for tokLine in book:
                        lineToks = tokLine.strip().split(' ')
                        enLines += 1
                        enToks += len(lineToks)

                        tmpEnChunk += tokLine
                        tmpEnToks += len(lineToks)

                        if tmpEnToks >= 2000:
                            enWriter.writerow(
                                [tmpEnIdx, tmpEnChunk, tmpEnToks])
                            tmpEnChunk = ''
                            tmpEnToks = 0
                            tmpEnIdx += 1

                            if tmpEnIdx % 100 == 0:
                                print('En chunk number %d has written' %
                                      tmpEnIdx)
            elif (isTranslated == 'T' and '.en.' in title):
                title = title.strip()
                with open('corpus/literature.EN-DE/books/%s' % title) as book:
                    for tokLine in book:
                        lineToks = tokLine.strip().split(' ')
                        frLines += 1
                        frToks += len(lineToks)

                        tmpFrChunk += tokLine
                        tmpFrToks += len(lineToks)

                        if tmpFrToks >= 2000:
                            frWriter.writerow(
                                [tmpFrIdx, tmpFrChunk, tmpFrToks])
                            tmpFrChunk = ''
                            tmpFrToks = 0
                            tmpFrIdx += 1

                            if tmpFrIdx % 100 == 0:
                                print('De chunk number %d has written' %
                                      tmpFrIdx)
        enWriter.writerow([tmpEnIdx, tmpEnChunk, tmpEnToks])
        frWriter.writerow([tmpFrIdx, tmpFrChunk, tmpFrToks])

    print('Literature', 'English Lines:', enLines)
    print('Literature', 'English Tokens:', enToks)
    print('Literature', 'German Lines:', frLines)
    print('Literature', 'German Tokens:', frToks)

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
    tokenizedChunk = chunk.strip().split(' ')
    stopwordsStat = dict()

    for token in tokenizedChunk:
        if token in functionWords:
            if token in stopwordsStat:
                stopwordsStat[token] += 1
            else:
                stopwordsStat[token] = 1

    for key in stopwordsStat.keys():
        stopwordsStat[key] /= int(chunkNum)

    return stopwordsStat


def getFeatures(nlp, corpusName):
    fEnChunks = open('chunks/%s/en-1.csv' % corpusName, 'r')
    fFrChunks = open('chunks/%s/de.csv' % corpusName, 'r')

    enReader = csv.reader(fEnChunks)
    frReader = csv.reader(fFrChunks)

    totalChunks = list()

    csvToTotalChunks(totalChunks, enReader, False, featuresLimit[corpusName])
    csvToTotalChunks(totalChunks, frReader, True, featuresLimit[corpusName])
    random.shuffle(totalChunks)

    print('Shuffled Chunks')

    fEnChunks.close()
    fFrChunks.close()

    stopwordsStats = list()
    for chunk, val, chunkNum in totalChunks:
        stopwordsStat = getStopWordsStat(nlp, chunk, chunkNum)
        stopwordsStats.append([stopwordsStat, val])

    print('Marking Function Words Complete')

    X = [[0 for _ in range(len(functionWords.keys()))]
         for _ in range(len(totalChunks))]
    y = [0 for _ in range(len(totalChunks))]

    print('Writing to csv File....')

    for i in range(len(totalChunks)):
        y[i] = 1 if stopwordsStats[i][1] else 0
        for j in range(len(functionWords)):
            stopwordsStat = stopwordsStats[i][0]
            if list(functionWords.keys())[j] in stopwordsStat:
                X[i][j] = stopwordsStat[list(functionWords.keys())[j]]

    fFeatures = open('features/%s/features-1.csv' % corpusName, 'w')
    featureWriter = csv.writer(fFeatures)
    featureWriter.writerow(list(functionWords.keys()) + ['val'])
    for i in range(len(totalChunks)):
        featureWriter.writerow(X[i] + [y[i]])
    fFeatures.close()

    return X, y


def getXy(corpusName):
    fFeatures = open('features/%s/features-1.csv' % corpusName, 'r')
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


if __name__ == '__main__':
    nlp = spacy.load('en')

    if 'europarl' in sys.argv:
        if 'features' not in sys.argv:
            chunkEuroparl(nlp)
        getFeatures(nlp, 'europarl')

    if 'literature' in sys.argv:
        if 'features' not in sys.argv:
            chunkLiterature(nlp)
        getFeatures(nlp, 'literature')
