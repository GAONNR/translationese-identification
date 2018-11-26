from function_words import function_words as functionWords
import random
import spacy
import sys
import csv

europarlInfo = {
    'dat': 'corpus/europarl-v7.EN-FR/europarl-v7.fr-en.dat',
    'tok': 'corpus/europarl-v7.EN-FR/europarl-v7.fr-en.en.aligned.tok'
}

featuresLimit = {
    'europarl': 1500,
    'literature': 400
}


def chunkEuroparl(nlp):
    enLines = 0
    enToks = 0
    tmpEnChunk = ''
    tmpEnToks = 0
    tmpEnIdx = 0
    fEnChunks = open('chunks/europarl/en.csv', 'w')
    enWriter = csv.writer(fEnChunks)

    frLines = 0
    frToks = 0
    tmpFrChunk = ''
    tmpFrToks = 0
    tmpFrIdx = 0
    fFrChunks = open('chunks/europarl/fr.csv', 'w')
    frWriter = csv.writer(fFrChunks)

    with open(europarlInfo['dat'], 'r') as corpusDat, \
            open(europarlInfo['tok'], 'r') as corpusTok:
        for datLine, tokLine in zip(corpusDat, corpusTok):
            if '\"EN\"' in datLine:
                lineToks = nlp(tokLine)
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
                lineToks = nlp(tokLine)
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
                        print('Fr chunk number %d has written' % tmpFrIdx)

        enWriter.writerow([tmpEnIdx, tmpEnChunk, tmpEnToks])
        frWriter.writerow([tmpFrIdx, tmpFrChunk, tmpFrToks])

        print('Europarl', 'English Lines:', enLines)
        print('Europarl', 'English Tokens:', enToks)
        print('Europarl', 'French Lines:', frLines)
        print('Europarl', 'French Tokens:', frToks)

    fEnChunks.close()
    fFrChunks.close()
    return


def chunkLiterature(nlp):
    enLines = 0
    enToks = 0
    tmpEnChunk = ''
    tmpEnToks = 0
    tmpEnIdx = 0
    fEnChunks = open('chunks/literature/en.csv', 'w')
    enWriter = csv.writer(fEnChunks)

    frLines = 0
    frToks = 0
    tmpFrChunk = ''
    tmpFrToks = 0
    tmpFrIdx = 0
    fFrChunks = open('chunks/literature/fr.csv', 'w')
    frWriter = csv.writer(fFrChunks)

    with open('corpus/literature.EN-FR/literature.dat', 'r') as fLiteratureDat:
        for datLine in fLiteratureDat:
            if len(datLine.split(' ')) < 2:
                continue
            isTranslated, title = datLine.split(' ')
            if (isTranslated == 'S' and '.en.' in title):
                title = title.strip()
                with open('corpus/literature.EN-FR/books/%s' % title) as book:
                    for tokLine in book:
                        lineToks = nlp(tokLine)
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
                with open('corpus/literature.EN-FR/books/%s' % title) as book:
                    for tokLine in book:
                        lineToks = nlp(tokLine)
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
                                print('Fr chunk number %d has written' %
                                      tmpFrIdx)
        enWriter.writerow([tmpEnIdx, tmpEnChunk, tmpEnToks])
        frWriter.writerow([tmpFrIdx, tmpFrChunk, tmpFrToks])

    print('Literature', 'English Lines:', enLines)
    print('Literature', 'English Tokens:', enToks)
    print('Literature', 'French Lines:', frLines)
    print('Literature', 'French Tokens:', frToks)

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
        if token.text in functionWords:
            if token.text in stopwordsStat:
                stopwordsStat[token.text] += 1
            else:
                stopwordsStat[token.text] = 1

    for key in stopwordsStat.keys():
        stopwordsStat[key] /= int(chunkNum)

    return stopwordsStat


def getFeatures(nlp, corpusName):
    fEnChunks = open('chunks/%s/en.csv' % corpusName, 'r')
    fFrChunks = open('chunks/%s/fr.csv' % corpusName, 'r')

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

    X = [[0 for _ in range(len(functionWords))]
         for _ in range(len(totalChunks))]
    y = [0 for _ in range(len(totalChunks))]

    print('Writing to csv File....')

    for i in range(len(totalChunks)):
        y[i] = 1 if stopwordsStats[i][1] else 0
        for j in range(len(functionWords)):
            stopwordsStat = stopwordsStats[i][0]
            if functionWords[j] in stopwordsStat:
                X[i][j] = stopwordsStat[functionWords[j]]

    fFeatures = open('features/%s/features.csv' % corpusName, 'w')
    featureWriter = csv.writer(fFeatures)
    featureWriter.writerow(functionWords + ['val'])
    for i in range(len(totalChunks)):
        featureWriter.writerow(X[i] + [y[i]])
    fFeatures.close()

    return X, y


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
