import spacy
import sys
import csv

europarlInfo = {
    'dat': 'corpus/europarl-v7.EN-FR/europarl-v7.fr-en.dat',
    'tok': 'corpus/europarl-v7.EN-FR/europarl-v7.fr-en.en.aligned.tok'
}


def calcStopWords(toks, tokDict):
    for tok in toks:
        if not tok.is_stop:
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


if __name__ == '__main__':
    nlp = spacy.load('en')

    if 'europarl' in sys.argv:
        chunkEuroparl(nlp)
