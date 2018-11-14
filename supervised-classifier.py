import nltk
import sys

stops = None

def pretty_print_dict(mydict):
    print('{')
    for key in mydict.keys():
        if mydict[key] > 0:
            print(' \"' + key + '\":', mydict[key], ',')
    print('}')

def num_of_FWs(tokens, stops):
    FWNum = 0
    FWDict = dict()
    for token in tokens:
        if token in stops:
            FWNum += 1

            if token in FWDict:
                FWDict[token] += 1
            else:
                FWDict[token] = 0

    return FWNum, FWDict

def tokenizer():
    nativeTKNum = 0
    nativeFWNum = 0
    nativeLNNum = 0
    nativeFWDict = dict()
    for word in stops:
        nativeFWDict[word] = 0

    with open('corpus/ENNTT/natives.tok', 'r') as fNatives:
        for line in fNatives:
            lineTokens = nltk.word_tokenize(line)
            nativeTKNum += len(lineTokens)
            
            tmpTuple = num_of_FWs(lineTokens, stops)
            nativeFWNum += tmpTuple[0]
            for key in tmpTuple[1].keys():
                nativeFWDict[key] += tmpTuple[1][key]

            nativeLNNum += 1
        
        print('=== Natives ===')
        print('Number of tokens:', nativeTKNum)
        print('Ratio of FWs:', nativeFWNum / nativeTKNum)
        print('Number of sentences:', nativeLNNum)
        print('Dict of FW freq:')
        pretty_print_dict(nativeFWDict)

    translationTKNum = 0
    translationFWNum = 0
    translationLNNum = 0
    translationFWDict = dict()
    for word in stops:
        translationFWDict[word] = 0

    with open('corpus/ENNTT/translations.tok', 'r') as fTranslations, \
         open('corpus/ENNTT/translations.dat', 'r') as fTransData: 
        for line, dat in zip(fTranslations, fTransData):
            if 'LANGUAGE=\"FR\"' in dat: 
                lineTokens = nltk.word_tokenize(line)
                translationTKNum += len(lineTokens)

                tmpTuple = num_of_FWs(lineTokens, stops)
                translationFWNum += tmpTuple[0]
                for key in tmpTuple[1].keys():
                    translationFWDict[key] += tmpTuple[1][key]

                translationLNNum += 1

        print('=== Translations ===')
        print('Number of tokens:', translationTKNum)
        print('Ratio of FWs:', translationFWNum / translationTKNum)
        print('Number of sentences:', translationLNNum)
        print('Dict of FW freq:')
        pretty_print_dict(translationFWDict)
    
    return (None, None)

if __name__ == '__main__':
    RUN = True
    if len(sys.argv) > 1:
        if 'setup' in sys.argv:
            nltk.download('punkt')
            nltk.download('stopwords')
            RUN = False
    
    if RUN:
        from nltk.corpus import stopwords
        stops = stopwords.words('english')

        nativeTokens, traslationTokens = tokenizer()