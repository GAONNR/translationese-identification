import nltk
import sys

stops = None

def tokenizer():
    with open('corpus/ENNTT/natives.tok', 'r') as fNatives:
        for line in fNatives:
            print(nltk.word_tokenize(line))
            break
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

        nativeTokens, traslationTokens = tokenizer();