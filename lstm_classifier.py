from keras.preprocessing import sequence
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from function_words import function_words
import pandas as pd
import argparse
import sklearn
import keras

#                                           original text      ...      number of fws
#
#
# 0     It is not a lot to ask . \n I therefore repeat...      ...                504
# 1     Most committee members have sought rather to f...      ...                516
# 2     The von Wogau report proposes referring superv...      ...                505
# 3     But the idea of a European Public Prosecutor, ...      ...                513
# 4     I shall point out that a directive on the appr...      ...                513
# 5     Curiously, a number of employers agreed with  ...      ...                508

columns = ['original text', 'fws only', 'numbered fws', 'number of fws']


def get_dataframes(corpus):
    en_df = pd.read_csv('chunks/%s/en_lstm.csv' % corpus)
    en_df = en_df.drop(columns=en_df.columns[0])

    fr_df = pd.read_csv('chunks/%s/fr_lstm.csv' % corpus)
    fr_df = fr_df.drop(columns=fr_df.columns[0])

    return en_df, fr_df


def get_features(args):
    max_words = args.max_words

    en_df, fr_df = get_dataframes(args.corpus)

    en_X = list(map(lambda x: list(map(int, x.split(' '))),
                    en_df['numbered fws']))
    en_X = sequence.pad_sequences(en_X, max_words, truncating='post')

    fr_X = list(map(lambda x: list(map(int, x.split(' '))),
                    fr_df['numbered fws']))
    fr_X = sequence.pad_sequences(fr_X, max_words, truncating='post')

    en_Xydf = pd.DataFrame(data=en_X, columns=['X%d' % i for i in range(500)])
    en_Xydf['y'] = False
    fr_Xydf = pd.DataFrame(data=fr_X, columns=['X%d' % i for i in range(500)])
    fr_Xydf['y'] = True

    mixed_Xydf = en_Xydf.append(fr_Xydf, ignore_index=True)
    mixed_Xydf = sklearn.utils.shuffle(mixed_Xydf).reset_index(drop=True)
    X = mixed_Xydf.drop(columns=['y']).values
    y = mixed_Xydf['y'].values

    return X, y


def create_model(input_length):
    embedding_size = 32
    model = Sequential()
    model.add(Embedding(len(function_words),
                        embedding_size, input_length=input_length))
    model.add(LSTM(100, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    return model


def lstm_train(args):
    if args.corpus not in ('europarl', 'literature'):
        print('incorrect corpus name')
        return

    X, y = get_features(args)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    model = create_model(len(X_train[0]))
    model.fit(X_train, y_train, validation_data=(
        X_test, y_test), batch_size=32, epochs=10, verbose=1)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', scores[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--corpus', help='name of the corpus', default='europarl')
    parser.add_argument(
        '--max_words', help='number of the words to be included in one feature', type=int, default=500)
    args = parser.parse_args()

    lstm_train(args)
