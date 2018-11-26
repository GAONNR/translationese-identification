import pandas as pd
import argparse
from function_words import function_words


class LangDesc:
    def __init__(self):
        self.line_idx = -1
        self.clear()

    def process(self, token_line):
        tmp_fws, tmp_fw_num = self.get_fws(token_line)
        self.original_chunk.append(token_line)
        self.fw_chunk += tmp_fws
        self.num_of_fws += tmp_fw_num

    def get_fws(self, token_line):
        tokens = token_line.strip().split(' ')
        tmp_fws = list()

        for tok in tokens:
            if tok in function_words:
                tmp_fws.append(tok)

        return tmp_fws, len(tmp_fws)

    def is_full(self, max_words):
        if self.num_of_fws >= max_words:
            return True
        return False

    def get_row(self):
        return [' '.join(self.original_chunk),
                ' '.join(self.fw_chunk),
                self.num_of_fws]

    def clear(self):
        self.original_chunk = list()
        self.fw_chunk = list()
        self.num_of_fws = 0
        self.line_idx += 1

        if self.line_idx % 100 == 0:
            print('%dst en chunk has added to dataframe' % self.line_idx)


def chunk_europarl(max_words):
    en_desc = LangDesc()
    fr_desc = LangDesc()

    en_dataframe = pd.DataFrame(
        columns=['original text', 'fws only', 'number of fws'])
    fr_dataframe = pd.DataFrame(
        columns=['original text', 'fws only', 'number of fws'])

    with open('corpus/europarl-v7.EN-FR/europarl-v7.fr-en.dat', 'r') as datafile, \
            open('corpus/europarl-v7.EN-FR/europarl-v7.fr-en.en.aligned.tok', 'r') as tokfile:
        for dat_line, token_line in zip(datafile, tokfile):
            if '\"EN\"' in dat_line:
                en_desc.process(token_line)
                if en_desc.is_full(max_words):
                    en_dataframe.loc[en_desc.line_idx] = en_desc.get_row()
                    en_desc.clear()
            elif '\"Fr\"' in dat_line:
                fr_desc.process(token_line)
                if fr_desc.is_full(max_words):
                    fr_dataframe.loc[fr_desc.line_idx] = fr_desc.get_row()
                    fr_desc.clear()

    en_dataframe.to_csv('chunks/europarl/en_lstm.csv')
    fr_dataframe.to_csv('chunks/europarl/fr_lstm.csv')

    return 'chunks/europarl/'


def chunk_literature(max_words):
    return 'chunks/literature'


def get_features(args):
    chunk_path = ''
    if args.corpus == 'europarl':
        chunk_path = chunk_europarl(args.max_words)
    elif args.corpus == 'literature':
        chunk_path = chunk_literature(args.max_words)
    else:
        print('no proper corpus')
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--corpus', help='name of the corpus', default='europarl')
    parser.add_argument(
        '--max_words', help='number of the words to be included in one feature', type=int, default=500)
    args = parser.parse_args()

    get_features(args)
