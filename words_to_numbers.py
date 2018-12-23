from function_words import function_words
import pandas as pd
import argparse

columns = ['original text', 'fws only', 'numbered fws', 'number of fws']


class LangDesc:
    def __init__(self, name):
        self.name = name
        self.line_idx = -1
        self.clear()

    def process(self, token_line):
        tmp_fws, tmp_fw_num = self.get_fws(token_line)
        self.original_chunk.append(token_line)
        self.fw_chunk += tmp_fws
        self.numbered_chunk += list(
            map(lambda x: str(function_words[x]), tmp_fws))
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
                ' '.join(self.numbered_chunk),
                self.num_of_fws]

    def clear(self):
        self.original_chunk = list()
        self.fw_chunk = list()
        self.numbered_chunk = list()
        self.num_of_fws = 0
        self.line_idx += 1

        if self.line_idx % 500 == 0:
            print('%dth %s chunk has added to dataframe' %
                  (self.line_idx, self.name))


def chunk_europarl(max_words, language):
    en_desc = LangDesc('en')
    fr_desc = LangDesc(language)

    en_dataframe = pd.DataFrame(
        columns=columns)
    fr_dataframe = pd.DataFrame(
        columns=columns)

    with open('corpus/europarl-v7.EN-%s/europarl-v7.%s-en.dat' % (language.upper(), language.upper()), 'r') as datafile, \
            open('corpus/europarl-v7.EN-%s/europarl-v7.%s-en.en.aligned.tok' % (language.upper(), language.upper()), 'r') as tokfile:
        for dat_line, token_line in zip(datafile, tokfile):
            if '\"EN\"' in dat_line:
                en_desc.process(token_line)
                if en_desc.is_full(max_words):
                    en_dataframe.loc[en_desc.line_idx] = en_desc.get_row()
                    en_desc.clear()
            elif '\"%s\"' % language.upper() in dat_line:
                fr_desc.process(token_line)
                if fr_desc.is_full(max_words):
                    fr_dataframe.loc[fr_desc.line_idx] = fr_desc.get_row()
                    fr_desc.clear()

    en_dataframe.to_csv('chunks/europarl/en-%s/en_lstm.csv' % language)
    fr_dataframe.to_csv('chunks/europarl/en-%s/%s_lstm.csv' %
                        (language, language))

    return 'chunks/europarl/'


def chunk_literature(max_words, language):
    en_desc = LangDesc('en')
    fr_desc = LangDesc(language)

    en_dataframe = pd.DataFrame(
        columns=columns)
    fr_dataframe = pd.DataFrame(
        columns=columns)

    with open('corpus/literature.EN-%s/literature.dat' % language.upper(), 'r') as datafile:
        for dat_line in datafile:
            if len(dat_line.split(' ')) < 2:
                continue
            is_translated, title = dat_line.split(' ')
            if is_translated == 'S' and '.en.' in title:
                title = title.strip()
                with open('corpus/literature.EN-%s/books/%s' % (language.upper(), title)) as book:
                    for token_line in book:
                        en_desc.process(token_line)
                        if en_desc.is_full(max_words):
                            en_dataframe.loc[en_desc.line_idx] = en_desc.get_row(
                            )
                            en_desc.clear()
            elif is_translated == 'T' and '.en.' in title:
                title = title.strip()
                with open('corpus/literature.EN-%s/books/%s' % (language.upper(), title)) as book:
                    for token_line in book:
                        fr_desc.process(token_line)
                        if fr_desc.is_full(max_words):
                            fr_dataframe.loc[fr_desc.line_idx] = fr_desc.get_row(
                            )
                            fr_desc.clear()

    en_dataframe.to_csv('chunks/literature/en-%s/en_lstm.csv' % language)
    fr_dataframe.to_csv('chunks/literature/en-%s/%s_lstm.csv' %
                        (language, language))

    return 'chunks/literature'


def get_features(args):
    print(args)
    if args.language not in ('fr', 'de'):
        print('incorrect language name')
        return

    if args.corpus == 'europarl':
        print('\nparsing europarl corpus')
        print('=== === === === === ===')
        chunk_europarl(args.max_words, args.language)
    elif args.corpus == 'literature':
        print('\nparsing literature corpus')
        print('=== === === === === ===')
        chunk_literature(args.max_words, args.language)
    elif args.corpus == 'all':
        print('\nparsing europarl corpus')
        print('=== === === === === ===')
        chunk_europarl(args.max_words, args.language)
        print('\nparsing literature corpus')
        print('=== === === === === ===')
        chunk_literature(args.max_words, args.language)
    else:
        print('no proper corpus')
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--corpus', help='name of the corpus', default='europarl')
    parser.add_argument(
        '--max_words', help='number of the words to be included in one feature', type=int, default=500)
    parser.add_argument(
        '--language', help='fr or de', default='fr')
    args = parser.parse_args()

    get_features(args)
