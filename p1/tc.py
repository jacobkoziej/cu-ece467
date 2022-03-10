# tc.py -- text categorization
# Copyright (C) 2022  Jacob Koziej <jacobkoziej@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import math
import pickle

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


class Collection:
    def __init__(self):
        self.doc_cnt = 0
        self.doc_frq = { }
        self.idf     = { }

    def add_doc(self, tokens):
        self.doc_cnt += 1

        for word in set(tokens):
            try:
                self.doc_frq[word] += 1
            except KeyError:
                self.doc_frq[word] = 1


class Database:
    def __init__(self):
        self.cat_vec    = { }
        self.collection = Collection()
        self.processor  = Processor()


class Processor:
    stopwords = stopwords.words('english')

    def __init__(self, insensitive=False, stemming=False, stop_words=False):
        self.insensitive = insensitive
        self.stemming    = stemming
        self.stop_words  = stop_words

    def gen_cat_file_tuples(self, file):
        tuples = [ ]
        for line in file.readlines():
            tmp = line.strip().split()
            tmp.reverse()
            tuples.append(tuple(tmp))

        return tuples

    def gen_file_list(self, file):
        list = [ ]
        for line in file.readlines():
            list.append(line.strip())

        return list

    def tokenize(self, string):
        if self.insensitive:
            string = string.lower()

        tokens = word_tokenize(string)

        if self.stemming:
            tmp = [ ]
            for word in tokens:
                ps = PorterStemmer()
                tmp.append(ps.stem(word))

            tokens = tmp

        if self.stop_words:
            filtered = [ ]
            for word in tokens:
                if word not in self.stopwords:
                    filtered.append(word)

            tokens = filtered

        return tokens

    def write_cat_file_tuples(self, tuples, file):
        for (cat, path) in tuples:
            file.write(f'{path} {cat}\n')


class Tester:
    def __init__(self, db=None, verbose=False):
        self.db      = db
        self.predict = [ ]
        self.verbose = verbose

    def load(self, file):
        if self.verbose:
            print(f'Importing database from file: {file.name}')

        self.db = pickle.load(file)

    def test(self, file):
        processor = self.db.processor

        token_tuples = [ ]
        for path in processor.gen_file_list(file):
            if self.verbose:
                print(f'Tokenizing file: {path}')

            f = open(path)
            token_tuples.append((path, processor.tokenize(f.read())))
            f.close()

        vec_tuples = [ ]
        for (path, tokens) in token_tuples:
            if self.verbose:
                print(f'Caching vector: {path}')

                vec = Vector()
                vec.add_doc(tokens)
                vec.cache()
                vec_tuples.append((path, vec))

        categories = list(self.db.cat_vec.keys())
        cat_tuples = [ ]
        for (path, vec) in vec_tuples:
            similarities = [ ]
            for cat in categories:
                similarities.append(Vector.sim(self.db.cat_vec[cat], vec))

            cat = categories[similarities.index(max(similarities))]

            if self.verbose:
                print(f'Labeled file: {cat} {path}')

            cat_tuples.append((cat, path))

        self.predict = cat_tuples

    def write(self, file):
        self.db.processor.write_cat_file_tuples(self.predict, file)


class Trainer:
    def __init__(self, insensitive=True, stemming=True, stop_words=True, verbose=False):
        self.db      = Database()
        self.verbose = verbose

        p = self.db.processor
        p.insensitive = insensitive
        p.stemming    = stemming
        p.stop_words  = stop_words

    def dump(self, file):
        if self.verbose:
            print(f'Dumping database to file: {file.name}')

        pickle.dump(self.db, file)

    def train(self, labels):
        processor  = self.db.processor
        normalized = [ ]

        for (cat, path) in processor.gen_cat_file_tuples(labels):
            if self.verbose:
                print(f'Tokenizing file: {cat} {path}')

            f = open(path, 'r')
            normalized.append((cat, processor.tokenize(f.read())))
            f.close()

        for (cat, tokens) in normalized:
            try:
                self.db.cat_vec[cat].add_doc(tokens)
            except KeyError:
                self.db.cat_vec[cat] = Vector()
                self.db.cat_vec[cat].add_doc(tokens)

        for cat, vec in self.db.cat_vec.items():
            if self.verbose:
                print(f'Caching vector: {cat}')

            vec.cache()


class Vector:
    class WordWeight:
        def __init__(self):
            self.tf    = 0     # term frequency
            self.df    = 0     # document frequency
            self.idf   = None  # inverse document frequency
            self.tfidf = None  # word weight

        def __str__(self):
            return str(vars(self))

    def __init__(self):
        self.cached  = False
        self.doc_cnt = 0
        self.feat    = { }
        self.norm    = None

    def __str__(self):
        return str(vars(self))

    def _calc_dot_prod(self, inherit):
        sum = 0.0
        for word, feat in self.feat.items():
            if word in inherit.feat:
                sum += feat.tf * inherit.feat[word].tf \
                    * (inherit.feat[word].idf ** 2)

        return sum

    def _calc_norm(self, inherit=None):
        sum = 0.0

        if inherit is None:
            for _, feat in self.feat.items():
                sum += feat.tfidf ** 2

            self.norm = math.sqrt(sum)
            return self.norm
        else:
            for word, feat in self.feat.items():
                if word in inherit.feat:
                    sum += (feat.tf * inherit.feat[word].idf) ** 2

            return math.sqrt(sum)

    def _calc_word_weight(self):
        for _, word in self.feat.items():
            word.idf = math.log10(self.doc_cnt / word.df)
            word.tfidf = word.tf * word.idf

    def add_doc(self, tokens):
        self.cached = False
        self.doc_cnt += 1

        for word in set(tokens):
            try:
                self.feat[word].df += 1
            except KeyError:
                self.feat[word] = self.WordWeight()
                self.feat[word].df += 1

        for word in tokens:
            self.feat[word].tf += 1

    def cache(self):
        self._calc_word_weight()
        self._calc_norm()
        self.cached = True

    def sim(self, vec):
        if not self.cached:
            self.cache()

        if not vec.cached:
            vec.cache()

        dot_prod = vec._calc_dot_prod(self)
        norm = self.norm * vec._calc_norm(self)

        return dot_prod / norm
