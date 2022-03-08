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

from nltk.tokenize import word_tokenize


class Database:
    def __init__(self):
        self.cat       = { }
        self.processor = Processor()


class Processor:
    def gen_cat_file_tuples(self, file):
        tuples = [ ]
        for line in file.readlines():
            tmp = line.strip().split()
            tmp.reverse()
            tuples.append(tuple(tmp))

        return tuples

    def tokenize(self, string):
        return word_tokenize(string)


class Tester:
    def __init__(self, db=None, verbose=False):
        self.db      = db
        self.verbose = verbose

    def load(self, file):
        if self.verbose:
            print(f'Importing database from file: {file.name}')

        self.db = pickle.load(file)


class Trainer:
    def __init__(self, verbose=False):
        self.db      = Database()
        self.verbose = verbose

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
                self.db.cat[cat].add_doc(tokens)
            except KeyError:
                self.db.cat[cat] = Vector()
                self.db.cat[cat].add_doc(tokens)

        for cat, vec in self.db.cat.items():
            if self.verbose:
                print(f'Caching vector: {cat}')

            vec.cache()


class Vector:
    class WordWeight:
        def __init__(self):
            self.tc    = 0     # term count
            self.tf    = None  # term frequency
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
            word.tf = math.log10(word.tc + 1)
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
            self.feat[word].tc += 1

    def cache(self):
        self._calc_word_weight()
        self._calc_norm()
        self.cached = True

    def sim(self, vec):
        if not self.cached:
            self.cache()

        if not vec.cached:
            vec.cache()

        dot_prod = Vector._calc_dot_prod(self, vec)
        norm = self.norm * vec._calc_norm(self)

        return dot_prod / norm
