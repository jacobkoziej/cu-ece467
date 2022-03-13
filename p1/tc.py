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
import random

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


class Collection:
    def __init__(self):
        self._cached  = True
        self._doc_cnt = 0
        self._doc_frq = { }
        self._idf     = { }

    def add_doc(self, tokens):
        self._cached   = False
        self._doc_cnt += 1

        for word in set(tokens):
            try:
                self._doc_frq[word] += 1
            except KeyError:
                self._doc_frq[word] = 1

    def cache(self):
        for word, frq in self._doc_frq.items():
            self._idf[word] = math.log10(self._doc_cnt / frq)

        self._cached = True

    def dot(self, vecu, vecv):
        if not self._cached:
            self.cache()

        inner_prod = 0.0

        for word in vecu:
            if word in vecv:
                inner_prod += vecu[word] * vecv[word] * (self._idf[word] ** 2)

        return inner_prod

    def norm(self, vec):
        if not self._cached:
            self.cache()

        radicand = 0.0

        for word, tf in vec.items():
            if word in self._idf:
                radicand += (tf * self._idf[word]) ** 2

        return math.sqrt(radicand)

    def sim(self, dot_prod, norm):
        try:
            return dot_prod / norm
        except ZeroDivisionError:
            return math.nan


class Database:
    def __init__(self):
        self.cat_norm   = { }
        self.cat_vec    = { }
        self.collection = Collection()
        self.processor  = Processor()


class Processor:
    def __init__(self, insensitive=False, stemming=False, stop_words=False):
        self.insensitive    = insensitive
        self.porter_stemmer = PorterStemmer()
        self.stemming       = stemming
        self.stop_words     = stop_words
        self.stopwords      = stopwords.words('english')

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

    def gen_vec(self, tokens):
        vec = { }

        for word in tokens:
            try:
                vec[word] += 1
            except KeyError:
                vec[word] = 1

        for word in vec:
            vec[word] = math.log10(vec[word] + 1)

        return vec

    def normalize(self, string):
        if self.insensitive:
            string = string.lower()

        tokens = word_tokenize(string)

        if self.stemming:
            tmp = [ ]
            for word in tokens:
                tmp.append(self.porter_stemmer.stem(word))

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

    def write_file_list(self, tuples, file):
        for (_, path) in tuples:
            file.write(f'{path}\n')


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
        collection = self.db.collection
        processor  = self.db.processor

        normalized = [ ]
        for path in processor.gen_file_list(file):
            if self.verbose:
                print(f"Normalizing: '{path}'")

            f = open(path, 'r')
            tokens = processor.normalize(f.read())
            f.close()

            normalized.append((path, tokens))

        uncat_vec = { }
        for (path, tokens) in normalized:
            if self.verbose:
                print(f"Generating vector: '{path}'")

            uncat_vec[path] = processor.gen_vec(tokens)

        uncat_norm = { }
        for path, vec in uncat_vec.items():
            if self.verbose:
                print(f"Normalizing vector: '{path}'")

            uncat_norm[path] = collection.norm(vec)

        cat_norm   = self.db.cat_norm
        cat_vec    = self.db.cat_vec
        categories = list(self.db.cat_vec.keys())
        predicted  = [ ]
        for path in uncat_vec:
            sim = { }
            for cat in categories:
                sim[cat] = collection.sim(
                    collection.dot(cat_vec[cat], uncat_vec[path]),
                    uncat_norm[path] * cat_norm[cat]
                )

                if self.verbose:
                    print(f"Similarity: '{path}' '{cat}' ==> {sim[cat]:.16f}")

            predicted.append((max(sim, key=sim.get), path))

        self.predict = predicted

    def write(self, file):
        self.db.processor.write_cat_file_tuples(self.predict, file)


class TestGenerator:
    def __init__(self):
        self.processor = Processor()

    def gen(self, file, prefix):
        processor = self.processor

        tuples = processor.gen_cat_file_tuples(file)
        tuple_cnt = (len(tuples) // 3) * 2

        random.shuffle(tuples)

        f = open(f'{prefix}_train.labels', 'w')
        processor.write_cat_file_tuples(tuples[:tuple_cnt], f)
        f.close()

        f = open(f'{prefix}_test.labels', 'w')
        processor.write_cat_file_tuples(tuples[tuple_cnt:], f)
        f.close()

        f = open(f'{prefix}_test.list', 'w')
        processor.write_file_list(tuples[tuple_cnt:], f)
        f.close()


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
        collection = self.db.collection
        processor  = self.db.processor

        cat_tokens = { }
        normalized = [ ]

        for (cat, path) in processor.gen_cat_file_tuples(labels):
            if self.verbose:
                print(f"Normalizing: '{path}' '{cat}'")

            f = open(path, 'r')
            tokens = processor.normalize(f.read())
            f.close()

            try:
                cat_tokens[cat] += tokens
            except KeyError:
                cat_tokens[cat]  = [ ]
                cat_tokens[cat] += tokens

            normalized.append((cat, path, tokens))

        for (cat, path, tokens) in normalized:
            if self.verbose:
                print(f"Adding to collection: '{path}' '{cat}'")

            collection.add_doc(tokens)

        if self.verbose:
            print(f"Calculating collection idfs")

        collection.cache()

        for cat, tokens in cat_tokens.items():
            if self.verbose:
                print(f"Generating vector: '{cat}'")

            self.db.cat_vec[cat] = processor.gen_vec(tokens)

        for cat, vec in self.db.cat_vec.items():
            if self.verbose:
                print(f"Normalizing vector: '{cat}'")

            self.db.cat_norm[cat] = collection.norm(vec)
