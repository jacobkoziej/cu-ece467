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

from nltk.tokenize import word_tokenize


class Database:
    def __init__(self):
        self.cat = { }


class Tester:
    pass


class Trainer:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def gen_file_tuples(self, file):
        tuples = [ ]
        for line in file.readlines():
            tmp = line.strip().split()
            tmp.reverse()
            tuples.append(tuple(tmp))

        return tuples

    def gen_token_tuples(self, files):
        tuples = [ ]
        for (cat, path) in files:
            if self.verbose:
                print(f'Tokenizing file: {cat} {path}')

            f = open(path, 'r')
            tokens = word_tokenize(f.read())
            tuples.append((cat, tokens))
            f.close()

        return tuples


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
        self.cache   = False
        self.doc_cnt = 0
        self.feat    = { }
        self.norm    = None

    def __str__(self):
        return str(vars(self))

    def add_doc(self, tokens):
        self.cache = False
        self.doc_cnt += 1

        for word in set(tokens):
            try:
                self.feat[word].df += 1
            except KeyError:
                self.feat[word] = Vector()
                self.feat[word].df += 1
