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

import nltk.tokenize


class Vector:
    class WordWeight:
        def __init__(self):
            self.tc    = 0     # term count
            self.tf    = None  # term frequency log10(tc + 1)
            self.df    = 0     # document frequency
            self.idf   = None  # inverse document frequency log10(doc_cnt/df)
            self.tfidf = None  # word weight tf * idf

    def __init__(self):
        self.clear()

    def _calc_word_weight(self):
        if self.doc_cnt > 1:
            for _, word in self.feat.items():
                word.tf = math.log10(word.tc + 1)
                word.idf = math.log10(self.doc_cnt / word.df)
                word.tfidf = word.tf * word.idf
        else:
            for _, word in self.feat.items():
                word.tf = math.log10(word.tc + 1)

    def _doc_process(self, doc):
        self.doc_cnt += 1

        uniq_feat = set(doc)

        for feat in uniq_feat:
            try:
                self.feat[feat].df += 1
            except KeyError:
                self.feat[feat] = self.WordWeight()
                self.feat[feat].df += 1

        for word in doc:
            self.feat[word].tc += 1

    def calc_norm(self, parent=None):
        norm = 0

        if parent is None:
            for _, feat in self.feat.items():
                norm += feat.tfidf ** 2
        else:
            for word in self.feat:
                if word in parent.feat:
                    norm += (self.feat[word].tf * parent.feat[word].idf) ** 2

        return math.sqrt(norm)

    def clear(self):
        self.doc_cnt = 0
        self.feat    = { }
        self.norm    = None

    def process(self, raw):
        if isinstance(raw[0], list):
            for doc in raw:
                self._doc_process(doc)

            self._calc_word_weight()
            self.norm = self.calc_norm()

        else:
            self._doc_process(raw)
            self._calc_word_weight()



class Trainer:
    def __init__(self, labels):
        self.labels     = labels
        self.categories = { }

    def _category_tokenize(self):
        categories = { }

        for path, category in self.labels:
            file   = open(path, 'r')
            tokens = nltk.tokenize.word_tokenize(file.read())
            file.close()

            try:
                categories[category].append(tokens)
            except KeyError:
                categories[category] = [tokens]

        return categories

    def train(self):
        cat_tokens = self._category_tokenize()

        for category, tokens in cat_tokens.items():
            vec = Vector()
            vec.process(tokens)
            self.categories[category] = vec

    def export(self, db):
        pickle.dump(self.categories, db)
