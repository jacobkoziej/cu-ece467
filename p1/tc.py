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

import nltk.tokenize


class Vector:
    class WordWeight:
        def __init__(self):
            self.tc    = 0     # term count
            self.tf    = None  # term frequency log10(tc)
            self.df    = 0     # document frequency
            self.idf   = None  # inverse document frequency log10(doc_cnt/df)
            self.tfidf = None  # word weight tf * idf

    def __init__(self, raw):
        self.doc_cnt = 0
        self.feat    = { }
        self.norm    = None

        if isinstance(raw[0], list):
            for doc in raw:
                self._doc_process(doc)
        else:
            self._doc_process(raw)

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

        for category in cat_tokens:
            self.categories[category] = Vector(cat_tokens[category])
