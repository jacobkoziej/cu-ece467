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


class Vector:
    class WordWeight:
        def __init__(self):
            self.tc    = 0     # term count
            self.tf    = None  # term frequency log10(tc)
            self.df    = 0     # document frequency
            self.idf   = None  # inverse document frequency log10(doc_cnt/df)
            self.tfidf = None  # word weight tf * idf

    def __init__(self):
        self.doc_cnt = 0
        self.feat    = { }
        self.norm    = None


class Trainer:
    def __init__(self, labels):
        self.labels = labels
