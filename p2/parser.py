# parser.py --  Cocke-Kasami-Younger (CKY) parser
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

from dataclasses import dataclass, field


@dataclass
class Grammar:
    rules:     dict[str, list[(str, str)]] = field(default_factory=dict, kw_only=True)
    terminals: dict[str, list[str]]        = field(default_factory=dict, kw_only=True)

    def add_rule(self, rule: str, nterma: str, ntermb: str):
        nterm = (nterma, ntermb)

        try:
            self.rules[rule].append(nterm)
        except KeyError:
            self.rules[rule] = [nterm]

    def add_terminal(self, rule: str, term: str):
        try:
            self.terminals[rule].append(term)
        except KeyError:
            self.terminals[rule] = [term]


class Interactive:
    def __init__(self, grammar: Grammar):
        self.grammar = grammar
