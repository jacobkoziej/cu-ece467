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
    rules:      dict[str, list[(str, str)]] = field(default_factory=dict, kw_only=True)
    terminals:  dict[str, list[str]]        = field(default_factory=dict, kw_only=True)
    start_symb: str                         = field(default='S',          kw_only=True)

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

    def parse(self, input: list[str]) -> list[tuple]:
        if not input:
            return [ ]

        # gen triangular matrix
        matrix = [ ]
        for i in range(len(input)):
            matrix.append([ ])
            for j in range(i + 1):
                matrix[i].append([ ])

        # tag terminal cells
        for i in range(len(input)):
            for rule, terminals in self.terminals.items():
                for lex in terminals:
                    if lex == input[i]:
                        matrix[i][i].append((rule, lex))
                        break

            # lexicon is not in our grammar
            if not matrix[i][i]:
                return [ ]

        # parse upper triangular cells
        for i in range(1, len(input)):
            for j in reversed(range(i)):
                for k in reversed(range(j + 1, i + 1)):
                    for l in range(k - 1, i):
                        # possible rules for cell i,j
                        rules_l = matrix[l][j]
                        rules_r = matrix[i][k]

                        # skip empty cells
                        if not rules_l or not rules_r:
                            continue

                        for rule_tup_l in rules_l:
                            for rule_tup_r in rules_r:
                                # possible rule from cells l,j and i,k
                                rule = (rule_tup_l[0], rule_tup_r[0])

                                for name, rules in self.rules.items():
                                    if rule in rules:
                                        matrix[i][j].append((name, (rule_tup_l, rule_tup_r)))

        # collect valid parses
        parses = [ ]
        for parse in matrix[len(input) - 1][0]:
            if self.start_symb == parse[0]:
                parses.append(parse)

        return parses


@dataclass
class Interactive:
    grammar:    Grammar = field(default_factory=Grammar, kw_only=True)
    indent_str: str     = field(default='    ',          kw_only=True)

    def parse_grammar(self, file_path: str):
        with open(file_path) as grammar:
            for line in grammar.readlines():
                tokens = line.split()

                match len(tokens):
                    case 4:
                        self.grammar.add_rule(tokens[0], tokens[2], tokens[3])
                    case 3:
                        self.grammar.add_terminal(tokens[0], tokens[2])
                    case _:
                        pass

    def parse_str(self, rule: tuple) -> str:
        if type(rule[1]) is str:
            return '[' + rule[0] + ' ' + rule[1] + ']'

        out  = '[' + rule[0]
        out += ' '
        out += self.parse_str(rule[1][0])
        out += ' '
        out += self.parse_str(rule[1][1])
        out += ']'

        return out

    def parse_tree_str(self, rule: tuple, indent: int = 0) -> str:
        indent_str = self.indent_str * indent

        if type(rule[1]) is str:
            return indent_str + '[' + rule[0] + ' ' + rule[1] + ']'

        out  = indent_str + '[' + rule[0]
        out += '\n'
        out += self.parse_tree_str(rule[1][0], indent + 1)
        out += '\n'
        out += self.parse_tree_str(rule[1][1], indent + 1)
        out += '\n'
        out += indent_str + ']'

        return out
