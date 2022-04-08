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

        matrix = [[ [] for i in range(len(input) + 1) ] for j in range(len(input) + 1)]

        for j in range(1, len(input) + 1):
            for rule, terminals in self.terminals.items():
                if input[j - 1] in terminals:
                    matrix[j - 1][j].append((rule, input[j - 1]))

            # lexicon is not in our grammar
            if not matrix[j - 1][j]:
                return [ ]

            for i in reversed(range(j - 1)):
                for k in range(i + 1, j):
                    rules_l = matrix[i][k]
                    rules_r = matrix[k][j]

                    # skip empty cells
                    if not rules_l or not rules_r:
                        continue

                    for rule_tup_l in rules_l:
                        for rule_tup_r in rules_r:
                            # possible rule from cells i,k and k,j
                            rule = (rule_tup_l[0], rule_tup_r[0])

                            for name, rules in self.rules.items():
                                if rule in rules:
                                    matrix[i][j].append((name, (rule_tup_l, rule_tup_r)))

        # collect valid parses
        parses = [ ]
        for parse in matrix[0][len(input)]:
            if self.start_symb == parse[0]:
                parses.append(parse)

        return parses


@dataclass
class Cli:
    grammar:    Grammar = field(default_factory=Grammar, kw_only=True)
    indent_str: str     = field(default='    ',          kw_only=True)
    prompt:     str     = field(default='(p2)',          kw_only=True)

    def cli(self, parse_tree=False):
        parse_str = self.parse_tree_str if parse_tree else self.parse_str

        print(
            'p2.py -- programming assignement #2 (parsing)\n'
            "for help, type 'help'\n"
            "to quit, type 'quit'\n"
            "to parse a sentence, type 'parse'"
        )

        while True:
            try:
                opt = input(self.prompt + ': ')
            except EOFError:
                print()  # newline
                break

            match opt:
                case 'help':
                    print(
                        "type 'help' for this message\n"
                        "type 'quit' to exit the program'\n"
                        "type 'parse' to input a sentence to parse"
                    )

                case 'parse':
                    try:
                        sentence = input(self.prompt + ' [sentence]: ')
                    except EOFError:
                        print()  # newline
                        continue

                    parses = self.grammar.parse(sentence.split())

                    if not parses:
                        print('NO VLAID PARSES')
                        continue

                    parse_cnt = 0
                    print('VALID SENTENCE')
                    for parse in parses:
                        parse_cnt += 1
                        print(f'parse: {parse_cnt}')
                        print(parse_str(parse))

                    print(f'number of valid parses: {len(parses)}')

                case 'quit' | 'q':
                    break

                case _:
                    print(f"error: unknown command '{opt}'")

        print('goodbye!')

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
