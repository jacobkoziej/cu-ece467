#!/usr/bin/env python3
# p2.py -- programming assignement #2
#          parsing
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

import argparse

import parser


def main():
    argparser = argparse.ArgumentParser(description='parsing')

    argparser.add_argument(
        'grammar',
        help='context free grammar in chomsky normal form',
        metavar='grammar.cnf',
    )
    argparser.add_argument(
        '-p',
        '--parse-tree',
        action='store_true',
        help='enable textual parse trees',
    )

    args = argparser.parse_args()

    cli = parser.Cli()

    cli.parse_grammar(args.grammar)

    cli.cli(args.parse_tree)


if __name__ == '__main__':
    main()
