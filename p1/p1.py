#!/usr/bin/env python3
# p1.py -- programming assignement #1
#          text categorization
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


def arg_gen():
    parser = argparse.ArgumentParser(description='text categorization')
    subparser = parser.add_subparsers(
        help='{test,train} -h/--help',
        metavar='mode',
        required=True,
    )

    test_subparser = subparser.add_parser('test')
    test_subparser.add_argument(
        '-d',
        help='input trained database',
        metavar='db',
        required=True,
        type=argparse.FileType('r'),
    )
    test_subparser.add_argument(
        '-i',
        help='input test documents',
        metavar='input',
        required=True,
        type=argparse.FileType('r'),
    )
    test_subparser.add_argument(
        '-o',
        help='output labeled documents',
        metavar='output',
        required=True,
        type=argparse.FileType('w'),
    )

    train_subparser = subparser.add_parser('train')
    train_subparser.add_argument(
        '-i',
        help='input training documents',
        metavar='input',
        required=True,
        type=argparse.FileType('r'),
    )
    train_subparser.add_argument(
        '-o',
        help='output trained database',
        metavar='db',
        required=True,
        type=argparse.FileType('w'),
    )

    return parser


def main():
    args = arg_gen()
    args.parse_args()


if __name__ == '__main__':
    main()