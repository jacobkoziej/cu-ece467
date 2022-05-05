#!/usr/bin/env python3
# p3.py -- programming assignment #3
#          recurrent neural networks (text generation)
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

import discord
import generator


def main():
    argparser = argparse.ArgumentParser(
        description='recurrent neural networks (text generation)',
    )
    subargparser = argparser.add_subparsers(
        dest='mode',
        help='train -h/--help',
        metavar='mode',
        required=True,
    )

    train_subargparser = subargparser.add_parser('train')
    train_subargparser.add_argument(
        'files',
        help='discord chat dump',
        metavar='dump.json',
        nargs='+',
    )

    args = argparser.parse_args()

    match args.mode:
        case 'train':
            text = [ ]
            for dump in args.files:
                text += discord.decode(dump)


if __name__ == '__main__':
    main()
