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

import tensorflow as tf

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
            print('Decoding discord chat dumps...', end='')
            text = [ ]
            for dump in args.files:
                text += discord.decode(dump)
            print('DONE')

            print('Generating training targets...', end='')
            chars = tf.strings.unicode_split(
                '\n'.join(text),
                input_encoding='UTF-8'
            )

            char2id = tf.keras.layers.StringLookup(
                vocabulary=sorted(set(''.join(text))),
                mask_token=None,
            )
            id2char = tf.keras.layers.StringLookup(
                vocabulary=char2id.get_vocabulary(),
                invert=True,
                mask_token=None,
            )

            ids         = char2id(chars)
            ids_dataset = tf.data.Dataset.from_tensor_slices(ids)
            print('DONE')


if __name__ == '__main__':
    main()
