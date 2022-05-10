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
import random
import string

import tensorflow as tf

import discord
import generator


def main():
    argparser = argparse.ArgumentParser(
        description='recurrent neural networks (text generation)',
    )
    subargparser = argparser.add_subparsers(
        dest='mode',
        help='{gen,train} -h/--help',
        metavar='mode',
        required=True,
    )

    gen_subargparser = subargparser.add_parser('gen')
    gen_subargparser.add_argument(
        '-c',
        '--character-count',
        default=64,
        dest='char_cnt',
        help='the number of character to generate',
        metavar='N',
        type=int,
    )
    gen_subargparser.add_argument(
        '-m',
        '--model-path',
        help='model path',
        metavar='path',
        required=True,
    )
    gen_subargparser.add_argument(
        '-p',
        '--prefix',
        help='autoregressive generation prefix',
        metavar='word(s)',
    )

    train_subargparser = subargparser.add_parser('train')
    train_subargparser.add_argument(
        'files',
        help='discord chat dump',
        metavar='dump.json',
        nargs='+',
    )
    train_subargparser.add_argument(
        '-b',
        '--batch-size',
        default=256,
        dest='batch_size',
        help='training batch size',
        metavar='N',
        type=int,
    )
    train_subargparser.add_argument(
        '-e',
        '--embed-dim',
        default=1024,
        dest='embed_dim',
        help='model embedding dimension',
        metavar='N',
        type=int,
    )
    train_subargparser.add_argument(
        '-o',
        '--output',
        default='a.out',
        help='model output name',
        metavar='name',
    )
    train_subargparser.add_argument(
        '-u',
        '--buf-size',
        default=2 ** 14,
        dest='buf_size',
        help='training buffer size',
        metavar='N',
        type=int,
    )
    train_subargparser.add_argument(
        '-s',
        '--seq-len',
        default=256,
        dest='seq_len',
        help='training sequence length',
        metavar='N',
        type=int,
    )
    train_subargparser.add_argument(
        '-r',
        '--rnn-units',
        default=1024,
        dest='rnn_units',
        help='number of model RNN units',
        metavar='N',
        type=int,
    )
    train_subargparser.add_argument(
        '-t',
        '--epochs',
        default=256,
        help='number of training epochs',
        metavar='N',
        type=int,
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
            text = '\n'.join(text)

            chars = tf.strings.unicode_split(
                text,
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

            args.seq_len = abs(args.seq_len)
            sequences = ids_dataset.batch(
                args.seq_len + 1,
                drop_remainder=True,
            )
            print('DONE')

            print('Generating training batches...', end='')
            args.batch_size = abs(args.batch_size)
            args.buf_size   = abs(args.buf_size)

            def split_input_target(sequence):
                return sequence[:-1], sequence[1:]

            dataset = sequences.map(split_input_target)
            dataset = (
                dataset
                .shuffle(args.buf_size)
                .batch(args.batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE)
            )
            print('DONE')

            print('TRAINING MODEL')
            model = generator.Model(
                vocab_size=len(char2id.get_vocabulary()),
                embedding_dim=abs(args.embed_dim),
                rnn_units=abs(args.rnn_units),
            )

            loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
            model.compile(optimizer='adam', loss=loss)


            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=f'./{args.output}_checkpoints/' + 'ckpt_{epoch}',
                save_weights_only=True,
            )

            history = model.fit(dataset, epochs=abs(args.epochs), callbacks=[checkpoint_callback])
            model.summary()

            one_step = generator.Step(model, char2id, id2char)

            # if we do not call generator.Step.gen_one_step() TensorFlow
            # will skip the serialization of our model, this is just a
            # dirty workaround for tf.saved_model.save() failing on us
            states = None
            next_char = tf.constant(['FORCE BUILD'])
            result = [next_char]
            for n in range(64):
                 next_char, states = one_step.gen_one_step(next_char, states=states)
                 result.append(next_char)

            tf.saved_model.save(one_step, args.output)

        case 'gen':
            one_step = tf.saved_model.load(args.model_path)

            if args.prefix == None:
                args.prefix = random.choice(string.ascii_lowercase)

            states = None
            next_char = tf.constant([args.prefix])
            result = [next_char]

            for n in range(abs(args.char_cnt)):
                 next_char, states = one_step.gen_one_step(next_char, states=states)
                 result.append(next_char)

            print(tf.strings.join(result)[0].numpy().decode('utf-8'))


if __name__ == '__main__':
    main()
