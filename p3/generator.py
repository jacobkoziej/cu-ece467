# generator.py -- RNN text generator
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

import tensorflow as tf


class Model(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            rnn_units,
            return_sequences=True,
            return_state=True,
        )
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)

        if states is None:
            states = self.gru.get_initial_state(x)

        x, states = self.gru(x, inital_state=states, training=training)
        x         = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


class Preprocess:
    def gen_char2id(self, vocab: list[str]) -> tf.keras.layers.StringLookup:
        return tf.keras.layers.StringLookup(vocabulary=vocab)

    def gen_id2char(self, vocab: list[str]) -> tf.keras.layers.StringLookup:
        return tf.keras.layers.StringLookup(invert=True, vocabulary=vocab)

    def gen_vec(self, input: list[str]) -> tf.RaggedTensor:
        return tf.strings.unicode_split(input, input_encoding='UTF-8')

    def gen_vocab(self, input: str) -> list[str]:
        return sorted(set(input))

    def gen_input_target_seq(self, input: list) -> tuple[list[str], list[str]]:
        input_seq  = input[:-1]
        target_seq = input[1:]

        return input_seq, target_seq
