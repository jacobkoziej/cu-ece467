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

        x, states = self.gru(x, initial_state=states, training=training)
        x         = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


class Step(tf.keras.Model):
    def __init__(self, model, char2id, id2char, temperature=1.0):
        super().__init__()

        self.model       = model
        self.char2id     = char2id
        self.id2char     = id2char
        self.temperature = temperature

        # add a mask to prevent '[UNK]' from generating
        skip_ids = self.char2id(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            values=[-float('inf')] * len(skip_ids),
            indices=skip_ids,
            dense_shape=[len(self.char2id.get_vocabulary())]
        )
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def gen_one_step(self, inputs, states=None):
        ids = self.char2id(tf.strings.unicode_split(inputs, 'UTF-8')).to_tensor()

        predicted_logits, states = self.model(
            inputs=ids,
            states=states,
            return_state=True,
        )

        # only use the last prediction and prevent '[UNK]'
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits / self.temperature
        predicted_logits = predicted_logits + self.prediction_mask

        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        predicted_chars = self.id2char(predicted_ids)

        return predicted_chars, states
