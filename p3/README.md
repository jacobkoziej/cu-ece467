# p3

> RNNs: text generation, completion, and predictions

Recurrent neural networks, or RNNs for short, are a class of neural
networks (NNs) concerned with connecting nodes of a NN along a temporal
sequence.  Each time step contains information about previous and
current inputs and, in concept, should function much how the mind
presumably consumes information.  Although great in concept, RNNs suffer
from the vanishing gradient problem, limiting RNNs to a handful of
timesteps in the recurrent hidden layer.  In an effort to mitigate the
issue, Long Short-Term Memory units (LSTMs) and, more recently, Gated
Recurrent Units (GRUs) have come into fruition.  Although these networks
have seen wide success in the field of Natural Language Processing
(NLP), they have recently grown out of fashion in favor of Transformers,
which not only perform better but are more versatile.  Regardless of
performance, the following project utilizes RNNs to a varying degree of
success.


## Usage

Help:

```
./p3.py -h
```

Training:

```
./p3.py train dump.json [dump.json ...]
```

Generation:

```
./p3.py gen -m model_path
```

To view additional options for each mode run `./p3.py [mode] -h`.


## Dump Format

All Discord chats were exported using [Tyrrrz]'s (Oleksii Holub)
[DiscordChatExporter] in the JSON file format.  Since the messages used
for training and generation were from my own friend group, they,
unfortunately, cannot be shared.  If you wish to reproduce the results
of this project, feel free to do so using messages from your own servers
or other public servers.


## Implementation Details

The following project was implemented using [TensorFlow] following the
official [tutorial on text generation with an RNN] ([permalink]).


### Inspiration

The inspiration for this project came from text suggestions that appear
on modern software keyboards (such as Gboard) and shell completions
(such as those found in Zsh).  Being someone who's rather
privacy-conscious and unhappy about data harvesting, I opt to disable
personalized completions in my software keyboards which comes at the
expense of slower typing.  Although I knew I would never come close to
the accuracy of modern keyboards, I was still interested in how well a
"DIY" message completion system would perform, given my messages with
friends.


### Training & Testing

Ideally, while training an RNN for text completion, each individual
message would be fed into the RNN.  Unfortunately, given the time
constraints of the project, and a lack of experience with TensorFlow,
training was instead done using batches of characters (though this
constraint ultimately grew into the "predicted response" feature of the
language model).  In doing so, sequences would be of the same length and
not require padding.  This decision came with the trade-off of having no
start-of-sentence (SOS) or end-of-sentence (EOS) tokens.  As a
compromise, any character could begin a sequence, and the newline
character is treated as the EOS token.  The reasoning behind this was
that the character was placed between each message while concatenating
all the messages into one contiguous string.

Since the generated text originated from personal messages between
friends, there was no "true" empirical method to evaluate the validity
of the generated sequences.  As will be discussed later, training
messages varied wildly in almost all aspects (length, content, style,
etc.).  Instead of relying on an empirical system, human judgment became
a substitute.  Although human judgment cannot assign a value to the
accuracy of the generated text, it does fall in line with Steven
Pinker's beliefs that grammatical rules of a language are themselves as
valid as we make them out to be.  Ultimately training was tuned until
sequences turned from unrecognizable gibberish (random characters) to
sequences of recognizable gibberish (ie. strings of words that resembled
the style of someone in the group chat).


[Tyrrrz]: https://tyrrrz.me/
[DiscordChatExporter]: https://github.com/Tyrrrz/DiscordChatExporter
[TensorFlow]: https://www.tensorflow.org/
[tutorial on text generation with an RNN]: https://www.tensorflow.org/text/tutorials/text_generation
[permalink]: https://github.com/tensorflow/text/blob/16235e4ad31c572e0cbe40a5decb54fdedc6931e/docs/tutorials/text_generation.ipynb
