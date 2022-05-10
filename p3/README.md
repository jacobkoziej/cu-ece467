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

All Discord chats were exported using [Tyrrrz]'s (Oleksii Holub
[DiscordChatExporter] in the JSON file format.  Since the messages used
for training and generation were from my own friend group, they,
unfortunately, cannot be shared.  If you wish to reproduce the results
of this project, feel free to do so using messages from your own servers
or other public servers.


[Tyrrrz]: https://tyrrrz.me/
[DiscordChatExporter]: https://github.com/Tyrrrz/DiscordChatExporter
