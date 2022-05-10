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
