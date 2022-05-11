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


### Architecture Experimentation

Training a language model on informal messages (especially for an
all-purpose group chat) brings with it a set of unique challenges not
usually found while training with a corpus of a distinct style.

To give some context to the erratic corpus used for training, here are
some quick statistics:
* 162,310 total messages
* 3,600,253 total characters
* 1,099 unique characters
* message lengths between 1 to 2,000 characters
* message content:
	* topics that range from school to personal life
	* emojis (including custom emojis in the form `:emoji:`)
	* mentions (in the form `@username`)
	* in-line links, LaTeX, & code blocks
	* copypastas & spam walls
	* keyboard mashes

As evident, such a corpus shares no distinct "style" beyond it being an
organized collection of chaos.  Regardless, there are ways around this
by using character embeddings.  Although word embeddings and subword
embeddings have proven to be very successful, character embeddings were
chosen simply for the property of being capable of circumventing minor
spelling mistakes.  Character embeddings also remove any distinction
between what qualifies as a word or subword rather, a sequence is
understood for what it is: characters assigned a meaning.  As such, the
input layer of the model consisted of a character embedding layer that
had the dimension of the input vocabulary and an additional token for
unrecognized characters.

The internal hidden layer consisted of a GRU.  The reasoning for
choosing this type of layer was to turn the language model into an RNN
and to increase efficiency compared to an LSTM.  An increase in
efficiency was a desirable trait as my primary work machine is rather
underpowered, and access to the school compute server was in high demand
at this time of the semester.

Lastly, the output layer consisted of a dense layer to map the GRU
hidden layer to an output of the same dimension as the input character
embedding layer.


#### Tuning

The architecture as described above leaves four parameters to be tuned
to achieve "optimal" accuracy, those being: GRU count, embedding
dimensions, batch size, and sequence length.  To simplify tuning the
parameters were tied together into pairs (ie. GRU count = embedding
dimensions and batch size = sequence length).  While training,
cross-entropy was used to calculate the loss between epochs.  If a
model's loss increased before 128 epochs, it was discarded.

Initial values for the parameters started at 256 for the embedding
dimension and 64 for the sequence length.  All parameters were kept as
powers of two so that they could easily be doubled or halved.

The embedding dimension was initially increased to 1024 as going any
higher would increase the loss after about 20 epochs.  This value does
appear to make sense as it is close to the training vocabulary size.
Since the vocabulary size is quite large, having this increased the size
most likely decreased collisions between characters during training.

Next, the sequence length increased up to 512 characters.  Although
smaller sequence sizes did see initial loss drop off quite fast (going
below 1.5 within eight epochs), loss did pick up after about 30 epochs.
At 256 the sequence length appeared to hit an asymptote at 1.0 loss as
the full 128 epochs did not see any significant change in loss for over
60 epochs.  Ultimately 512 characters did break through this asymptote,
but it did come at the cost of requiring far more epochs (over 80), but
at 128 epochs it did manage to reach a loss of only 0.85.  Subsequently,
the epoch count was increased to 256, and the loss reached an asymptotic
loss at around 0.75.

Ultimately, I was happy with the loss achieved given such erratic input.
The generated output did resemble the messaging style of the individuals
who's messages did make it into the training data.


### Results

Going into this project, I had very high expectations for the quality of
the generated completions.  Ultimately, I feel let down by my lack of
background in a deep learning library such as TensorFlow, as this was my
first programming project in a while, where I felt out of control of
what I could create.  Atop of the craziness of the end of the semester
and there just not being a whole lot of time for experimentation, I had
to settle for just a few hours of tuning.

Besides all the pitfalls mentioned beforehand, I feel like my language
model performed rather well.  Not only are most generated messages
recognizable as someone's style of messaging, but messages for the most
part, consist of real worlds, even if the sentences don't have much
meaning.  But this usually happens after neumerous words, which wasn't
the intent of this project *anyways*.  In general, generated words
following an immediate input sequence seem to make sense.

Most surprising of all, training in sequences does seem to "just work."
Although fixed-length sequences were unintended, they did give way to
the predicted responses feature of my project.  I was actually surprised
at how accurate some of these responses could be given how people in the
group chat respond to one another.  Although some of these responses are
great, if they're anything more than a handful of words, they quickly
seem to lose meaning as expected.  Anyways, great *"bug"* turned
feature.


### Sample Output

Text completion:

```
input:      cooper union
completion: cooper union for the advancement of science and art was nothing worse all dimensions
```

```
input:      i am
completion: i am ready 🤔
```

```
input:      i am in
completion: i am in pain
```

```
input:      do y'all wanna go
completion: do y'all wanna go to office hours for me to check if they refuse
```

```
input:      can we go
completion: can we go back to cars for the funny
```

```
input:      should we
completion: should we call
```

```
input:      nlp is
completion: nlp is really rough linux or something
```

```
input:      linux is
completion: linux is so weird
```

Response prediction:

```
input:      guys i think i'm late
prediction: oop
```

```
input:      i am sad
prediction: nooo
```

```
input:      i am so happy today
prediction: YO
```

```
input:      i'm gonna buy some oreos
prediction: ooooh idk what to ur dorm again
```

```
input:      pain
prediction: DNJSNSJNSJSNS
```

```
input:      lol
prediction: i know right
```

Funny output:

```
input:      cope
completion: copecopecopecopecopecopecopecopecopecopecopecopecopecope
```

```
input:      i love cooper union
completion: i love cooper union?
```

```
input:      suffering makes you
completion: suffering makes you a better engineer. Suffering makes you a better engineer. It's called we do a little trolling
```

```
input:      can you save me a seat?
prediction: See pinned message in #blackmail
```


[Tyrrrz]: https://tyrrrz.me/
[DiscordChatExporter]: https://github.com/Tyrrrz/DiscordChatExporter
[TensorFlow]: https://www.tensorflow.org/
[tutorial on text generation with an RNN]: https://www.tensorflow.org/text/tutorials/text_generation
[permalink]: https://github.com/tensorflow/text/blob/16235e4ad31c572e0cbe40a5decb54fdedc6931e/docs/tutorials/text_generation.ipynb
