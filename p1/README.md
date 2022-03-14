# p1

> Text Categorization: Rocchio/TF*IDF


## Usage

Help:

```
./p1.py -h
```

Training:

```
./p1.py train -i input_labels -d output_database
```

Testing:

```
./p1.py test -d trained_database -i input_list -o output_labels
```

Test Generation:

```
./p1.py testgen -i labeled_corpus -o output_prefix
```

Optionally, `-v` or `--verbose` can be passed in training or testing
mode to enable output information.


## Text Categorization Approach

This project implements the Rocchio/TF*IDF text categorization (TC)
approach.  This TC approach implements a vector space model where
"document" vectors, consisting of weighted word features, are compared
to determine the similarity between the two documents.

Although it is often easy to quantify what constitutes a document, a
vector need not be the traditional definition of a document; rather, it
can take on the form of either a category or query.  Word weight is
handled by multiplying a term's frequency (TF) by its inverse document
frequency (IDF).  Doing so allows a TC system to easily tag uncommon
words (defining features) with a higher weight while assigning a lower
weight to common words.  Ultimately the TC system is left with vectors
that can be compared using a cosine similarity metric (the normalized
dot product of two document vectors).


## Implementation Details

The following TC system can be divided into three logical structures: a
collection that contains IDF values for a training set, category vectors
that contain TF values from the same training set, and document vectors
that contain TF values from a testing set.

During training, a database is created with certain text normalization
features enabled or disabled.  These options are then used for all
training and testing purposes and should not change without regenerating
the database.

During text normalization, each input document is converted into tokens
for processing.  Initially, only a Punkt sentence tokenizer was utilized
for normalization, but several optimizations, were later added; these
are discussed later in the document.

When normalization is complete, each document is added to the training
collection, causing the document count to increment, along with the
document frequencies for each unique token.  In addition to this, tokens
tagged as a certain category are appended to an array consisting of
category tokens.  Once all documents are added to the collection, the
IDFs for all the tokens in the collection are calculated.  Next,
category tokens are converted into category vectors and stored in the
database. Finally, the norms of the category vectors are precomputed to
speed up similarity calculations during testing.  Once all training is
complete, the database is exported to a file.

When testing, the trained database is loaded into memory.  All documents
are then normalized with the same settings used while training.  Each
normalized document is then converted into a vector, computing the norm
along the way to speed up similarity calculations.

When predicting a document's category, the normalized dot product is
taken between it and a category.  The result with the highest value is
then deemed to be the predicted category.


### Weighting Scheme

As touched upon earlier in the document, two weighting metrics were
utilized when calculating word weights.

At first, TF was used as a raw value, but this turned out to be a bit of
a drawback when using category vectors as certain features could occur a
magnitude more times than other features.  The issue with this approach
is that a feature that occurs a magnitude more often than another is not
necessarily more important.  To get around this drawback, a logarithm
was applied to the raw TF.  One issue with this approach is that a
feature that has a raw TF of one will have a weighted TF of zero.  To
prevent this, the raw TF was always incremented by one before taking the
logarithm.

Similar to the TF, the IDF frequency was also squashed using a logarithm
to prevent large values, skewing results.  When calculating IDF, the
document count of a collection is divided by the document frequency,
which is then input into a logarithmic function.  In doing so, a term
that occurs in every document will result in a weight of zero as it is
not a defining trait.

Finally, both the TF and IDF are multiplied together to arrive at a word
weight for each feature of a document.


### Optimizations

Although the initial performance was decent, several optimizations were
applied to increase the performance of the TC system during text
normalization.

Firstly, all input was made lowercase.  This made the TC system case
insensitive, giving marginal improvements in performance as
capitalization no longer affected features.

Another small improvement came in the form of filtering stop words.
Stop words are typically the most common words in a language and can be
safely discarded (for the most part) before processing.  Though this did
result in a marginal increase in performance, TF*IDF weighting already
does a good job at taking care of common words by assigning them a low
weight.

Finally, a Porter stemmer was applied to all document tokens.  Stemming
removes morphological and inflexional endings from words.  Since the
current TC system is based on a bag-of-words approach, stemming allowed
for words to be mapped to the same stem, generalizing the features of a
document or category, improving accuracy far more than the previous two
optimizations.

Interestingly enough, these optimizations *decreased* performance in
corpus two, which consisted of image captions tagged as either indoor or
outdoor.  Since these optimizations were concerned with generalizing
input features, a defining trait of an indoor or outdoor location that
was identifiable through capitalization, common words, or the
morphological or inflexional endings of a feature were removed.
Ultimately the gains seen in both corpus one and three were greater than
the loss seen in corpus two, and these optimizations were all kept.


## Evaluating Performance

The F1 score was utilized to evaluate the effectiveness of the TC
system.  This metric combines both precision and recall into a single
value.

Although three corpora were provided by the instructor, only one corpus
consisted of both a training and test set.  To get around this, a
training and test set was generated from the provided training corpora.
Documents were randomly shuffled and divided into a training and testing
set in a ratio of 2:1, roughly the same ratio as that of the provided
corpus, which had both a training and test set.
