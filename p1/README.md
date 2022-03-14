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
