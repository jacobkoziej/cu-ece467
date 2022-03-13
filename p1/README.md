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
