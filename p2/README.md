# p2

> Parsing: Cocke-Kasami-Younger (CKY)


## Usage

Regular Usage:

```
./p2.py grammar.cnf
```

Enabling Parse Trees:

```
./p2.py -p grammar.cnf
```

Help:

```
./p2.py -h
```


### CLI

While the application is running, you are dropped into a CLI.  You can
print usage information by typing `help`, quit by typing `quit` or `q`,
or parse a sentence by typing `parse`.


## Grammar Format

Grammar is expected to be in Chomsky Normal Form (CNF).  Non-terminal
rules are to be in the form `A --> B C`, where `A`, `B`, and `C` are
non-terminals.  Terminal rules are to be in the form `A --> w`, where
`A` is a non-terminal and `w` is a terminal.
