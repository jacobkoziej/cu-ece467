# cu-ece467

> The Cooper Union - ECE 467: Natural Language Processing

The following was completed under the supervision of Professor Carl
Sable for the Spring of 2022.  All distributed course material can be
found on the [course page].


## Projects

* Text Categorization `p1/`


## Runtime Dependencies

The following was developed and tested with Python 3.10 running on
Linux.

Though not required, it is recommended to use a Python virtual
environment to streamline the installation of project dependencies.

```sh
python3 -m venv cu-ece467
. cu-ece467/bin/activate
```

All third-party libraries can be installed using `pip`.

```sh
python3 -m pip install -r requirements.txt
```

After third-party libraries have been installed, the necessary NLTK data
must be downloaded.

```sh
python3 -m nltk.downloader punkt
python3 -m nltk.downloader stopwords
```


## Copyright & Licensing

Copyright (C) 2022  Jacob Koziej [`<jacobkoziej@gmail.com>`]

Distributed under the [GPLv3] or later.


[course page]: http://faculty.cooper.edu/sable2/courses/spring2022/ece467/
[`<jacobkoziej@gmail.com>`]: mailto:jacobkoziej@gmail.com
[GPLv3]: LICENSE.md
