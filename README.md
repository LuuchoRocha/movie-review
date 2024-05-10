# movie-review

First of all you will need to train the model by running:

```
$ python main.py
```

Then, when the training is done, to use the classifier:

```python
>>> from inference import *
>>> classify("awesome movie")
'POSITIVE'
>>> classify("I did not like it")
'NEGATIVE'

```
