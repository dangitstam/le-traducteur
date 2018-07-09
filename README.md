# Le Traducteur

A framework for neural machine translation. The name is in reference to the framework's first task of English-to-French translation.

### Supported Models:
* Sutskever et al.'s ["Sequence to Sequence with Neural Networks"](https://arxiv.org/abs/1409.3215)

### Recommended Corpora
* [Europarl Parallel Corpora](http://www.statmt.org/europarl/): Proceedings of the European Parliament from 1996 to 2011

## Getting Started

The system is built with PyTorch and AllenNLP, which are the main dependencies.

### Prerequisites

* Python 3.6 (3.6.5+ recommended)

### Installing

It is recommended to first create a **virtual environment** before installing dependencies.

#### Using Conda
`conda create --name le-traducteur python=3.6`

#### Using VirtualEnv
`python3 -m venv /path/to/new/virtual/environment`

Download PyTorch and AllenNLP via

```
`pip install -r requirements.txt`
```

### Caveat with installing AllenNLP using pip / conda
The current version of AllenNLP doesn't support restricting vocabulary by namespace. To enable this and run the
provided experiments, you'll have to download AllenNLP from [source](https://github.com/allenai/allennlp).

Once version 0.5.2 is released, this should no longer be a problem.

### Pre-trained NLP models
Several tokenizers used rely on [NLTK](https://www.nltk.org/) and [spaCy](https://spacy.io/)'s pre-trained models for tokenizing English as well as French and Spanish. Feel free to not explicit download these models yourself. **They will be downloaded automatically if a tokenizer in the config is specified to use a spaCy model that does not yet exist on your machine**.

## Dataset Reading

### Running initial tests
Go to the root directory of this repository and run `pytest` to verify the provided dataset readers are working.

### Creating a corpus
`scripts/generate_parallel_europarl.py` is a provided tool to create combined parallel Europarl corpus for any language pair are provided. It is recommend to refer to languages via their [ISO codes](https://en.wikipedia.org/wiki/ISO_3166-1) when using this script and the framework in general.

This script can be used for any parallel corpus. It only makes the assumption that the files it is given are:
* The same number of lines
* Where a line in one file is a translation of the other file at the same line

Arguments to this script are:
* src language: The ISO code of the source language in which to translate from
* dst language: The ISO code of the destination language in which to translate to
* src path: The path to the source language utterances
* dst path: The path to the destination language utterances
* save dir: The directory in which to save the new corpus

Output:
A [jsonl](http://jsonlines.org/) file containing a single JSON object per line of the form
```
{
    'id': <Line number of the >
    <src language>: <The src language utterance>
    <dst language>: <The dst language utterance>
}
```

An example dataset reader meant for reading the Europarl French-to-English dataset is provided in [europarl_french_english.py](library/dataset_readers/europarl_english_french.py).

## Experiments

Example parallel corpora and configurations are provided in `experiments/` and `tests/fixtures`. 

Experiments are run by doing
```
allennlp train <path to the current experiment's JSON configuration> \
-s <directory for serialization>  \
--include-package library
```

A recommended workflow for extending beyond the provided models and supported language pairs is provided in [conventions.md](CONVENTIONS.md).

## Built With

* [AllenNLP](https://allennlp.org/) - The NLP framework used, built by AI2
* [PyTorch](https://pytorch.org/) - The deep learning library used

## Authors

* **Tam Dang**

## License

This project is licensed under the Apache License - see the [LICENSE.md](LICENSE) file for details.
