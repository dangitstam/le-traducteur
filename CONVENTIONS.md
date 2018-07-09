# Project Standards and Practices

Nothing in here is obligatory to get the project running, but it will make navigating the code
easier and gives a foundation with which to built the framework with consistency.

## Python
* PEP8

## Naming Conventions

`<source>` and `<dest>` are always in alphabetical order unless they're part of a model
name that implies direction.

### Variables
* Dictionary references for a language are its abbreviation (ISO code) (ex. `'en'`, `'fr'`)
* Variables and objects pertaining to a language are prefixed with the language abbreviation
* Registered model names are `<source>_to_<dest>_method.py` (ex. `english_to_french_seq2seq.py`)

In short, internals are abbreviated, externals are not.

**It saves a lot of headache if dictionary references, data instance field names, and parameter's for each model's `forward()` function all use the ISO codes since instance field names and forward parameters have to match.**

More on this below.

### Models: `libarary/models`
Files containing models translating source to dest are named `<source>_to_<dest>_translator.py`
Provided models will always assume language inputs are passed named with their ISO codes. Note that **these have to match the instance field names** that are provided by the dataset reader. 

For example, a model `forward()` for English-French like this
```
    def forward(self,
                en: Dict[str, torch.LongTensor],
                fr: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
```

implies that the dataset reader returns
   ```
   fields = {
       id: int
       "en": str
       "fr": str
   }
   ```
in its implementation of `text_to_instance()`.

### Datasets
Data files are named `<experiment>_<corpus>_<language1 ISO>_<language2 ISO>.jsonl`

Where each entry consists of
```
{
    id: int
    "language1's ISO": str
    "language2's ISO": str
}
```

These can produced by following the [README][README.md] instructions for building a parallel corpus by passing the ISO codes of the each monolingual transcription's language as the source and destination languages.

### Dataset Readers: `libarary/dataset_readers`
Dataset readers are named `<corpus>_<language1>_<language2>.py`. Dataset readers will always assume
languages are accessed via their ISO codes. Fields should be created as
```
fields = {
    id: int
    "language1's ISO": str
    "language2's ISO": str
}

instance = Instance(fields)
```
to be consistent with what provided models are expecting.

Separating language dataset readers into their own files instead of sharing a base class allows defining
different language-specific defaults and customizations. It also allows tokenizers, indexers, and instances
to follow the variable naming conventions.

### Experiments: `experiements`
* smoke - Get it to run without crashing
* steam - Get it to run with non-trivial datasets / parameters

### Translators: `library/translators`
