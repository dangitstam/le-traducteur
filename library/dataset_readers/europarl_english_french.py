import json
import logging
from typing import Dict

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter, SpacyWordSplitter
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("europarl_parallel_english_french")
class EuroparlEnglishFrenchReader(DatasetReader):
    """
    Reads a jsonl containing parallel English-French utterances from the
    proceedings of the European Parliament - http://www.statmt.org/europarl/

    Expected format of each line: {"id": int, "en": str, "fr": str}

    Fields not listed above will be ignored.

    Each ``read`` yields a data instance of
        en: ``TextField``
        fr: ``TextField``

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    en_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split English utterances into tokens.
        Defaults to ``WordTokenizer()``.
    fr_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split French utterances into tokens.
        Defaults to ``WordTokenizer(SpacyWordSplitter(language=;fr_core_news_sm)``.
    en_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define English token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer(namespace="en", lowercase_tokens=True)}``.
    fr_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define French token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer(namespace="fr", lowercase_tokens=True)}``.
    """
    def __init__(self,
                 lazy: bool = False,
                 en_tokenizer: Tokenizer = None,
                 fr_tokenizer: Tokenizer = None,
                 en_token_indexers: Dict[str, TokenIndexer] = None,
                 fr_token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._en_tokenizer = en_tokenizer or WordTokenizer()
        self._fr_tokenizer = fr_tokenizer or WordTokenizer(
            # Specify spaCy's French model instead (English is the default).
            word_splitter=SpacyWordSplitter(language='fr_core_news_sm')
        )
        self._en_token_indexers = en_token_indexers or {
            "tokens": SingleIdTokenIndexer(namespace="en", lowercase_tokens=True)
        }
        self._fr_token_indexers = fr_token_indexers or {
            "tokens": SingleIdTokenIndexer(namespace="fr", lowercase_tokens=True)
        }

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), 'r') as data_file:
            logger.info("Reading instances from lines in file: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                parallel_utterance = json.loads(line)
                en_utterance = parallel_utterance['en']
                fr_utterance = parallel_utterance['fr']
                yield self.text_to_instance(en_utterance, fr_utterance)

    @overrides
    def text_to_instance(self, en_utterance: str, fr_utterance: str) -> Instance:  # type: ignore
        en_utterance_tokenized = self._en_tokenizer.tokenize(en_utterance)
        fr_utterance_tokenized = self._fr_tokenizer.tokenize(fr_utterance)
        fields = {
            'en': TextField(en_utterance_tokenized, self._en_token_indexers),
            'fr': TextField(fr_utterance_tokenized, self._fr_token_indexers)
        }
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'EuroparlDatasetReader':
        lazy = params.pop('lazy', False)
        en_tokenizer = Tokenizer.from_params(params.pop('en_tokenizer', {}))
        fr_tokenizer = Tokenizer.from_params(params.pop('fr_tokenizer', {}))
        en_token_indexers = TokenIndexer.dict_from_params(params.pop('en_token_indexers', {}))
        fr_token_indexers = TokenIndexer.dict_from_params(params.pop('fr_token_indexers', {}))
        params.assert_empty(cls.__name__)
        return cls(lazy=lazy, en_tokenizer=en_tokenizer, en_token_indexers=en_token_indexers,
                              fr_tokenizer=fr_tokenizer, fr_token_indexers=fr_token_indexers)

@DatasetReader.register("europarl_parallel_english_french_pretokenized")
class EuroparlEnglishFrenchReaderPretokenized(DatasetReader):
    """
    Identical to ``EuroparlEnglishFrenchReader`` but assumes its input is already tokenized.

    Each ``read`` yields a data instance of
        en: ``TextField``
        fr: ``TextField``

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    en_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define English token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer(namespace="en", lowercase_tokens=True)}``.
    fr_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define French token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer(namespace="fr", lowercase_tokens=True)}``.
    """
    def __init__(self,
                 lazy: bool = False,
                 en_token_indexers: Dict[str, TokenIndexer] = None,
                 fr_token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._en_tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())
        self._fr_tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())
        self._en_token_indexers = en_token_indexers or {
            "tokens": SingleIdTokenIndexer(namespace="en", lowercase_tokens=True)
        }
        self._fr_token_indexers = fr_token_indexers or {
            "tokens": SingleIdTokenIndexer(namespace="fr", lowercase_tokens=True)
        }

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), 'r') as data_file:
            logger.info("Reading instances from lines in file: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                parallel_utterance = json.loads(line)
                en_utterance = parallel_utterance['en']
                fr_utterance = parallel_utterance['fr']
                yield self.text_to_instance(en_utterance, fr_utterance)

    @overrides
    def text_to_instance(self, en_utterance: str, fr_utterance: str) -> Instance:  # type: ignore
        en_utterance_tokenized = self._en_tokenizer.tokenize(en_utterance)
        fr_utterance_tokenized = self._fr_tokenizer.tokenize(fr_utterance)
        fields = {
            'en': TextField(en_utterance_tokenized, self._en_token_indexers),
            'fr': TextField(fr_utterance_tokenized, self._fr_token_indexers)
        }
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'EuroparlDatasetReader':
        lazy = params.pop('lazy', False)
        en_token_indexers = TokenIndexer.dict_from_params(params.pop('en_token_indexers', {}))
        fr_token_indexers = TokenIndexer.dict_from_params(params.pop('fr_token_indexers', {}))
        params.assert_empty(cls.__name__)
        return cls(lazy=lazy, 
                   en_token_indexers=en_token_indexers,
                   fr_token_indexers=fr_token_indexers)
