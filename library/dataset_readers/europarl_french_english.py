from typing import Dict
import json
import logging

from overrides import overrides

import tqdm

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("europarl_combined_french_english")
class EuroparlDatasetReader(DatasetReader):
    """
    Reads a jsonl containing parallel English-French utterances from the
    proceedings of the European Parliament - http://www.statmt.org/europarl/

    Expected format of each line: {"id": int, "en": str, "fr": str}

    Fields not listed above will be ignored.


    """
    def __init__(self,
                 lazy: bool = False,
                 en_tokenizer : Tokenizer = None,
                 fr_tokenizer: Tokenizer = None,
                 en_token_indexers : Dict[str, TokenIndexer] = None,
                 fr_token_indexers : Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._en_tokenizer = en_tokenizer or WordTokenizer()
        self._en_token_indexers = en_token_indexers or {"en_tokens": SingleIdTokenIndexer()}

        # To tokenize French, will have to opt for spaCy's pre-trained French
        # model. SpaCy's English parser is the default for WordTokenizer.
        fr_splitter = SpacyWordSplitter(language='fr_core_news_sm')
        self._fr_tokenizer = fr_tokenizer or WordTokenizer(word_splitter=fr_splitter)
        self._fr_token_indexers = fr_token_indexers or {"fr_tokens": SingleIdTokenIndexer()}

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
    def text_to_instance(self, en_utterance : str, fr_utterance : str) -> Instance: # type: ignore
        en_utterance_tokenized = self._en_tokenizer.tokenize(en_utterance)
        fr_utterance_tokenized = self._fr_tokenizer.tokenize(fr_utterance)
        en_utterance_field = TextField(en_utterance_tokenized, self._en_token_indexers)
        fr_utterance_field = TextField(fr_utterance_tokenized, self._fr_token_indexers)
        fields = {
            'en': en_utterance_field,
            'fr': fr_utterance_field
        }
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'EuroparlDatasetReader':
        lazy = params.pop('lazy', False)
        en_tokenizer = Tokenizer.from_params(params.pop('en_tokenizer', {}))
        en_token_indexers = TokenIndexer.dict_from_params(params.pop('en_token_indexers', {}))
        fr_tokenizer = Tokenizer.from_params(params.pop('fr_tokenizer', {}))
        fr_token_indexers = TokenIndexer.dict_from_params(params.pop('fr_token_indexers', {}))
        params.assert_empty(cls.__name__)
        return cls(lazy=lazy,
                   en_tokenizer=en_tokenizer, en_token_indexers=en_token_indexers,
                   fr_tokenizer=fr_tokenizer, fr_token_indexers=fr_token_indexers)
