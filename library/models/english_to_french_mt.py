from typing import Dict, Optional

import torch
from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from overrides import overrides

from ..modules.encoder_decoders.sequence_to_sequence import SequenceToSequence

# TODO: Make this more generalized. When generalized, others can subclass this without having
# to write the from_param's function.

@Model.register("english_to_french_seq2seq")
class EnglishToFrenchEncoderSeq2Seq(SequenceToSequence):
    """
    Replication of Sutskever et al.'s ``Sequence-to-Sequence`` model for the English to French task.

    English is encoded via a deep LSTM ``backwards``. The resulting hidden state then primes
    generation of the equivalent French utterance using another deep LSTM.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 en_field_embedder: TextFieldEmbedder,
                 fr_embedder: int,
                 en_encoder: Seq2SeqEncoder,
                 fr_decoder_type: str,
                 fr_decoder_num_layers,
                 output_projection_layer: FeedForward,
                 apply_attention: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, "en", "fr", en_field_embedder, fr_embedder,
                         en_encoder, fr_decoder_type, output_projection_layer,
                         apply_attention=apply_attention,
                         decoder_num_layers=fr_decoder_num_layers)

    @overrides
    def preprocess_input(self, source: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        """
        Reverse the English utterance before encoding it.
        """
        source_max_len = source['tokens'].size()[-1]
        source_revered_indices = torch.linspace(source_max_len - 1, 0, source_max_len).long()
        source_revered_indices = source_revered_indices.to(source['tokens'].device)  # CPU/GPU invariant.
        source_reversed_tokens = source['tokens'].index_select(-1, source_revered_indices)
        assert source['tokens'].equal(source_reversed_tokens.index_select(-1, source_revered_indices))
        source['tokens'] = source_reversed_tokens
        return source

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'EnglishToFrenchEncoderSeq2Seq':
        en_field_embedder = TextFieldEmbedder.from_params(vocab, params.pop("en_field_embedder"))
        en_encoder = Seq2SeqEncoder.from_params(params.pop("en_encoder"))
        fr_embedder = Embedding.from_params(vocab, params.pop("fr_embedder"))

        # Unwrap the decoder parameters. They need to be fed to the model since there's no
        # Seq2Seq that also returns cell states.
        fr_decoder_params = params.pop("fr_decoder").params
        fr_decoder_type = fr_decoder_params['type']
        fr_decoder_num_layers = fr_decoder_params['num_layers']
        apply_attention = params.pop("apply_attention", False)
        output_projection_layer = FeedForward.from_params(params.pop("output_projection_layer"))
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))
        return cls(vocab=vocab,
                   en_field_embedder=en_field_embedder,
                   fr_embedder=fr_embedder,
                   en_encoder=en_encoder,
                   fr_decoder_type=fr_decoder_type,
                   fr_decoder_num_layers=fr_decoder_num_layers,
                   output_projection_layer=output_projection_layer,
                   apply_attention=apply_attention,
                   initializer=initializer,
                   regularizer=regularizer)
