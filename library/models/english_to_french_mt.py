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
                 fr_field_embedding_size: int,
                 en_encoder: Seq2SeqEncoder,
                 fr_decoder_type: str,
                 fr_decoder_num_layers,
                 output_projection_layer: FeedForward,
                 apply_attention: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, "en", "fr", en_field_embedder, fr_field_embedding_size,
                         en_encoder, fr_decoder_type, output_projection_layer,
                         apply_attention=apply_attention,
                         decoder_num_layers=fr_decoder_num_layers)

    @overrides
    def preprocess_input(self, source: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        """
        Reverse the English utterance before encoding it.
        """
        device = source['tokens'].device  # For moving tensors to the GPU.
        source_max_len = source['tokens'].size()[-1]
        source_revered_indices = torch.linspace(source_max_len - 1, 0, source_max_len).long()
        source_revered_indices = source_revered_indices.to(source['tokens'].device)  # CPU/GPU invariant.
        source_reversed_tokens = source['tokens'].index_select(-1, source_revered_indices)
        assert source['tokens'].equal(source_reversed_tokens.index_select(-1, source_revered_indices))

        # Padding has been shoved to the front as a result of reversing.
        # We have to move it to the back.
        padding_length = source["tokens"].size(-1)
        for i, example in enumerate(source_reversed_tokens):
            example = example[example.nonzero()].squeeze()

            # Some utterances are as short as one word. In this case, squeezing
            # results in a zero-dimeisional tensor.
            if example.dim() == 0:
                padding = torch.LongTensor([0] * (padding_length - 1)).to(device=device)
                example = torch.cat((example.unsqueeze(0), padding), 0)
            else:
                padding = torch.LongTensor([0] * (padding_length - len(example))).to(device=device)
                example = torch.cat((example, padding), 0)

            source_reversed_tokens[i] = example

        source['tokens'] = source_reversed_tokens
        return source
