from typing import Dict, Optional

import torch
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import (FeedForward, Seq2SeqEncoder, Seq2VecEncoder,
                              TextFieldEmbedder)
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from overrides import overrides


@Model.register("english_to_french_seq2seq")
class EnglishToFrenchEncoderSeq2Seq(Model):
    """
    Replication of Sutskever et al.'s ``Sequence-to-Sequence`` model for the English to French task.

    English is encoded via a deep LSTM ``backwards``. The resulting hidden state then primes
    generation of the equivalent French utterance using another deep LSTM.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 en_field_embedder: TextFieldEmbedder,
                 fr_field_embedder: TextFieldEmbedder,
                 # Only the last hidden state is needed from english.
                 en_encoder: Seq2VecEncoder,
                 # But each French word will have to be encoded and pushed through a decoder.
                 fr_encoder: Seq2SeqEncoder,
                 fr_decoder: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        if en_encoder.get_input_dim() != en_field_embedder.get_output_dim():
            raise ConfigurationError("The input dimension of the en_encoder must match the "
                                     "embedding size of the en_field_embedder. Found {} and {}, "
                                     "respectively.".format(en_encoder.get_input_dim(),
                                                            en_field_embedder.get_output_dim()))
        if fr_encoder.get_input_dim() != fr_field_embedder.get_output_dim():
            raise ConfigurationError("The input dimension of the fr_encoder must match the "
                                     "embedding size of the fr_field_embedder. Found {} and {}, "
                                     "respectively.".format(fr_encoder.get_input_dim(),
                                                            fr_field_embedder.get_output_dim()))
        if fr_decoder.get_input_dim() != fr_encoder.get_output_dim():
            raise ConfigurationError("The input dimension of the fr_decoder must match the "
                                     "output dimension of the fr_encoder. Found {} and {}, "
                                     "respectively.".format(fr_decoder.get_input_dim(),
                                                            fr_encoder.get_output_dim()))
        if fr_decoder.get_output_dim() != vocab.get_vocab_size("fr"):
            raise ConfigurationError("The output dimension of the fr_decoder must match the "
                                     "size of the French vocabulary. Found {} and {}, "
                                     "respectively.".format(fr_decoder.get_output_dim(),
                                                            vocab.get_vocab_size("fr")))

        self.en_vocab_size = vocab.get_vocab_size("en")
        self.fr_vocab_size = vocab.get_vocab_size("fr")
        self.en_field_embedder = en_field_embedder or TextFieldEmbedder()
        self.fr_field_embedder = fr_field_embedder or TextFieldEmbedder()
        self.en_encoder = en_encoder
        self.fr_encoder = fr_encoder
        self.fr_decoder = fr_decoder

        # Used for prepping the translation primer
        # (initialization of the French word-level encoder's hidden state).
        #
        # If the French word-level encoder is an LSTM, the hidden state as well as the
        # cell state must be initialized.
        #
        # Also, hidden states that prime translation via this encoder must be duplicated
        # across by number of layers it has.
        self._en_encoder_hidden_size = self.en_encoder._module.hidden_size
        self._fr_encoder_is_lstm = isinstance(self.fr_encoder._module, torch.nn.LSTM)
        self._fr_encoder_num_layers = self.fr_encoder._module.num_layers

        # Trains to maximize likelihood of translations.
        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self,
                en: Dict[str, torch.LongTensor],
                fr: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        output_dict = {}

        # Reverse the utterance before embedding.
        en_max_seq_len = en['tokens'].size()[-1]
        en_reversed_indices = torch.linspace(en_max_seq_len - 1, 0, en_max_seq_len).long()
        en_reversed_indices = en_reversed_indices.to(en['tokens'].device)  # CPU/GPU invariant.
        en_reversed_utterance = en['tokens'].index_select(-1, en_reversed_indices)
        assert(en['tokens'].equal(en_reversed_utterance.index_select(-1, en_reversed_indices)))
        en['tokens'] = en_reversed_utterance

        # Embed and encode the English utterance.
        # Results in a single vector representing the utterance.
        embedded_en_utterance = self.en_field_embedder(en)
        en_utterance_mask = util.get_text_field_mask(en)
        encoded_en_utterance = self.en_encoder(embedded_en_utterance, en_utterance_mask)

        # Prep the hidden state initialization of the word-level French LSTM.
        # Shape (no cell state): (num_layers, batch, en_hidden_size)
        # Shape (with cell state): Tuple of (num_layers, batch, en_hidden_size)'s
        fr_translation_primer = encoded_en_utterance.unsqueeze(0)
        fr_translation_primer = fr_translation_primer.expand(
            self._fr_encoder_num_layers,
            -1,  # Inferred from the other two.
            self._en_encoder_hidden_size
        )
        if self._fr_encoder_is_lstm:
            fr_translation_primer = (fr_translation_primer,
                                     torch.zeros_like(fr_translation_primer))

        # Embed and encode the French utterance.
        # Results in several vectors representing the utterance.
        # Shape: (batch, sequence_length, fr_hidden_size)
        embedded_fr_utterance = self.fr_field_embedder(fr)
        fr_utterance_mask = util.get_text_field_mask(fr)
        encoded_fr_utterance = self.fr_encoder(embedded_fr_utterance, fr_utterance_mask,
                                               hidden_state=fr_translation_primer)

        # Logits are likelihoods of each word in the vocabulary being the correct
        # word at that time step.
        # Shape: (batch_size x fr_max_seq_len x fr_encoder_hidden_size)
        logits = self.fr_decoder(encoded_fr_utterance)
        output_dict["logits"] = logits

        # Flatten predictions and compute loss.
        # Shape(s): Logits - (batch_size x fr_max_seq_len, fr_vocab_size)
        #           Targets - (batch_size x fr_max_seq_len)
        batch_size = logits.size(0)
        fr_max_seq_len = logits.size(1)
        logits = logits.view(batch_size * fr_max_seq_len, -1)
        targets = fr['tokens'].view(batch_size * fr_max_seq_len)
        output_dict["loss"] = self.loss(logits, targets)
        
        return output_dict

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'EnglishToFrenchEncoderDecoder':
        en_embedder_params = params.pop("en_field_embedder")
        fr_embedder_params = params.pop("fr_field_embedder")
        en_field_embedder = TextFieldEmbedder.from_params(vocab, en_embedder_params)
        fr_field_embedder = TextFieldEmbedder.from_params(vocab, fr_embedder_params)
        en_encoder = Seq2VecEncoder.from_params(params.pop("en_encoder"))
        fr_encoder = Seq2SeqEncoder.from_params(params.pop("fr_encoder"))
        fr_decoder = FeedForward.from_params(params.pop("fr_decoder"))
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))
        return cls(vocab=vocab,
                   en_field_embedder=en_field_embedder,
                   fr_field_embedder=fr_field_embedder,
                   en_encoder=en_encoder,
                   fr_encoder=fr_encoder,
                   fr_decoder=fr_decoder,
                   initializer=initializer,
                   regularizer=regularizer)
