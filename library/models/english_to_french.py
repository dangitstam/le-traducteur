from typing import Dict, Optional

from overrides import overrides
import torch

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.nn import InitializerApplicator, RegularizerApplicator

@Model.register("english_to_french")
class EnglishToFrenchEncoderDecoder(Model):
    """
    Replication of Sutskever et al.'s ``Sequence-to-Sequence`` model for the English to French task.

    English is encoded via a deep LSTM ``backwards``. The resulting hidden state then primes
    generation of the equivalent French utterance using another deep LSTM.
    """

    def __init__(self,
                 vocab : Vocabulary,
                 en_field_embedder : TextFieldEmbedder,
                 fr_field_embedder : TextFieldEmbedder,
                 # Only the last hidden state is needed from english.
                 en_encoder : Seq2VecEncoder,
                 # But each French word will have to be encoded and pushed through a decoder.
                 fr_encoder : Seq2SeqEncoder,
                 fr_decoder : FeedForward,
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
        # if fr_encoder.get_hid() != en_encoder.get_output_dim():
        #     raise ConfigurationError("The hidden dimension of the fr_encoder must match the "
        #                              "output dimension of the en_encoder. Found {} and {}, "
        #                              "respectively.".format(fr_encoder.get_hid(),
        #                                                     en_encoder.get_output_dim()))
        if fr_decoder.get_input_dim() != fr_encoder.get_output_dim():
            raise ConfigurationError("The input dimension of the fr_decoder must match the "
                                     "output dimension of the fr_encoder. Found {} and {}, "
                                     "respectively.".format(fr_decoder.get_input_dim(),
                                                            fr_encoder.get_output_dim()))
        # if fr_decoder.get_hid() != fr_encoder.get_output_dim():
        #     raise ConfigurationError("The input dimension of the fr_encoder must match the "
        #                              "hidden size of the en_encoder. Found {} and {}, "
        #                              "respectively.".format(fr_encoder.get_input_dim(),
        #                                                     fr_encoder.get_output_dim()))
        # if fr_decoder.get_output_dim() != vocab.get_vocab_size("fr"):
        #     raise ConfigurationError("The output dimension of the fr_decoder must match the "
        #                              "size of the French vocabulary. Found {} and {}, "
        #                              "respectively.".format(fr_decoder.get_output_dim(),
        #                                                     vocab.get_vocab_size("fr")))

        self.en_vocab_size = vocab.get_vocab_size("en")
        self.fr_vocab_size = vocab.get_vocab_size("fr")
        self.en_field_embedder = en_field_embedder or TextFieldEmbedder()
        self.fr_field_embedder = fr_field_embedder or TextFieldEmbedder()
        self.en_encoder = en_encoder
        self.fr_encoder = fr_encoder
        self.fr_decoder = fr_decoder

        # Trains on negative log likelihood to maximize likelihood of tranlsations.
        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self,
                en: Dict[str, torch.LongTensor],
                fr: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        output_dict = {}
        # Embed and encode the English utterance.
        # Results in a single vector representing the utterance.
        # Shape: (batch, en_hidden_size)
        embedded_en_utterance = self.en_field_embedder(en)
        en_utterance_mask = util.get_text_field_mask(en)
        encoded_en_utterance = self.en_encoder(embedded_en_utterance, en_utterance_mask)

        # Embed and encode the French utterance.
        # Results in several vectors representing the utterance.
        # Shape: (batch, sequence_length, fr_hidden_size)
        embedded_fr_utterance = self.fr_field_embedder(fr)
        fr_utterance_mask = util.get_text_field_mask(fr)
        encoded_fr_utterance = self.fr_encoder(embedded_fr_utterance,
                                               fr_utterance_mask,
                                               hidden_state=encoded_en_utterance
                                                            .unsqueeze(0))

        logits = self.fr_decoder(encoded_fr_utterance)
        output_dict["logits"] = logits

        # Flatten predictions and compute loss.
        # Shape(s): Logits - (batch_size x max_sequence_length, fr_vocab)
        #           Targets - (batch_size x max_sequence_length)
        batch_size = logits.size(0)
        max_sequence_length = logits.size(1)
        logits = logits.view(batch_size * max_sequence_length, -1)
        targets = fr['tokens'].view(batch_size * max_sequence_length)
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
