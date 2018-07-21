from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.attention import BilinearAttention
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from overrides import overrides

# This is largely based on AllenNLP's general Seq2Seq encoder-decoder:
# https://github.com/allenai/allennlp/blob/master/allennlp/models/encoder_decoders/simple_seq2seq.py
#
# but offers more flexibility. Maybe I'll subclass this module when they've addressed their TODOs.


@Model.register("sequence_to_sequence")
class SequenceToSequence(Model):
    """
    Base class for sequence-to-sequence models.
    """
    DECODERS = {"rnn": torch.nn.RNN, "lstm": torch.nn.LSTM, "gru": torch.nn.GRU}
    def __init__(self,
                 # Vocabluary.
                 vocab: Vocabulary,
                 source_namespace: str,
                 target_namespace: str,

                 # Embeddings.
                 source_field_embedder: TextFieldEmbedder,
                 target_embedder: Embedding,

                 # Encoders and Decoders.
                 encoder: Seq2SeqEncoder,
                 decoder_type: str,
                 output_projection_layer: FeedForward,

                 # Hyperparamters and flags.
                 decoder_attention_function: BilinearAttention = None,
                 decoder_is_bidirectional: bool = False,
                 decoder_num_layers: int = 1,
                 apply_attention: Optional[bool] = False,
                 max_decoding_steps: int = 100,
                 scheduled_sampling_ratio: float = 0.4,

                 # Logistical.
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        if encoder.get_input_dim() != source_field_embedder.get_output_dim():
            raise ConfigurationError("The input dimension of the encoder must match the embedding"
                                     "size of the source_field_embedder. Found {} and {}, respectively."
                                     .format(encoder.get_input_dim(),
                                             source_field_embedder.get_output_dim()))
        if output_projection_layer.get_output_dim() != vocab.get_vocab_size("fr"):
            raise ConfigurationError("The output dimension of the output_projection_layer must match the "
                                     "size of the French vocabulary. Found {} and {}, "
                                     "respectively.".format(output_projection_layer.get_output_dim(),
                                                            vocab.get_vocab_size("fr")))
        if decoder_type not in SequenceToSequence.DECODERS:
            raise ConfigurationError("Unrecognized decoder option '{}'".format(decoder_type))

        # For dealing with input.
        self.source_vocab_size = vocab.get_vocab_size(source_namespace)
        self.target_vocab_size = vocab.get_vocab_size(target_namespace)
        self.source_field_embedder = source_field_embedder or TextFieldEmbedder()
        self.encoder = encoder

        # For dealing with / producing output.
        self.target_vocab_size = vocab.get_vocab_size(target_namespace)
        self.target_embedder = target_embedder
        self.decoder = SequenceToSequence.DECODERS[decoder_type](
                target_embedder.output_dim,  # Input size.
                encoder.get_output_dim(),  # Hidden size.
                num_layers=decoder_num_layers,
                batch_first=True,
                bias=True,
                bidirectional=decoder_is_bidirectional
        )
        self.output_projection_layer = output_projection_layer
        self.apply_attention = apply_attention
        self.decoder_attention_function = decoder_attention_function or BilinearAttention(
                matrix_dim=encoder.get_output_dim(),
                vector_dim=self.decoder.hidden_size
        )

        # Hyperparameters.
        self._max_decoding_steps = max_decoding_steps
        self._scheduled_sampling_ratio = scheduled_sampling_ratio

        # Used for prepping the translation primer (initialization of the target word-level
        # encoder's hidden state).
        #
        # If the decoder is an LSTM, both hidden states and cell states must be initialized.
        # Also, hidden states that prime translation via this encoder must be duplicated
        # across by number of layers they has.
        self._decoder_is_lstm = isinstance(self.decoder, torch.nn.LSTM)
        self._decoder_num_layers = decoder_num_layers

        self._start_index = vocab.get_token_index(START_SYMBOL, "fr")
        self._batch_size = None

        initializer(self)

    @overrides
    def forward(self,
                source: Dict[str, torch.LongTensor],
                target: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        output_dict: dict = {}
        source = self.preprocess_input(source)

        # Embed and encode the source sequence.
        source_encoded = self.encode_input(source)
        source_mask = util.get_text_field_mask(source)
        batch_size = source_encoded.size(0)

        # Determine number of decoding steps. If training or computing validation, we decode
        # target_seq_len times and compute loss.
        if target:
            target_tokens = target['tokens']
            target_seq_len = target['tokens'].size()[1]
            num_decoding_steps = target_seq_len - 1
        else:
            num_decoding_steps = self.max_decoding_steps

        # Begin decoding the encoded source, swapping in predictions for ground truth at the
        # scheduled sampling rate.
        last_predictions = None
        step_logits, step_probabilities, step_predictions = [], [], []
        decoder_hidden = self.init_decoder_hidden_state(source_encoded)
        for timestep in range(num_decoding_steps):
            if self.training and torch.rand(1).item() >= self._scheduled_sampling_ratio:
                input_choices = target_tokens[:, timestep]
            else:
                if timestep == 0:  # Initialize decoding with the start token.
                    input_choices = source_mask.new_full((batch_size,),
                                                         fill_value=self._start_index)
                else:
                    input_choices = last_predictions
            decoder_input = self.prepare_decode_step_input(input_choices, decoder_hidden,
                                                           source_encoded, source_mask)
            if len(decoder_input.shape) < 3:
                decoder_input = decoder_input.unsqueeze(1)

            _, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

            # Probability distribution for what the next decoded class should be.
            output_projection = self.output_projection_layer(decoder_hidden[0][-1]
                                                             if self._decoder_is_lstm
                                                             else decoder_hidden[-1])
            step_logits.append(output_projection.unsqueeze(1))

            # Collect predicted classes and their probabilities.
            class_probabilities = F.softmax(output_projection, dim=-1)
            _, predicted_classes = torch.max(class_probabilities, 1)
            step_probabilities.append(class_probabilities.unsqueeze(1))
            step_predictions.append(predicted_classes.unsqueeze(1))
            last_predictions = predicted_classes

        import pdb; pdb.set_trace()
        logits = torch.cat(step_logits, 1)
        class_probabilities = torch.cat(step_probabilities, 1)
        all_predictions = torch.cat(step_predictions, 1)
        output_dict = {"logits": logits,
                       "class_probabilities": class_probabilities,
                       "predictions": all_predictions}
        if target:
            target_mask = util.get_text_field_mask(target)
            relevant_targets = target['tokens'][:, 1:].contiguous()
            relevant_mask = target_mask[:, 1:].contiguous()
            loss = util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)
            output_dict["loss"] = loss

        return output_dict

    def preprocess_input(self, source: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        """
        Perform any preprocessing on the input text field you like; returns the source unchanged
        by default.
        """
        return source

    def encode_input(self, source: Dict[str, torch.LongTensor]) -> Tuple[torch.FloatTensor,
                                                                         torch.FloatTensor]:
        """
        Encode the source utterance how you see fit, as long as you return a tuple of
        tensors.

        By default, embeds the source utterance and feeds it to the source encoder.

        Required shapes: (batch_size, sequence_length, target_decoder_input_size) x
                         (batch_size, target_decoder_input_size)
        """
        embedded_source_sequence = self.source_field_embedder(source)
        source_sequence_mask = util.get_text_field_mask(source)
        encoded_source_sequence = self.encoder(embedded_source_sequence, source_sequence_mask)
        return encoded_source_sequence

    def init_decoder_hidden_state(self, source_encoded: torch.FloatTensor) -> torch.FloatTensor:
        """
        Prep the hidden state initialization of the word-level Target decoder any way
        you like.

        By default, uses only the final hidden state of the encoded source.

        Required shape: (batch_size, num_decoder_layers, encoder_hidden_size)
        """
        target_translation_primer = source_encoded.unsqueeze(0)
        encoder_hidden_size = source_encoded.size()[-1]
        target_translation_primer = target_translation_primer.expand(
                self._decoder_num_layers, -1, encoder_hidden_size).contiguous()
        assert target_translation_primer.is_contiguous()
        if self._decoder_is_lstm:
            target_translation_primer = (target_translation_primer,
                                         torch.zeros_like(target_translation_primer))
            assert target_translation_primer[0].is_contiguous()
            assert target_translation_primer[1].is_contiguous()

        return target_translation_primer

    def prepare_decode_step_input(self,
                                  input_indices: torch.LongTensor,
                                  decoder_hidden_state: torch.LongTensor,
                                  encoder_outputs: torch.LongTensor,
                                  encoder_outputs_mask: torch.LongTensor) -> torch.LongTensor:
        """
        Prepares the current timestep input for the decoder, embedding the input and applying the
        default attention (BiLinearAttention) if attention was enabled.
        """
        # input_indices : (batch_size,)  since we are processing these one timestep at a time.
        # (batch_size, target_embedding_dim)
        embedded_input = self.target_embedder(input_indices)
        if self.apply_attention:
            # encoder_outputs : (batch_size, input_sequence_length, encoder_output_dim)
            # Ensuring mask is also a FloatTensor. Or else the multiplication within attention will
            # complain.
            encoder_outputs_mask = encoder_outputs_mask.float()
            # (batch_size, input_sequence_length)
            input_weights = self.decoder_attention_function(decoder_hidden_state, encoder_outputs,
                                                            encoder_outputs_mask)
            # (batch_size, encoder_output_dim)
            attended_input = util.weighted_sum(encoder_outputs, input_weights)
            # (batch_size, encoder_output_dim + target_embedding_dim)
            return torch.cat((attended_input, embedded_input), -1)
        else:
            return embedded_input


# TODO: from_params
