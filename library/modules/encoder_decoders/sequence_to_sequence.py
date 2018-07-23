from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
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

# TODO: Add more asserts so people don't do dumb shit
# TODO: Better docstrings.


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

        # Input size will either be the target embedding size or the target embedding size plus the
        # encoder hidden size to attend on the input.
        #
        # When making a custom attention function that uses neither of those input sizes, you will
        # have to define the decoder yourself.
        decoder_input_size = target_embedder.output_dim
        if apply_attention:
            decoder_input_size += encoder.get_output_dim()

        # Hidden size of the encoder and decoder should match.
        decoder_hidden_size = encoder.get_output_dim()
        self.decoder = SequenceToSequence.DECODERS[decoder_type](
                decoder_input_size,
                decoder_hidden_size,
                num_layers=decoder_num_layers,
                batch_first=True,
                bias=True,
                bidirectional=decoder_is_bidirectional
        )
        self.output_projection_layer = output_projection_layer
        self.apply_attention = apply_attention
        self.decoder_attention_function = decoder_attention_function or BilinearAttention(
                matrix_dim=encoder.get_output_dim(),
                vector_dim=encoder.get_output_dim()
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

        import pdb; pdb.set_trace()
        self._start_index = vocab.get_token_index(START_SYMBOL, target_namespace)
        self._end_index = vocab.get_token_index(END_SYMBOL, target_namespace)
        self._source_namespace = source_namespace
        self._target_namespace = target_namespace
        self._batch_size = None

        initializer(self)

    @overrides
    def forward(self,
                source: Dict[str, torch.LongTensor],
                target: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        output_dict: dict = {}
        source = self.preprocess_input(source)

        # Embed and encode the source sequence.
        source_sequence_encoded = self.encode_input(source)
        source_encoded = source_sequence_encoded[:, -1]
        source_mask = util.get_text_field_mask(source)
        batch_size = source_encoded.size(0)

        # Determine number of decoding steps. If training or computing validation, we decode
        # target_seq_len times and compute loss.
        if target:
            target_tokens = target['tokens']
            target_seq_len = target['tokens'].size(1)
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
                                                           source_sequence_encoded, source_mask)
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
        # pylint: disable=R0201
        return source

    def encode_input(self, source: Dict[str, torch.LongTensor]) -> Tuple[torch.FloatTensor,
                                                                         torch.FloatTensor]:
        """
        Encode the source utterance how you see fit, as long as you return a tuple of
        tensors.

        By default, embeds the source utterance and feeds it to the source encoder.
        Note that when subclassing this module, the decoder_hidden_size should be the same as
        the encoder's hidden size.

        Required shapes: (batch_size, sequence_length, decoder_hidden_size)
        """
        source_sequence_embedded = self.source_field_embedder(source)
        source_sequence_mask = util.get_text_field_mask(source)
        encoded_source_sequence = self.encoder(source_sequence_embedded, source_sequence_mask)
        return encoded_source_sequence

    def init_decoder_hidden_state(self, source_sequence_encoded: torch.FloatTensor) -> torch.FloatTensor:
        """
        Prep the hidden state initialization of the word-level Target decoder any way
        you like.

        By default, uses only the final hidden state of the encoded source.

        Required shape: (batch_size, num_decoder_layers, encoder_hidden_size)
        """
        decoder_primer = source_sequence_encoded.unsqueeze(0)
        decoder_primer = decoder_primer.expand(
                self._decoder_num_layers, -1, self.encoder.get_output_dim()
        ).contiguous()

        # If the decoder is an LSTM, we need to initialize a cell state.
        if self._decoder_is_lstm:
            decoder_primer = (decoder_primer, torch.zeros_like(decoder_primer))

        return decoder_primer

    def prepare_decode_step_input(self,
                                  input_indices: torch.LongTensor,
                                  decoder_hidden: torch.LongTensor,
                                  encoder_outputs: torch.LongTensor,
                                  encoder_outputs_mask: torch.LongTensor) -> torch.LongTensor:
        """
        Prepares the current timestep input for the decoder.

        By default, simply embeds and returns the input. If using attention, the default attention
        (BiLinearAttention) is applied to attend on the step input given the encoded source
        sequence and the previous hidden state.

        Parameters:
        -----------
        input_indices : torch.LongTensor
            Indices of either the gold inputs to the decoder or the predicted labels from the
            previous timestep.
        decoder_hidden : torch.LongTensor, optional (not needed if no attention)
            Output from the decoder at the last time step. Needed only if using attention.
        encoder_outputs : torch.LongTensor, optional (not needed if no attention)
            Encoder outputs from all time steps. Needed only if using attention.
        encoder_outputs_mask : torch.LongTensor, optional (not needed if no attention)
            Masks on encoder outputs. Needed only if using attention.
        """
        # input_indices : (batch_size,)  since we are processing these one timestep at a time.
        # (batch_size, target_embedding_dim)
        embedded_input = self.target_embedder(input_indices)
        if self.apply_attention:
            if isinstance(decoder_hidden, tuple):
                decoder_hidden = decoder_hidden[0]
            # encoder_outputs : (batch_size, input_sequence_length, encoder_output_dim)
            # Ensuring mask is also a FloatTensor. Or else the multiplication within attention will
            # complain.
            encoder_outputs_mask = encoder_outputs_mask.float()
            # (batch_size, input_sequence_length)
            input_weights = self.decoder_attention_function(decoder_hidden[-1], encoder_outputs,
                                                            encoder_outputs_mask)
            # (batch_size, encoder_output_dim)
            attended_input = util.weighted_sum(encoder_outputs, input_weights)
            # (batch_size, encoder_output_dim + target_embedding_dim)
            return torch.cat((attended_input, embedded_input), -1)
        else:
            return embedded_input

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, np.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for indices in predicted_indices:
            indices = list(indices)
            # Collect indices till the first END_SYMBOL.
            if self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]
            predicted_tokens = [self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                                for x in indices]
            all_predicted_tokens.append(predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict
