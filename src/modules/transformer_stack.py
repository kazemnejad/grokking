from typing import Optional, List

import torch
from allennlp.common import Registrable
from overrides import overrides
from torch import nn

from common import Lazy
from common import Params
from common.config import HoconConfig
from modules.t5 import (
    T5Attention,
    T5DecoderStack,
    T5LayerNorm,
    T5LayerFF,
    T5Block,
    T5LayerSelfAttention, T5StackOutput, T5Stack,
)
class TransformerStack(Registrable):
    pass

@TransformerStack.register("t5_decoder_stack")
class DecoderStack(T5DecoderStack, TransformerStack):
    def __init__(
            self,
            num_blocks: int,
            hidden_size: int = -1,
            self_attention: Lazy[T5Attention] = Lazy(T5Attention),
            final_layer_norm: Optional[T5LayerNorm] = None,
            feed_forward: Lazy[T5LayerFF] = Lazy(T5LayerFF),
            dropout: float = 0.1,
    ):
        blocks: List[T5Block] = []
        for i in range(num_blocks):
            block = T5Block(
                attention=T5LayerSelfAttention(
                    self_attention=self_attention.construct(
                        is_decoder=True,
                        has_relative_attention_bias=False,
                        relative_attention_num_buckets=False,
                    )
                ),
                cross_attention=None,
                ff=feed_forward.construct(),
            )
            blocks.append(block)

        super().__init__(
            None, blocks, final_layer_norm=final_layer_norm, dropout=dropout
        )


@TransformerStack.register("pt_encoder_stack")
class PytorchEncoderStack(nn.Module, TransformerStack):
    """
    Implements a stacked self-attention encoder similar to the Transformer
    architecture in [Attention is all you Need]
    (https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077).

    This class adapts the Transformer from torch.nn for use in AllenNLP. Optionally, it adds positional encodings.

    Registered as a `Seq2SeqEncoder` with name "pytorch_transformer".

    # Parameters

    input_dim : `int`, required.
        The input dimension of the encoder.
    num_layers : `int`, required.
        The number of stacked self attention -> feedforward -> layer normalisation blocks.
    feedforward_hidden_dim : `int`, required.
        The middle dimension of the FeedForward network. The input and output
        dimensions are fixed to ensure sizes match up for the self attention layers.
    num_attention_heads : `int`, required.
        The number of attention heads to use per layer.
    positional_encoding : `str`, optional, (default = `None`)
        Specifies the type of positional encodings to use. Your options are
         * `None` to have no positional encodings.
         * `"sinusoidal"` to have sinusoidal encodings, as described in https://api.semanticscholar.org/CorpusID:13756489.
         * `"embedding"` to treat positional encodings as learnable parameters
        Without positional encoding, the self attention layers have no idea of absolute or relative
        position (as they are just computing pairwise similarity between vectors of elements),
        which can be important features for many tasks.
    positional_embedding_size : `int`, optional, (default = `512`)
        The number of positional embeddings.
    dropout_prob : `float`, optional, (default = `0.1`)
        The dropout probability for the feedforward network.
    activation : `str`, (default = `"relu"`)
        The activation function of intermediate layers. Must be either `"relu"` or `"gelu"`.
    """  # noqa

    def __init__(
            self,
            input_dim: int,
            num_layers: int,
            feedforward_hidden_dim: int = 2048,
            num_attention_heads: int = 8,
            dropout_prob: float = 0.1,
            causal_attention: bool = False,
            activation: str = "relu",
            hidden_size: int = -1,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_attention_heads,
            dim_feedforward=feedforward_hidden_dim,
            dropout=dropout_prob,
            activation=activation,
        )
        self._transformer = nn.TransformerEncoder(layer, num_layers)
        self._input_dim = input_dim

        self.causal_attention = causal_attention

        # initialize parameters
        # We do this before the embeddings are initialized so we get the default initialization for the embeddings.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @property
    def hidden_size(self) -> int:
        return self._input_dim

    @overrides
    def forward(
            self,
            inputs_embeds: torch.Tensor,
            output_attentions: bool = True,
            output_all_hidden_states: bool = True,
            padding_mask: torch.BoolTensor = None,
    ):
        output = inputs_embeds
        batch_size, seq_len = output.size()[:-1]
        # For some reason the torch transformer expects the shape (sequence, batch, features), not the more
        # familiar (batch, sequence, features), so we have to fix it.
        output = output.permute(1, 0, 2)

        # For some other reason, the torch transformer takes the mask backwards.
        if padding_mask is not None:
            padding_mask = ~padding_mask

        attention_mask = None
        if self.causal_attention:
            attention_mask = generate_square_subsequent_mask(seq_len)
            attention_mask = attention_mask.to(output.device)

        output = self._transformer(
            output, mask=attention_mask, src_key_padding_mask=padding_mask
        )
        hidden_state = output.permute(1, 0, 2)

        output = T5StackOutput(last_hidden_state=hidden_state)

        return output


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


if __name__ == "__main__":
    params = """{
        type = pt_encoder_stack
        input_dim = 128
        num_layers = 2
        feedforward_hidden_dim = 512
        num_attention_heads = 4
        dropout_prob = 0.0
        causal_attention = true
    }
    """
    params = HoconConfig.from_str(params).to_dict()
    ds = TransformerStack.from_params(Params(params))
    # ds = Lazy(DecoderStack, constructor_extras=params).construct(hidden_size=16)
    print()
