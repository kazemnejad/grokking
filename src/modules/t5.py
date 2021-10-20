"""
An implementation of [T5](https://api.semanticscholar.org/CorpusID:204838007), adapted from [HuggingFace]
(https://github.com/huggingface/transformers/blob/4c32f9f26e6a84f0d9843fec8757e6ce640bb44e/src/transformers/models/t5/modeling_t5.py).
"""  # noqa: E401

import logging
from typing import Optional, Tuple, List, Union, TYPE_CHECKING, NamedTuple

import torch
import torch.nn.functional as F
from allennlp.common import FromParams, Lazy, Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.modules import LayerNorm
from allennlp.modules.transformer.attention_module import (
    AttentionOutput,
)
from allennlp.modules.transformer import attention_module
from allennlp.modules.transformer.transformer_module import TransformerModule
from allennlp.modules.transformer.util import (
    get_extended_attention_mask,
    FloatT,
    BoolT,
)
from allennlp.nn.checkpoint import CheckpointWrapper
from allennlp.nn.parallel import DdpAccelerator
from torch import nn

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class T5LayerNorm(LayerNorm, FromParams):
    """T5-style layer norm does not have bias and does not subtract the mean."""

    def __init__(self, hidden_size: int = 512):
        # super().__init__(hidden_size)
        super(T5LayerNorm, self).__init__(hidden_size)


class T5FeedForwardProjection(TransformerModule, Registrable):
    def forward(self, hidden_states) -> FloatT:
        raise NotImplementedError


@T5FeedForwardProjection.register("relu")
class T5DenseReluDense(TransformerModule, FromParams):
    def __init__(
        self, hidden_size: int = 512, ff_size: int = 2048, dropout: float = 0.1
    ):
        super().__init__()
        self.wi = nn.Linear(hidden_size, ff_size, bias=False)
        self.wi.weight.data.normal_(mean=0.0, std=hidden_size ** -0.5)
        self.wo = nn.Linear(ff_size, hidden_size, bias=False)
        self.wo.weight.data.normal_(mean=0.0, std=ff_size ** -0.5)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size

    def forward(self, hidden_states) -> FloatT:
        hidden_states = self.wi(hidden_states)
        hidden_states = F.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


@T5FeedForwardProjection.register("gated-gelu")
class T5DenseGatedGeluDense(TransformerModule, FromParams):
    def __init__(
        self, hidden_size: int = 512, ff_size: int = 2048, dropout: float = 0.1
    ):
        super().__init__()
        self.wi_0 = nn.Linear(hidden_size, ff_size, bias=False)
        self.wi_0.weight.data.normal_(mean=0.0, std=hidden_size ** -0.5)
        self.wi_1 = nn.Linear(hidden_size, ff_size, bias=False)
        self.wi_1.weight.data.normal_(mean=0.0, std=hidden_size ** -0.5)
        self.wo = nn.Linear(ff_size, hidden_size, bias=False)
        self.wo.weight.data.normal_(mean=0.0, std=ff_size ** -0.5)
        self.dropout = nn.Dropout(dropout)
        from allennlp.nn import Activation

        self.gelu_act = Activation.by_name("gelu_new")()
        self.hidden_size = hidden_size

    def forward(self, hidden_states) -> FloatT:
        hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(TransformerModule, FromParams):
    _pretrained_mapping = {"DenseReluDense": "ff_proj"}

    def __init__(
        self,
        ff_proj: Optional[T5FeedForwardProjection] = None,
        layer_norm: Optional[T5LayerNorm] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ff_proj = ff_proj or T5DenseReluDense()
        self.layer_norm = layer_norm or T5LayerNorm(
            hidden_size=self.ff_proj.hidden_size
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states) -> FloatT:
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.ff_proj(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5Attention(attention_module.T5Attention):
    def _normalize(self) -> None:
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)

        if hasattr(self, "output"):
            nn.init.xavier_uniform_(self.output.weight)

        if hasattr(self, "relative_attention_bias"):
            self.relative_attention_bias.weight.data.normal_(
                mean=0.0, std=self.hidden_size ** -0.5
            )

    def _position_bias(
        self,
        position_bias: Optional[torch.Tensor],
        seq_lengths: Tuple[int, int, int],
        past_key_states: Optional[torch.Tensor],
        attention_scores: torch.Tensor,
    ) -> torch.Tensor:
        seq_length, real_seq_length, key_length = seq_lengths

        # if position_bias is None:
        #     if self.relative_attention_num_buckets is not None:
        #         position_bias = self.compute_bias(real_seq_length, key_length)
        #     else:
        #         position_bias = torch.zeros(
        #             (1, self.num_attention_heads, real_seq_length, key_length),
        #             device=attention_scores.device,
        #             dtype=attention_scores.dtype,
        #         )
        #
        #     # if key and values are already calculated
        #     # we want only the last query position bias
        #     if past_key_states is not None:
        #         position_bias = position_bias[:, :, -seq_length:, :]
        return torch.zeros(
            (1, self.num_attention_heads, real_seq_length, key_length),
            device=attention_scores.device,
            dtype=attention_scores.dtype,
        )


class T5LayerSelfAttentionOutput(NamedTuple):
    hidden_states: FloatT
    attn_key_value_state: Optional[Tuple[FloatT, FloatT]]
    attn_position_bias: FloatT
    attn_weights: Optional[FloatT] = None


class T5LayerSelfAttention(TransformerModule, FromParams):
    _pretrained_mapping = {"SelfAttention": "self_attention"}

    def __init__(
        self,
        self_attention: Optional[T5Attention] = None,
        layer_norm: Optional[T5LayerNorm] = None,
        dropout: float = 0.1,
        has_relative_attention_bias: bool = False,
    ):
        super().__init__()
        self.self_attention = self_attention or T5Attention(
            has_relative_attention_bias=has_relative_attention_bias
        )
        self.layer_norm = layer_norm or T5LayerNorm(
            hidden_size=self.self_attention.hidden_size
        )
        self.dropout = nn.Dropout(dropout)

    @property
    def hidden_size(self) -> int:
        return self.self_attention.hidden_size

    def forward(
        self,
        hidden_states: FloatT,
        attention_mask: Optional[torch.BoolTensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.BoolTensor] = None,
        past_key_value: Optional[Tuple[FloatT]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> T5LayerSelfAttentionOutput:
        normed_hidden_states = self.layer_norm(hidden_states)

        attention_output: AttentionOutput = self.self_attention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        hidden_states = hidden_states + self.dropout(attention_output.hidden_states)

        return T5LayerSelfAttentionOutput(
            hidden_states,
            attention_output.key_value_state,
            attention_output.position_bias,
            attention_output.attention_probs,
        )


class T5LayerCrossAttentionOutput(NamedTuple):
    hidden_states: FloatT
    attn_key_value_state: Optional[Tuple[FloatT, FloatT]]
    attn_position_bias: FloatT
    attn_weights: Optional[FloatT] = None


class T5LayerCrossAttention(TransformerModule, FromParams):
    _pretrained_mapping = {"EncDecAttention": "enc_dec_attention"}

    def __init__(
        self,
        enc_dec_attention: Optional[T5Attention] = None,
        layer_norm: Optional[T5LayerNorm] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.enc_dec_attention = enc_dec_attention or T5Attention(
            is_decoder=True,
            has_relative_attention_bias=False,
            is_cross_attention=True,
        )
        self.layer_norm = layer_norm or T5LayerNorm(
            hidden_size=self.enc_dec_attention.hidden_size
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: FloatT,
        key_value_states: Optional[FloatT],
        attention_mask: Optional[torch.BoolTensor] = None,
        position_bias: Optional[FloatT] = None,
        layer_head_mask: Optional[torch.BoolTensor] = None,
        past_key_value: Optional[Tuple[Tuple[FloatT]]] = None,
        use_cache: bool = False,
        query_length: int = None,
        output_attentions: bool = False,
    ) -> T5LayerCrossAttentionOutput:
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output: AttentionOutput = self.enc_dec_attention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = hidden_states + self.dropout(attention_output.hidden_states)

        return T5LayerCrossAttentionOutput(
            layer_output,
            attention_output.key_value_state,
            attention_output.position_bias,
            attention_output.attention_probs,
        )


KeyValueStates = Union[
    Tuple[FloatT, FloatT],  # without cross attention
    Tuple[FloatT, FloatT, FloatT, FloatT],  # with cross attention
]


class T5BlockOutput(NamedTuple):
    hidden_states: FloatT
    present_key_value_states: Optional[KeyValueStates]
    self_attn_weights: Optional[FloatT]
    self_attn_position_bias: Optional[FloatT]
    cross_attn_weights: Optional[FloatT] = None
    cross_attn_position_bias: Optional[FloatT] = None


class T5Block(TransformerModule, FromParams):
    def __init__(
        self,
        attention: Optional[T5LayerSelfAttention] = None,
        cross_attention: Optional[T5LayerCrossAttention] = None,
        ff: Optional[T5LayerFF] = None,
    ):
        super().__init__()
        self.layer = nn.ModuleList()
        self.layer.append(attention or T5LayerSelfAttention())

        if cross_attention is not None:
            assert (
                cross_attention.enc_dec_attention.is_decoder
                == attention.self_attention.is_decoder
            )
            self.layer.append(cross_attention)

        self.is_decoder = attention.self_attention.is_decoder
        self.layer.append(ff or T5LayerFF())

    @property
    def hidden_size(self) -> int:
        return self.layer[0].hidden_size

    def forward(
        self,
        hidden_states: FloatT,
        attention_mask: Optional[torch.BoolTensor] = None,
        position_bias: Optional[FloatT] = None,
        encoder_hidden_states: Optional[FloatT] = None,
        encoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_decoder_position_bias: Optional[FloatT] = None,
        layer_head_mask: Optional[torch.BoolTensor] = None,
        encoder_layer_head_mask: Optional[torch.BoolTensor] = None,
        past_key_value: Optional[KeyValueStates] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> T5BlockOutput:
        if past_key_value is not None:
            assert self.is_decoder, "Only decoder can use `past_key_values`"
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            error_message = (
                f"There should be {expected_num_past_key_values} past states. "
            )
            error_message += "2 (past / key) for self attention. "
            if expected_num_past_key_values == 4:
                error_message += "2 (past / key) for cross attention. "
            error_message += f"Got {len(past_key_value)} past key / value states"
            assert len(past_key_value) == expected_num_past_key_values, error_message

        self_attention_outputs: T5LayerSelfAttentionOutput = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=None if past_key_value is None else past_key_value[:2],
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = self_attention_outputs.hidden_states
        present_key_value_state: Optional[
            Tuple[FloatT, FloatT]
        ] = self_attention_outputs.attn_key_value_state

        # clamp inf values to enable fp16 training
        if torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs: T5LayerCrossAttentionOutput = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=encoder_layer_head_mask,
                past_key_value=None if past_key_value is None else past_key_value[2:],
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs.hidden_states
            if torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(
                    hidden_states, min=-clamp_value, max=clamp_value
                )

            # Combine self attn and cross attn key value states
            if (
                present_key_value_state is not None
                and cross_attention_outputs.attn_key_value_state is not None
            ):
                present_key_value_state: KeyValueStates = (  # type: ignore[no-redef]
                    present_key_value_state
                    + cross_attention_outputs.attn_key_value_state
                )

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)
        if torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        output = T5BlockOutput(
            hidden_states,
            present_key_value_state,
            self_attention_outputs.attn_weights,
            self_attention_outputs.attn_position_bias,
            cross_attn_weights=(
                None if not do_cross_attention else cross_attention_outputs.attn_weights
            ),
            cross_attn_position_bias=(
                None
                if not do_cross_attention
                else cross_attention_outputs.attn_position_bias
            ),
        )
        return output


class T5StackOutput(NamedTuple):
    last_hidden_state: FloatT
    past_key_values: Optional[List[KeyValueStates]] = None
    all_hidden_states: Optional[List[FloatT]] = None
    attentions: Optional[List[FloatT]] = None
    cross_attentions: Optional[List[FloatT]] = None


class T5Stack(TransformerModule, FromParams):
    _pretrained_mapping = {"embed_tokens": "token_embeddings", "block": "blocks"}

    def __init__(
        self,
        token_embeddings: nn.Embedding,
        blocks: List[T5Block],
        final_layer_norm: Optional[T5LayerNorm] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.is_decoder = blocks[0].is_decoder
        if not all(b.is_decoder == self.is_decoder for b in blocks):
            raise ConfigurationError("Found mismatched blocks in stack.")
        self.blocks = nn.ModuleList(blocks)
        self.token_embeddings = token_embeddings

        self.final_layer_norm = final_layer_norm or T5LayerNorm(
            hidden_size=self.hidden_size
        )
        self.dropout = nn.Dropout(dropout)

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def hidden_size(self) -> int:
        return self.blocks[0].hidden_size

    @staticmethod
    def get_head_mask(
        head_mask: Optional[torch.BoolTensor], num_hidden_layers: int
    ) -> BoolT:
        if head_mask is not None:
            # -> [num_hidden_layers x batch x num_heads x seq_length x seq_length]
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            assert (
                head_mask.dim() == 5
            ), f"head_mask.dim != 5, instead {head_mask.dim()}"
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    def forward(
        self,
        input_ids: Optional[torch.IntTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        encoder_hidden_states: Optional[FloatT] = None,
        encoder_attention_mask: Optional[torch.BoolTensor] = None,
        inputs_embeds: Optional[FloatT] = None,
        head_mask: Optional[torch.BoolTensor] = None,
        encoder_head_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[KeyValueStates] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_all_hidden_states: bool = False,
    ) -> T5StackOutput:
        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}inputs "
                f"and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You have to specify either {err_msg_prefix}inputs or {err_msg_prefix}inputs_embeds"
            )

        if inputs_embeds is None:
            assert (
                self.token_embeddings is not None
            ), "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.token_embeddings(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = (
            seq_length
            if past_key_values is None
            else past_key_values[0][0].shape[2] + seq_length
        )

        if use_cache is True:
            assert (
                self.is_decoder
            ), ":obj:`use_cache` can only be set to `True` if {} is used as a decoder".format(
                self
            )

        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size,
                mask_seq_length,
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        if (
            self.is_decoder
            and encoder_attention_mask is None
            and encoder_hidden_states is not None
        ):
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size,
                encoder_seq_length,
                device=inputs_embeds.device,
                dtype=torch.bool,
            )

        extended_attention_mask = get_extended_attention_mask(
            attention_mask, input_shape, inputs_embeds.dtype, is_decoder=self.is_decoder
        )

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.num_blocks)
        encoder_head_mask = self.get_head_mask(encoder_head_mask, self.num_blocks)
        present_key_value_states: Optional[List[KeyValueStates]] = (
            [] if use_cache else None
        )
        all_hidden_states: Optional[List[FloatT]] = (
            [] if output_all_hidden_states else None
        )
        all_attentions: Optional[List[FloatT]] = [] if output_attentions else None
        all_cross_attentions: Optional[List[FloatT]] = (
            [] if (output_attentions and self.is_decoder) else None
        )
        position_bias: Optional[FloatT] = None
        encoder_decoder_position_bias: Optional[FloatT] = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(
            zip(self.blocks, past_key_values or [None] * self.num_blocks)
        ):
            layer_head_mask = head_mask[i]
            encoder_layer_head_mask = encoder_head_mask[i]
            if output_all_hidden_states:
                all_hidden_states.append(hidden_states)  # type: ignore[union-attr]

            layer_outputs: T5BlockOutput = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=layer_head_mask,
                encoder_layer_head_mask=encoder_layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            # If the blocks were wrapped with a `CheckpointWrapper`, the output
            # may just be a raw tuple, not the NamedTuple that we want.
            if not isinstance(layer_outputs, T5BlockOutput):
                layer_outputs = T5BlockOutput(*layer_outputs)
            hidden_states = layer_outputs.hidden_states

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention weights),
            # (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            position_bias = layer_outputs.self_attn_position_bias
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs.cross_attn_position_bias
            if use_cache:
                present_key_value_states.append(layer_outputs.present_key_value_states)  # type: ignore
            if output_attentions:
                all_attentions.append(layer_outputs.self_attn_weights)  # type: ignore[union-attr]
                if self.is_decoder:
                    all_cross_attentions.append(layer_outputs.cross_attn_weights)  # type: ignore[union-attr]

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_all_hidden_states:
            all_hidden_states.append(hidden_states)  # type: ignore[union-attr]

        return T5StackOutput(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            all_hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class T5EncoderStack(T5Stack, FromParams):
    def __init__(
        self,
        token_embeddings: nn.Embedding,
        blocks: List[T5Block],
        final_layer_norm: Optional[T5LayerNorm] = None,
        dropout: float = 0.1,
    ):
        if any(b.is_decoder for b in blocks):
            raise ConfigurationError(
                "Found a decoder block in an encoder stack. This won't work."
            )

        super().__init__(
            token_embeddings,
            blocks,
            final_layer_norm=final_layer_norm,
            dropout=dropout,
        )

    @classmethod
    def basic_encoder(
        cls,
        token_embeddings: nn.Embedding,
        num_blocks: int = 6,
        block_self_attention: Lazy[T5Attention] = Lazy(T5Attention),
        final_layer_norm: Optional[T5LayerNorm] = None,
        block_ff: Lazy[T5LayerFF] = Lazy(T5LayerFF),
        dropout: float = 0.1,
        ddp_accelerator: Optional[DdpAccelerator] = None,
        checkpoint_wrapper: Optional[CheckpointWrapper] = None,
    ) -> "T5EncoderStack":
        if ddp_accelerator is not None:
            logger.info(
                "Initializing T5 encoder with DdpAccelerator %s", ddp_accelerator
            )
        blocks: List[T5Block] = []
        for i in range(num_blocks):
            block = T5Block(
                attention=T5LayerSelfAttention(
                    self_attention=block_self_attention.construct(
                        is_decoder=False, has_relative_attention_bias=(i == 0)
                    )
                ),
                cross_attention=None,
                ff=block_ff.construct(),
            )
            if checkpoint_wrapper is not None:
                block = checkpoint_wrapper.wrap_module(block)
            if ddp_accelerator is not None:
                block = ddp_accelerator.wrap_module(block)
            blocks.append(block)
        return cls(
            token_embeddings, blocks, final_layer_norm=final_layer_norm, dropout=dropout
        )


class T5DecoderStack(T5Stack, FromParams):
    def __init__(
        self,
        token_embeddings: nn.Embedding,
        blocks: List[T5Block],
        final_layer_norm: Optional[T5LayerNorm] = None,
        dropout: float = 0.1,
    ):
        if not all(b.is_decoder for b in blocks):
            raise ConfigurationError(
                "Found an encoder block in a decoder stack. This won't work."
            )

        super().__init__(
            token_embeddings,
            blocks,
            final_layer_norm=final_layer_norm,
            dropout=dropout,
        )

    @classmethod
    def basic_decoder(
        cls,
        token_embeddings: nn.Embedding,
        num_blocks: int = 6,
        block_self_attention: Lazy[T5Attention] = Lazy(T5Attention),
        block_cross_attention: Lazy[T5Attention] = Lazy(T5Attention),
        final_layer_norm: Optional[T5LayerNorm] = None,
        block_ff: Lazy[T5LayerFF] = Lazy(T5LayerFF),
        dropout: float = 0.1,
        ddp_accelerator: Optional[DdpAccelerator] = None,
        checkpoint_wrapper: Optional[CheckpointWrapper] = None,
    ) -> "T5DecoderStack":
        if ddp_accelerator is not None:
            logger.info(
                "Initializing T5 decoder with DdpAccelerator %s", ddp_accelerator
            )
        blocks: List[T5Block] = []
        for i in range(num_blocks):
            block = T5Block(
                attention=T5LayerSelfAttention(
                    self_attention=block_self_attention.construct(
                        is_decoder=True, has_relative_attention_bias=(i == 0)
                    )
                ),
                cross_attention=T5LayerCrossAttention(
                    enc_dec_attention=block_cross_attention.construct(
                        is_decoder=True,
                        has_relative_attention_bias=False,
                    )
                ),
                ff=block_ff.construct(),
            )
            if checkpoint_wrapper is not None:
                block = checkpoint_wrapper.wrap_module(block)
            if ddp_accelerator is not None:
                block = ddp_accelerator.wrap_module(block)
            blocks.append(block)
        return cls(
            token_embeddings, blocks, final_layer_norm=final_layer_norm, dropout=dropout
        )
