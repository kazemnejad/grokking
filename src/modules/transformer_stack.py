from typing import Optional, List

import allennlp.modules.transformer.t5


from common import Lazy
from common.config import HoconConfig
from common import Params
from modules.t5 import (
    T5Attention,
    T5DecoderStack,
    T5LayerNorm,
    T5LayerFF,
    T5Block,
    T5LayerSelfAttention,
)


class DecoderStack(T5DecoderStack):
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


if __name__ == "__main__":
    params = """{
        num_blocks = 2
        hidden_size = 16
        self_attention {
            key_value_proj_dim = 4
            num_heads = 4
            hidden_size = ${hidden_size}
            dropout = 0.1
        }
        feed_forward {
            ff_proj {
                type = relu
                hidden_size = ${hidden_size}
                ff_size = 64
                dropout = 0.1
            }
            dropout = 0.1
        }
        dropout = 0.1
    }
    """
    params = HoconConfig.from_str(params).to_dict()
    ds = DecoderStack.from_params(Params(params))
    # ds = Lazy(DecoderStack, constructor_extras=params).construct(hidden_size=16)
    print()
