from torch import nn, Tensor

from common.tensor_types import FloatT


class ScaledEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        self._init_embedding_weight()

    def _init_embedding_weight(self):
        nn.init.normal_(self.weight, mean=0.0, std=self.embedding_dim ** -0.5)

    def forward(self, inputs: Tensor) -> Tensor:
        # Create binary mask of size [batch_size, length]
        embeddings: FloatT = super(ScaledEmbedding, self).forward(inputs)

        mask = (inputs != 0.0).type(embeddings.dtype)

        embeddings *= mask.unsqueeze(-1)
        # Scale embedding by the sqrt of the hidden size
        embeddings *= self.embedding_dim ** 0.5

        return embeddings