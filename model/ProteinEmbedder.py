import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from model.FFN import SwiGLUFFN


@torch.compile
class MultiHeadSelfAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // nheads
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """

    def __init__(
        self,
        inputDimension: int,
        hiddenDimension: int,
        headNumber: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.headNumber = headNumber
        self.dropout = dropout
        self.qkvProjection = nn.Linear(inputDimension, hiddenDimension * 3, bias=False)
        assert hiddenDimension % headNumber == 0
        self.headDimension = hiddenDimension // headNumber
        self.outputProjection = nn.Linear(hiddenDimension, inputDimension, bias=False)

    def forward(
        self,
        input: Tensor
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            input (torch.Tensor): shape (N, TokenLength, InputDim)

        Returns:
            output (torch.Tensor): output of shape (N, TokenLength, InputDim)
        """
        # Step 1. Apply input projection
        qkvOutput = self.qkvProjection(input)
        query, key, value = torch.chunk(qkvOutput, 3, dim=-1)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, TokenLength, HiddenDim) -> (N, TokenLength, headNumber, headDimension) -> (N, headNumber, TokenLength, headDimension)
        query = query.unflatten(-1, [self.headNumber, self.headDimension]).transpose(1, 2)
        # (N, TokenLength, HiddenDim) -> (N, TokenLength, headNumber, headDimension) -> (N, headNumber, TokenLength, headDimension)
        key = key.unflatten(-1, [self.headNumber, self.headDimension]).transpose(1, 2)
        # (N, TokenLength, HiddenDim) -> (N, TokenLength, headNumber, headDimension) -> (N, headNumber, TokenLength, headDimension)
        value = value.unflatten(-1, [self.headNumber, self.headDimension]).transpose(1, 2)

        # Step 3. Run SDPA
        # (N, headNumber, TokenLength, headDimension)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            output = F.scaled_dot_product_attention(
                query, key, value, dropout_p=self.dropout if self.training else 0.0, is_causal=False
            )
        # (N, headNumber, TokenLength, headDimension) -> (N, TokenLength, headNumber, headDimension) -> (N, TokenLength, HiddenDim)
        output = output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, TokenLength, HiddenDim) -> (N, TokenLength, InputDim)
        output = self.outputProjection(output)

        return output

@torch.compile
class EmbeddingLayer(nn.Module):

    def __init__(
        self,
        inputDimension: int,
        hiddenDimension: int,
        headNumber: int,
        dropout: float = 0.0
    ):
        """

        (N, TokenLength, InputDim) -> (N, TokenLength, InputDim)

        """
        super().__init__()
        # attention: (N, TokenLength, InputDim) -> (N, TokenLength, InputDim)
        print(f"input: {inputDimension}")
        print(f"head: {headNumber}")
        self.attentionBlock = MultiHeadSelfAttention(inputDimension, inputDimension, headNumber, dropout)
        # norm & add
        self.attentionLayerNorm = nn.RMSNorm(inputDimension)
        # FFN
        self.ffn = SwiGLUFFN(inputDimension, hiddenDimension, dropout)
        # norm & add
        self.outputLayerNorm = nn.RMSNorm(inputDimension)

    def forward(self, input: Tensor):
        # attention
        attentionOutput = self.attentionBlock(input)
        # add & norm
        ffnInput = self.attentionLayerNorm(attentionOutput + input)
        # ffn
        ffnOutput = self.ffn(ffnInput)
        # add & norm & return
        return self.outputLayerNorm(ffnOutput + ffnInput)

class Embedder(nn.Module):

    def __init__(
        self,
        inputDimension: int,
        hiddenDimension: int,
        headNumber: int,
        layerNumber: int,
        dropout: float = 0.0
    ):
        super().__init__()

        self.embeddingLayers = nn.Sequential()
        for i in range(layerNumber):
            self.embeddingLayers.append(
                EmbeddingLayer(
                    inputDimension=inputDimension,
                    hiddenDimension=hiddenDimension,
                    headNumber=headNumber,
                    dropout=dropout
                )
            )
        from model.ProteinDecoder import QueryPooling
        self.embeddingQuery = QueryPooling(inputDimension, 1, 1, headNumber, dropout)

    def forward(self, inputSeq):
        outputSeq = self.embeddingLayers(inputSeq)
        outputEmbedding = self.embeddingQuery(outputSeq).squeeze(0)
        return outputEmbedding

