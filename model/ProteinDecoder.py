import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from model.FFN import SwiGLUFFN
from model.ProteinEmbedder import EmbeddingLayer, MultiHeadSelfAttention
from collections import OrderedDict


@torch.compile
class QKVSymmetricProjection(nn.Module):
    """
    Computes QKV Projection through a packed projection operation
    The same input dim for all the input tokens is required
    """

    def __init__(
            self,
            inputDimension: int,
            headNumber: int,
            headDimension: int
    ):
        super().__init__()
        self.headNumber = headNumber
        self.headDimension = headDimension
        self.qkvProjection = nn.Linear(inputDimension, self.headNumber * self.headDimension * 3, bias=False)

    def forward(self, inputA, inputB):
        # Step 1. Pack the tensor and record their token length
        # TokenLengthA & TokenLengthB
        lenA = inputA.size(1)
        lenB = inputB.size(1)

        # TokenLengthA + TokenLengthB = TokenLength
        # (N, TokenLength, HiddenDim)
        inputConcat = torch.cat((inputA, inputB), dim=1)
        qkvOutput = self.qkvProjection(inputConcat)
        query, key, value = torch.chunk(qkvOutput, 3, dim=-1)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head

        # (N, TokenLength, HiddenDim) -> (N, TokenLength, headNumber, headDimension)
        # -> (N, headNumber, TokenLength, headDimension) -> ((...,...,TokenLengthA,...),(...,...,TokenLengthB,...))

        queryA, queryB = query.unflatten(-1, [self.headNumber, self.headDimension]).transpose(1, 2).split([lenA, lenB],
                                                                                                          -2)
        keyA, keyB = key.unflatten(-1, [self.headNumber, self.headDimension]).transpose(1, 2).split([lenA, lenB], -2)
        valueA, valueB = value.unflatten(-1, [self.headNumber, self.headDimension]).transpose(1, 2).split([lenA, lenB],
                                                                                                          -2)

        return queryA, queryB, keyA, keyB, valueA, valueB, lenA, lenB


@torch.compile
class MultiHeadSymmetricSelfAttention(nn.Module):
    """
    Computes multi-head attention. Symmetry for two input.
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
        assert hiddenDimension % headNumber == 0
        self.headDimension = hiddenDimension // headNumber
        self.outputProjection = nn.Linear(hiddenDimension, inputDimension, bias=False)
        self.qkvProjection = QKVSymmetricProjection(inputDimension, headNumber, self.headDimension)

    def forward(
            self,
            inputA: Tensor,
            inputB: Tensor
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            inputA (torch.Tensor): shape (N, TokenLengthA, InputDim)
            inputB (torch.Tensor): shape (N, TokenLengthB, InputDim)

        Returns:
            output (torch.Tensor): output of shape (N, TokenLength, InputDim)
        """

        # Step 1. Apply input projection for query, key and value.
        queryA, queryB, keyA, keyB, valueA, valueB, lenA, lenB = self.qkvProjection(inputA, inputB)

        # Step 2. Run SDPA
        # (N, headNumber, TokenLengthA, headDimension) & (N, headNumber, TokenLengthB, headDimension)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            outputA = F.scaled_dot_product_attention(
                queryA, keyA, valueA, dropout_p=self.dropout if self.training else 0.0, is_causal=False
            )
            outputB = F.scaled_dot_product_attention(
                queryB, keyB, valueB, dropout_p=self.dropout if self.training else 0.0, is_causal=False
            )
        # Concat output
        # (N, headNumber, TokenLength, headDimension)
        output = torch.cat((outputA, outputB), dim=-2)
        # (N, headNumber, TokenLength, headDimension) -> (N, TokenLength, headNumber, headDimension) -> (N, TokenLength, HiddenDim)
        output = output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, TokenLength, HiddenDim) -> (N, TokenLength, InputDim)
        output = self.outputProjection(output)
        # (N, TokenLengthA, InputDim) & (N, TokenLengthB, InputDim)
        return output.split([lenA, lenB], -2)


@torch.compile
class MultiHeadSymmetricCrossAttention(nn.Module):
    """
    Computes multi-head cross attention.
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

        assert hiddenDimension % headNumber == 0
        self.headDimension = hiddenDimension // headNumber
        self.outputProjection = nn.Linear(hiddenDimension, inputDimension, bias=False)
        self.qkvProjection = QKVSymmetricProjection(inputDimension, headNumber, self.headDimension)

    def forward(
            self,
            inputA: Tensor,
            inputB: Tensor
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            inputA (torch.Tensor): shape (N, TokenLengthA, InputDim)
            inputB (torch.Tensor): shape (N, TokenLengthB, InputDim)

        Returns:
            output (torch.Tensor): output of shape (N, TokenLength, InputDim)
        """
        # Step 1. Apply input projection for query, key and value.
        queryA, queryB, keyA, keyB, valueA, valueB, lenA, lenB = self.qkvProjection(inputA, inputB)

        # Step 2. Run SDPA
        # (N, headNumber, TokenLengthA, headDimension) & (N, headNumber, TokenLengthB, headDimension)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            outputA = F.scaled_dot_product_attention(
                queryA, keyB, valueB, dropout_p=self.dropout if self.training else 0.0, is_causal=False
            )
            outputB = F.scaled_dot_product_attention(
                queryB, keyA, valueA, dropout_p=self.dropout if self.training else 0.0, is_causal=False
            )
        # Concat output
        # (N, headNumber, TokenLength, headDimension)
        output = torch.cat((outputA, outputB), dim=-2)
        # (N, headNumber, TokenLength, headDimension) -> (N, TokenLength, headNumber, headDimension) -> (N, TokenLength, HiddenDim)
        output = output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, TokenLength, HiddenDim) -> (N, TokenLength, InputDim)
        output = self.outputProjection(output)
        # (N, TokenLengthA, InputDim) & (N, TokenLengthB, InputDim)
        return output.split([lenA, lenB], -2)


@torch.compile
class SymmetricLayerNormWithSkipConnection(nn.Module):

    def __init__(self, inputDimension: int):
        super().__init__()
        self.layerNorm = nn.RMSNorm(inputDimension)

    def forward(self, outputA: Tensor, outputB: Tensor, inputA: Tensor, inputB: Tensor):
        return self.layerNorm(inputA + outputA), self.layerNorm(inputB + outputB)


@torch.compile
class SymmetricIdentity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, inputA, inputB):
        return inputA, inputB


@torch.compile
class SymmetricIdentityWithSkipConnection(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputA: Tensor, outputB: Tensor, inputA: Tensor, inputB: Tensor):
        return inputA, inputB


class SymmetricDecoderLayer(nn.Module):

    def __init__(
            self,
            inputDimension: int,
            attentionFactor: int,
            ffnFactor: int,
            headNumber: int,
            dropout: float = 0.0,
            useSelfAttentionEncoder: bool = True
    ):
        """

        (N, TokenLength, InputDim) -> (N, TokenLength, InputDim)

        """
        attnDim = attentionFactor * inputDimension
        ffnDim = ffnFactor * inputDimension
        super().__init__()
        if useSelfAttentionEncoder:
            # SelfAttention: (N, TokenLengthA & TokenLengthB, InputDim) -> (N, TokenLengthA & TokenLengthB, InputDim)
            self.selfAttnBlock = MultiHeadSymmetricSelfAttention(inputDimension, attnDim, headNumber, dropout)
            # Norm & Add
            self.selfAttnNorm = SymmetricLayerNormWithSkipConnection(inputDimension)
        else:
            self.selfAttnBlock = SymmetricIdentity()
            self.selfAttnNorm = SymmetricIdentityWithSkipConnection()
        # CrossAttention: (N, TokenLengthA & TokenLengthB, InputDim) -> (N, TokenLengthA & TokenLengthB, InputDim)
        self.crossAttnBlock = MultiHeadSymmetricCrossAttention(inputDimension, attnDim, headNumber, dropout)
        # Norm & Add
        self.crossAttnNorm = SymmetricLayerNormWithSkipConnection(inputDimension)
        # FFN
        self.ffn = SwiGLUFFN(inputDimension, ffnDim, dropout)
        # Norm & Add
        self.outputNorm = SymmetricLayerNormWithSkipConnection(inputDimension)

    def forward(self, inputList: list[Tensor]):
        inputA, inputB = inputList
        # self attention
        selfAttnOutA, selfAttnOutB = self.selfAttnBlock(inputA, inputB)
        # add & norm
        selfAttnOutA, selfAttnOutB = self.selfAttnNorm(selfAttnOutA, selfAttnOutB, inputA, inputB)
        # cross attention
        crossAttnOutA, crossAttnOutB = self.crossAttnBlock(selfAttnOutA, selfAttnOutB)
        # add & norm
        crossAttnOutA, crossAttnOutB = self.crossAttnNorm(crossAttnOutA, crossAttnOutB, selfAttnOutA, selfAttnOutB)
        # ffn
        ffnOutA = self.ffn(crossAttnOutA)
        ffnOutB = self.ffn(crossAttnOutB)
        # add & norm
        ffnOutA, ffnOutB = self.outputNorm(ffnOutA, ffnOutB, crossAttnOutA, crossAttnOutB)
        return ffnOutA, ffnOutB

@torch.compile
class MultiHeadCrossAttention(nn.Module):
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
        self.kvProjection = nn.Linear(inputDimension, hiddenDimension * 2, bias=False)
        self.qProjection = nn.Linear(inputDimension, hiddenDimension, bias=False)
        assert hiddenDimension % headNumber == 0
        self.headDimension = hiddenDimension // headNumber
        self.outputProjection = nn.Linear(hiddenDimension, inputDimension, bias=False)

    def forward(
        self,
        inputKV: Tensor,
        inputQ: Tensor,
    ) -> tuple[Tensor, Tensor]:
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
        kvOutput = self.kvProjection(inputKV)
        key, value = torch.chunk(kvOutput, 2, dim=-1)
        query = self.qProjection(inputQ)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, TokenLength, HiddenDim) -> (N, TokenLength, headNumber, headDimension) -> (N, headNumber, TokenLength, headDimension)
        query = query.unflatten(-1, [self.headNumber, self.headDimension]).transpose(1, 2)
        # (N, TokenLength, HiddenDim) -> (N, TokenLength, headNumber, headDimension) -> (N, headNumber, TokenLength, headDimension)
        key = key.unflatten(-1, [self.headNumber, self.headDimension]).transpose(1, 2)
        # (N, TokenLength, HiddenDim) -> (N, TokenLength, headNumber, headDimension) -> (N, headNumber, TokenLength, headDimension)
        value = value.unflatten(-1, [self.headNumber, self.headDimension]).transpose(1, 2)
        # Get attention score (N, headNumber, TokenQ, TokenKV)
        attentionScore = query @ key.transpose(2, 3)

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

        return output, attentionScore


@torch.compile
class MultiHeadCrossAttentionWithoutAttentionMatrix(nn.Module):
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
        self.kvProjection = nn.Linear(inputDimension, hiddenDimension * 2, bias=False)
        self.qProjection = nn.Linear(inputDimension, hiddenDimension, bias=False)
        assert hiddenDimension % headNumber == 0
        self.headDimension = hiddenDimension // headNumber
        self.outputProjection = nn.Linear(hiddenDimension, inputDimension, bias=False)

    def forward(
        self,
        inputKV: Tensor,
        inputQ: Tensor,
    ) -> tuple[Tensor, Tensor]:
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
        kvOutput = self.kvProjection(inputKV)
        key, value = torch.chunk(kvOutput, 2, dim=-1)
        query = self.qProjection(inputQ)

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
class DecoderLayer(nn.Module):

    def __init__(
            self,
            inputDimension: int,
            attentionFactor: int,
            ffnFactor: int,
            headNumber: int,
            dropout: float = 0.0,
            useSelfAttentionEncoder: bool = True
    ):
        """

        (N, TokenLength, InputDim) -> (N, TokenLength, InputDim)

        """
        attnDim = attentionFactor * inputDimension
        ffnDim = ffnFactor * inputDimension
        super().__init__()
        if useSelfAttentionEncoder:
            # SelfAttention: (N, TokenLengthA & TokenLengthB, InputDim) -> (N, TokenLengthA & TokenLengthB, InputDim)
            self.selfAttnBlock = MultiHeadSelfAttention(inputDimension, attnDim, headNumber, dropout)
            # Norm & Add
            self.selfAttnNorm = nn.RMSNorm(inputDimension)
        else:
            self.selfAttnBlock = nn.Identity()
            self.selfAttnNorm = nn.Identity()
        # CrossAttention: (N, TokenLengthA & TokenLengthB, InputDim) -> (N, TokenLengthA & TokenLengthB, InputDim)
        self.crossAttnBlock = MultiHeadCrossAttentionWithoutAttentionMatrix(inputDimension, attnDim, headNumber, dropout)
        # Norm & Add
        self.crossAttnNorm = nn.RMSNorm(inputDimension)
        # FFN
        self.ffn = SwiGLUFFN(inputDimension, ffnDim, dropout)
        # Norm & Add
        self.outputNorm = nn.RMSNorm(inputDimension)

    def forward(self, inputList: tuple[Tensor, Tensor]):
        inputQ, inputKV = inputList
        # self attention
        selfAttnOut = self.selfAttnBlock(inputQ)
        # add & norm
        selfAttnNormOut = self.selfAttnNorm(selfAttnOut + inputQ)
        # cross attention
        # inputKV: Tensor,
        # inputQ: Tensor,
        # score : (N, headNumber, TokenQ, TokenKV)
        crossAttnOut = self.crossAttnBlock(inputKV=inputKV, inputQ=selfAttnNormOut)
        # add & norm
        crossAttnOut = self.crossAttnNorm(crossAttnOut + selfAttnNormOut)
        # ffn
        ffnOut = self.ffn(crossAttnOut)
        # add & norm
        ffnOut = self.outputNorm(ffnOut + crossAttnOut)
        return [ffnOut, inputKV]

class SymmetricTransformerDecoder(nn.Module):

    def __init__(
            self,
            headNumber: int,
            layerNumber: int,
            inputDimension: int,
            attnFactor: int = 1,
            ffnFactor: int = 3,
            dropout: float = 0.0,
            useEncoderModule = True
    ):
        """

        headNumber: 注意力头数
        layerNumber: 解码器层数
        inputDimension: 输入维度
        outputDimension: 输出维度
        attentionDimension: 决定注意力中间层的dim
        ffnFactor: Transformers所采用的FFN的中间维度

        """
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(layerNumber):
            self.layers.append(
                SymmetricDecoderLayer
                (inputDimension, attnFactor, ffnFactor, headNumber, dropout, useEncoderModule)
            )

        self.outputLayer = nn.Sequential(
            nn.Linear(inputDimension, ffnFactor * inputDimension, bias=False),
            nn.SiLU(),
            nn.Linear(ffnFactor * inputDimension, 1, bias=False)
        )

    def forward(self, proteinAEmbedding, proteinBEmbedding) -> Tensor:
        # 1, L, D
        proteinADecodeEmbedding, proteinBDecodeEmbedding = self.layers([proteinAEmbedding, proteinBEmbedding])

        # 1, D
        proteinA: torch.Tensor = proteinADecodeEmbedding.mean(dim=1)
        proteinB: torch.Tensor = proteinBDecodeEmbedding.mean(dim=1)
        proteinSum = (proteinA + proteinB) / 2.0 # (N*L1*L2*hiddenDimFromGraphModule)

        # 1, 1
        protein = self.outputLayer(proteinSum) # (N*L1*L2*(2*hiddenDimFromGraphModule))

        return protein

@torch.compile
class QueryPoolingLayer(nn.Module):

    def __init__(
            self,
            inputDimension: int,
            attentionFactor: int,
            ffnFactor: int,
            headNumber: int,
            dropout: float = 0.0,
    ):
        """

        (N, TokenLength, InputDim) -> (N, TokenLength, InputDim)

        """
        attnDim = attentionFactor * inputDimension
        super().__init__()
        # CrossAttention: (N, TokenLengthA & TokenLengthB, InputDim) -> (N, TokenLengthA & TokenLengthB, InputDim)
        self.crossAttnBlock = MultiHeadCrossAttentionWithoutAttentionMatrix(inputDimension, attnDim, headNumber, dropout)
        # Norm & Add
        self.crossAttnNorm = nn.RMSNorm(inputDimension)
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(inputDimension, ffnFactor * inputDimension, bias=False),
            nn.SiLU(),
            nn.Linear(ffnFactor * inputDimension, inputDimension, bias=False)
        )
        # Norm & Add
        self.outputNorm = nn.RMSNorm(inputDimension)

    def forward(self, inputList: tuple[Tensor, Tensor]):
        inputQ, inputKV = inputList
        # cross attention
        # inputKV: Tensor,
        # inputQ: Tensor,
        # score : (N, headNumber, TokenQ, TokenKV)
        crossAttnOut = self.crossAttnBlock(inputKV=inputKV, inputQ=inputQ)
        # add & norm
        crossAttnOut = self.crossAttnNorm(crossAttnOut + inputQ)
        # ffn
        ffnOut = self.ffn(crossAttnOut)
        # add & norm
        ffnOut = self.outputNorm(ffnOut + crossAttnOut)
        return [ffnOut, inputKV]

@torch.compile
class QueryPooling(nn.Module):

    def __init__(
            self,
            inputDimension: int,
            attentionFactor: int,
            ffnFactor: int,
            headNumber: int,
            dropout: float = 0.0,
            layerNumber: int = 2,
    ):
        super().__init__()
        self.normQuery = nn.Parameter(torch.rand(1, 1, inputDimension).bfloat16())
        self.poolingLayers = nn.Sequential()
        for i in range(layerNumber):
            self.poolingLayers.append(QueryPoolingLayer(inputDimension, attentionFactor, ffnFactor, headNumber, dropout))

    def forward(self, inputTokens: Tensor):
        inputList = [self.normQuery, inputTokens]
        return self.poolingLayers(inputList)[0] # （1， 1， D）

@torch.compile
class InterfacesProjector(nn.Module):

    def __init__(self,
            headNumber: int,
            inputDimension: int,
            attnFactor: int = 1,
            ffnFactor: int = 3,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.decoderLayer = DecoderLayer(inputDimension, attnFactor, ffnFactor, headNumber, dropout)
        self.ffnProjector = nn.Sequential(
            nn.Linear(inputDimension, ffnFactor * inputDimension, bias=False),
            nn.SiLU(),
            nn.Linear(ffnFactor * inputDimension, 4, bias=False),
        )

    def forward(self, inputTokens: Tensor):
        queryOut, _ = self.decoderLayer(inputTokens)
        logitsProjection = self.ffnProjector(queryOut)
        return logitsProjection


class PlainTransformerDecoder(nn.Module):

    def __init__(
            self,
            headNumber: int,
            layerNumber: int,
            inputDimension: int,
            attnFactor: int = 1,
            ffnFactor: int = 3,
            dropout: float = 0.0,
            useEncoderModule = True
    ):
        """

        headNumber: 注意力头数
        layerNumber: 解码器层数
        inputDimension: 输入维度
        outputDimension: 输出维度
        attentionDimension: 决定注意力中间层的dim
        ffnFactor: Transformers所采用的FFN的中间维度

        """
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(layerNumber):
            self.layers.append(
                DecoderLayer
                (inputDimension, attnFactor, ffnFactor, headNumber, dropout, useEncoderModule)
            )

        self.outputLayer = nn.Sequential(
            nn.Linear(inputDimension, ffnFactor * inputDimension, bias=False),
            nn.SiLU(),
            nn.Linear(ffnFactor * inputDimension, 1, bias=False)
        )

        self.monomerPoolingLayer = QueryPooling(
            inputDimension,
            attnFactor,
            ffnFactor,
            headNumber,
            dropout,
            2,
        )

        self.interfaceProjector = InterfacesProjector(
            headNumber,
            inputDimension,
            attnFactor,
            ffnFactor,
            dropout,
        )

    def FreezeCrossChainModule(self):
        self.layers.requires_grad_(True)
        self.interfaceProjector.requires_grad_(False)
        self.outputLayer.requires_grad_(True)
        self.monomerPoolingLayer.requires_grad_(True)

    def forward(self, proteinAEmbedding, proteinBEmbedding) -> Tensor:
        # 1, L, D
        # [1, HeadNum, L_A, L_B] * N
        proteinADecodeEmbedding, _ = self.layers([proteinAEmbedding, proteinBEmbedding])
        # [1, HeadNum, L_B, L_A] * N
        proteinBDecodeEmbedding, _ = self.layers([proteinBEmbedding, proteinAEmbedding])

        # interfaces
        proteinAInterface = self.interfaceProjector([proteinADecodeEmbedding, proteinBEmbedding])
        proteinBInterface = self.interfaceProjector([proteinBDecodeEmbedding, proteinAEmbedding])

        # 1, D
        """
        proteinA: torch.Tensor = proteinADecodeEmbedding.mean(dim=1)
        proteinB: torch.Tensor = proteinBDecodeEmbedding.mean(dim=1)
        proteinSum = (proteinA + proteinB) / 2.0 # (N*L1*L2*hiddenDimFromGraphModule)
        """

        proteinA = self.monomerPoolingLayer(proteinADecodeEmbedding).squeeze(0)
        proteinB = self.monomerPoolingLayer(proteinBDecodeEmbedding).squeeze(0)

        # 1, 1
        proteinAScore = self.outputLayer(proteinA) # (N*L1*L2*(2*hiddenDimFromGraphModule))
        proteinBScore = self.outputLayer(proteinB) #

        return [proteinAScore, proteinBScore], [proteinAInterface, proteinBInterface]

    def GetInteractionScore(self, proteinAEmbedding, proteinBEmbedding):
        proteinADecodeEmbedding, _ = self.layers([proteinAEmbedding, proteinBEmbedding])
        # [1, HeadNum, L_B, L_A] * N
        proteinBDecodeEmbedding, _ = self.layers([proteinBEmbedding, proteinAEmbedding])
        proteinA = self.monomerPoolingLayer(proteinADecodeEmbedding).squeeze(0)
        proteinB = self.monomerPoolingLayer(proteinBDecodeEmbedding).squeeze(0)
        proteinAScore = self.outputLayer(proteinA)  # (N*L1*L2*(2*hiddenDimFromGraphModule))
        proteinBScore = self.outputLayer(proteinB)  #
        return [[proteinAScore, proteinBScore]]

    def GetInterface(self, proteinAEmbedding, proteinBEmbedding):
        proteinADecodeEmbedding, _ = self.layers([proteinAEmbedding, proteinBEmbedding])
        # [1, HeadNum, L_B, L_A] * N
        proteinBDecodeEmbedding, _ = self.layers([proteinBEmbedding, proteinAEmbedding])

        # interfaces
        proteinAInterface = self.interfaceProjector([proteinADecodeEmbedding, proteinBEmbedding])
        proteinBInterface = self.interfaceProjector([proteinBDecodeEmbedding, proteinAEmbedding])
        return [[proteinAInterface, proteinBInterface]]