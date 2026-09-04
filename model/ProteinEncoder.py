from collections import OrderedDict

import torch.nn
from torch import nn
from model.ProteinEmbedder import EmbeddingLayer

@torch.compile
class ThreeLayerCompressor(nn.Module):

    def __init__(self, inputDim, outputDim):
        super().__init__()
        interDim = int((inputDim + outputDim) / 2)
        self.linearProj0 = nn.Linear(inputDim, interDim, bias=False)
        self.act = nn.SiLU()
        self.norm0 = nn.RMSNorm(interDim)
        self.linearProj1 = nn.Linear(interDim, interDim, bias=False)
        self.norm1 = nn.RMSNorm(interDim)
        self.linearProj2 = nn.Linear(interDim, outputDim, bias=False)
        self.norm2 = nn.RMSNorm(outputDim)

    def forward(self, inputTokens):
        output = self.linearProj0(inputTokens)
        output = self.act(output)
        output = self.norm0(output)
        output  = self.linearProj1(output)
        output = self.act(output)
        output = self.norm1(output)
        output = self.linearProj2(output)
        output = self.act(output)
        output = self.norm2(output)
        return output



class BertEncoder(torch.nn.Module):

    def __init__(self, encoderLayerNumber: int, inputNodeDimension: int, hiddenNodeDimension: int,
                 ffnFactor: int, headNumber: int, dropout: float=0.0):
        super().__init__()
        inter_dim = int((inputNodeDimension + hiddenNodeDimension) / 2)
        print(hiddenNodeDimension)
        print(inter_dim)
        print(inputNodeDimension)
        self.inputDropout = nn.Dropout(p=dropout)
        self.embedRes = ThreeLayerCompressor(inputNodeDimension, hiddenNodeDimension)
        self.embeddingDropout = nn.Dropout(p=dropout)

        self.encoderLayers = torch.nn.Sequential()
        for i in range(encoderLayerNumber):
            self.encoderLayers.append(
                EmbeddingLayer(
                    inputDimension=hiddenNodeDimension,
                    hiddenDimension=hiddenNodeDimension * ffnFactor,
                    headNumber=headNumber,
                    dropout=dropout
                )
            )

    def forward(self, inputProtein):
        tokens = inputProtein
        tokens = self.inputDropout(tokens)
        tokensEmbedding = self.embedRes(tokens)
        tokensEmbedding = self.embeddingDropout(tokensEmbedding)
        encodingOutput = self.encoderLayers(tokensEmbedding)
        return encodingOutput
