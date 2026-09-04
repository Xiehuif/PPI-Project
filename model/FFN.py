import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

try:
    from xformers.ops import swiglu
    useXformers = True
except:
    useXformers = False

@torch.compile
class SwiGLUFFN(nn.Module):

    def __init__(
        self,
        inputDimension,
        hiddenDimension,
        dropout: float = 0.0,
        outputDimension: int = 0
    ):
        if outputDimension == 0:
            outputDimension = inputDimension
        super().__init__()
        # gate & fc net
        self.linearProjection = nn.Linear(inputDimension, hiddenDimension, bias=False)
        self.gateProjection = nn.Linear(inputDimension, hiddenDimension, bias=False)
        # activation
        self.activation = nn.SiLU()
        # output
        self.outputProjection = nn.Linear(hiddenDimension, outputDimension, bias=False)
        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        if useXformers:
            # print('Use Xformers')
            return self._fusedForward(input)
        else:
            return self._eagerForward(input)

    def _eagerForward(self, input):
        fcOut = self.linearProjection(input)
        gate = self.gateProjection(input)
        return self.dropout(self.outputProjection(self.activation(fcOut) * gate))

    @torch.compiler.disable
    def _fusedForward(self, input):
        gluOutput = swiglu(
            input,
            w1=self.linearProjection.weight, w2=self.gateProjection.weight, w3=self.outputProjection.weight,
            b1=None, b2=None, b3=None
        )
        return self.dropout(gluOutput)
